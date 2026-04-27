import asyncio
import logging
import time
from collections import deque
from concurrent.futures import Executor
from typing import Any, Awaitable

import numpy as np
import httpx
from scipy.signal import firwin, resample_poly

from asr.denoiser import RNNoiseFilter
from asr.engine import ASREngine
from asr.vad import VADDetector
from config import nacos_config as cfg

logger = logging.getLogger(__name__)


class _Resampler8kTo16k:
    """8kHz PCM16 → 16kHz PCM16 上采样器（预计算 FIR，避免每帧重算系数）。"""

    _up = 2
    _down = 1
    _h = firwin(63, 1.0 / _up, window=("kaiser", 5.0)) * _up

    def process(self, audio_bytes: bytes) -> bytes:
        if not audio_bytes:
            return b""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        out = resample_poly(samples, up=self._up, down=self._down, window=self._h)
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    def reset(self) -> None:
        return None


class StreamHandler:
    """
    通话全周期流处理器，每路 WebSocket 连接独立一个实例。

    流程：
    1. 停顿最大时长：VAD 静音超过 silence_max_ms → 判定句子结束 → 整句推 Intent
    2. 打断检测：interrupt_enabled=True 且连续说话 >= vad_interrupt_threshold_ms → POST interrupt_url
    3. 客户无应答：连接后超过 no_answer_timeout_ms 未收到有效文本 → POST event=no_answer
    4. 客户匹配时长：有文本但未命中意图，超过 match_timeout_ms → POST event=match_timeout

    P0 优化：
    - http_client: 外部注入的全局 httpx.AsyncClient，复用连接池
    - vad_executor: 专用线程池，低延迟，供 RNNoise + FSMN-VAD 使用
    - asr_executor: 专用线程池，高吞吐，供在线/离线 Paraformer 使用
    """

    def __init__(self, call_id: str, uuid: str, model_id: int, http_client: httpx.AsyncClient, vad_executor: Executor, asr_executor: Executor):
        self.call_id = call_id
        self.uuid = uuid
        self.model_id = model_id
        self._http_client = http_client
        self._vad_executor = vad_executor
        self._asr_executor = asr_executor

        # 初始化时先用全局配置，start 事件后由 load_conf() 覆盖
        self._silence_max_ms: int = cfg.get("silence_max_ms", 800)
        self._no_answer_timeout_ms: int = cfg.get("no_answer_timeout_ms", 10000)
        self._match_timeout_ms: int = cfg.get("match_timeout_ms", 15000)
        self._interrupt_enabled: bool = cfg.get("interrupt_enabled", True)
        self._interrupt_threshold_ms: int = cfg.get("vad_interrupt_threshold_ms", 2000)
        self._noise_gate_rnnoise_vad_prob: float = cfg.get("noise_gate_rnnoise_vad_prob", 0.45)
        self._noise_gate_min_speech_frames: int = cfg.get("noise_gate_min_speech_frames", 2)
        self._noise_gate_filter_chars: set[str] = set(cfg.get("noise_gate_filter_chars", ["嗯", "喂"]))
        self._noise_gate_lookback_ms: int = cfg.get("noise_gate_lookback_ms", 180)
        self._interrupt_ignore_start_ms: int = 0   # 开启后前 N ms 禁止打断
        self._word_count: int = 2
        self._question_similarity: float | None = None

        self._vad = VADDetector(threshold_ms=self._interrupt_threshold_ms)
        self._asr = ASREngine()
        self._denoiser = RNNoiseFilter()  # 每路连接独立实例，线程安全
        self._resampler = _Resampler8kTo16k() if cfg.get("audio_input_sample_rate", 16000) != 16000 else None
        self._paused = True

        # 当前句子文本缓冲
        self._sentence_parts: list[str] = []
        self._sentence_start_ms: int = 0
        self._sentence_epoch: int = 0

        # 全量转写片段
        self._all_segments: list[dict[str, Any]] = []
        self._elapsed_ms: int = 0
        self._total_samples: int = 0

        # 停顿检测状态
        self._in_speech: bool = False
        self._last_speech_end_ms: int | None = None

        # VAD 前置回看缓冲：解决首字丢失（VAD 检测滞后导致语音起始帧未送 ASR）
        self._audio_lookback: deque[bytes] = deque(maxlen=3)
        self._lookback_samples: int = 0
        self._noise_speech_streak: int = 0

        # 计时器
        self._no_answer_task: asyncio.Task | None = None
        self._match_timeout_task: asyncio.Task | None = None
        self._first_text_received: bool = False
        self._no_answer_sent: bool = False

        # 打断前置禁区：resume() 时记录起始时刻
        self._interrupt_allow_after: float = 0.0

    def start_timers(self) -> None:
        """连接建立后调用，启动无应答计时器。"""
        self._no_answer_task = asyncio.create_task(self._no_answer_timer())

    def _timer_start_delay_s(self) -> float:
        """统一计时器起始延迟窗：start 事件后的 ignore_start_seconds。"""
        return max(0.0, self._interrupt_allow_after - time.monotonic())

    def stop_timers(self) -> None:
        """连接断开或流程结束时调用。"""
        if self._no_answer_task:
            self._no_answer_task.cancel()
            self._no_answer_task = None
        if self._match_timeout_task:
            self._match_timeout_task.cancel()
            self._match_timeout_task = None

    def pause(self) -> None:
        """由 pause/stop 事件调用：暂停识别并重置状态。"""
        self.stop_timers()
        self._paused = True
        self._no_answer_sent = False
        self._reset_sentence_state()

    def resume(self) -> None:
        """由 start 事件调用：重置状态并重新启动计时器。"""
        self.stop_timers()
        self._paused = False
        self._no_answer_sent = False
        self._reset_sentence_state()
        # 记录打断禁区截止时刻
        self._interrupt_allow_after = time.monotonic() + self._interrupt_ignore_start_ms / 1000
        self.start_timers()

    def load_conf(self, call_conf: dict) -> None:
        """由 start 事件触发，从 Redis conf 覆盖计时参数；call_conf 为空则保持全局默认值。"""
        if not call_conf:
            logger.info("call_id=%s model_id=%s no redis conf, using global defaults", self.call_id, self.model_id)
            return

        interrupt_cfg = call_conf.get("interruptConfig") or {}
        intervene_cfg = call_conf.get("interveneConfig") or {}

        self._silence_max_ms = call_conf.get("maxPauseTime") or self._silence_max_ms
        no_response_time = call_conf.get("noResponseTime")
        if no_response_time is not None:
            try:
                parsed_no_response_time = float(no_response_time)
                if parsed_no_response_time >= 0:
                    self._no_answer_timeout_ms = int(parsed_no_response_time * 1000)
            except (TypeError, ValueError):
                logger.warning(
                    "call_id=%s model_id=%s invalid noResponseTime=%r, keep previous=%s",
                    self.call_id,
                    self.model_id,
                    no_response_time,
                    self._no_answer_timeout_ms,
                )
        self._match_timeout_ms = call_conf.get("matchTimeout") or self._match_timeout_ms
        interrupt_enabled = interrupt_cfg.get("enable")
        if interrupt_enabled is not None:
            self._interrupt_enabled = bool(interrupt_enabled)
            if self._interrupt_enabled:
                self._interrupt_threshold_ms = interrupt_cfg.get("interruptTime") or self._interrupt_threshold_ms
                self._interrupt_ignore_start_ms = (interrupt_cfg.get("startIgnoreSeconds") or 0) * 1000

        type_threshold = intervene_cfg.get("wordCount")

        normalized_threshold: int | None = None
        if type_threshold is not None:
            try:
                parsed = int(type_threshold)
                if parsed >= 1:
                    normalized_threshold = parsed
                else:
                    logger.warning(
                        "call_id=%s model_id=%s invalid wordCount=%r (<1), keep previous=%s",
                        self.call_id,
                        self.model_id,
                        type_threshold,
                        self._word_count,
                    )
            except (TypeError, ValueError):
                logger.warning(
                    "call_id=%s model_id=%s invalid wordCount=%r (non-int), keep previous=%s",
                    self.call_id,
                    self.model_id,
                    type_threshold,
                    self._word_count,
                )

        if "enable" in intervene_cfg:
            if intervene_cfg.get("enable"):
                if normalized_threshold is not None:
                    self._word_count = normalized_threshold
            else:
                self._word_count = None

        question_similarity = call_conf.get("questionSimilarity")
        if question_similarity is None:
            question_similarity = intervene_cfg.get("questionSimilarity")
        normalized_similarity: float | None = None
        if question_similarity is not None:
            try:
                parsed_similarity = float(question_similarity)
                if parsed_similarity < 0:
                    logger.warning(
                        "call_id=%s model_id=%s invalid question_similarity=%r (<0), ignore",
                        self.call_id,
                        self.model_id,
                        question_similarity,
                    )
                else:
                    normalized_similarity = parsed_similarity
            except (TypeError, ValueError):
                logger.warning(
                    "call_id=%s model_id=%s invalid question_similarity=%r (non-float), ignore",
                    self.call_id,
                    self.model_id,
                    question_similarity,
                )

        self._question_similarity = normalized_similarity

        noise_gate_vad_prob = call_conf.get("noiseGateRnnoiseVadProb")
        if noise_gate_vad_prob is not None:
            try:
                parsed_prob = float(noise_gate_vad_prob)
                if parsed_prob >= 0:
                    self._noise_gate_rnnoise_vad_prob = parsed_prob
            except (TypeError, ValueError):
                logger.warning(
                    "call_id=%s model_id=%s invalid noiseGateRnnoiseVadProb=%r, keep previous=%s",
                    self.call_id,
                    self.model_id,
                    noise_gate_vad_prob,
                    self._noise_gate_rnnoise_vad_prob,
                )

        noise_gate_min_frames = call_conf.get("noiseGateMinSpeechFrames")
        if noise_gate_min_frames is not None:
            try:
                parsed_frames = int(noise_gate_min_frames)
                if parsed_frames >= 1:
                    self._noise_gate_min_speech_frames = parsed_frames
            except (TypeError, ValueError):
                logger.warning(
                    "call_id=%s model_id=%s invalid noiseGateMinSpeechFrames=%r, keep previous=%s",
                    self.call_id,
                    self.model_id,
                    noise_gate_min_frames,
                    self._noise_gate_min_speech_frames,
                )

        noise_gate_lookback_ms = call_conf.get("noiseGateLookbackMs")
        if noise_gate_lookback_ms is not None:
            try:
                parsed_lookback_ms = int(noise_gate_lookback_ms)
                if parsed_lookback_ms >= 0:
                    self._noise_gate_lookback_ms = parsed_lookback_ms
            except (TypeError, ValueError):
                logger.warning(
                    "call_id=%s model_id=%s invalid noiseGateLookbackMs=%r, keep previous=%s",
                    self.call_id,
                    self.model_id,
                    noise_gate_lookback_ms,
                    self._noise_gate_lookback_ms,
                )

        noise_gate_chars = call_conf.get("noiseGateFilterChars")
        if isinstance(noise_gate_chars, list) and noise_gate_chars:
            self._noise_gate_filter_chars = set(noise_gate_chars)


        self._vad = VADDetector(threshold_ms=self._interrupt_threshold_ms)
        logger.info(
            "call_id=%s model_id=%s conf loaded: silence=%dms no_answer=%dms interrupt=%s/%dms ignore_start=%dms type_threshold=%s question_similarity=%s",
            self.call_id, self.model_id,
            self._silence_max_ms, self._no_answer_timeout_ms,
            self._interrupt_enabled, self._interrupt_threshold_ms,
            self._interrupt_ignore_start_ms, self._word_count,self._question_similarity
        )

    def _reset_sentence_state(self) -> None:
        """每句话处理完毕后调用，重置句子级状态，允许持续识别下一句。"""
        self._sentence_parts = []
        self._in_speech = False
        self._last_speech_end_ms = None
        self._first_text_received = False
        self._audio_lookback.clear()
        self._lookback_samples = 0
        self._noise_speech_streak = 0
        if self._match_timeout_task:
            self._match_timeout_task.cancel()
            self._match_timeout_task = None
        self._asr.reset()
        self._vad.reset()
        self._denoiser.reset()

    async def close(self) -> None:
        """连接断开时调用：刷新模型内部缓冲区，处理最后一帧剩余文本。"""
        self.stop_timers()
        if self._paused:
            return
        loop = asyncio.get_running_loop()

        text = await loop.run_in_executor(self._asr_executor, self._asr.transcribe, b"", True)
        if text:
            self._sentence_parts.append(text)

        if self._sentence_parts:
            final_sentence = "".join(self._sentence_parts)
            self._sentence_parts = []
            logger.info("call_id=%s final flush: %s", self.call_id, final_sentence)
            await self._call_intent(final_sentence)

    async def handle_audio(self, audio_bytes: bytes) -> None:
        if self._paused:
            logger.debug("call_id=%s handle_audio skipped: paused", self.call_id)
            return

        loop = asyncio.get_running_loop()
        _t0 = time.monotonic()

        # 0a. 8kHz→16kHz 上采样（如需要，放入 executor 避免阻塞事件循环）
        if self._resampler is not None:
            raw_audio = await loop.run_in_executor(self._vad_executor, self._resampler.process, audio_bytes)
        else:
            raw_audio = audio_bytes

        # 0b. RNNoise 降噪 + VAD 语音判定（并行执行）
        denoise_future = loop.run_in_executor(self._vad_executor, self._denoiser.process, raw_audio)
        vad_future = loop.run_in_executor(self._vad_executor, self._vad.is_speech, raw_audio)
        (audio_bytes, rnnoise_vad_prob), is_speech = await asyncio.gather(denoise_future, vad_future)
        _t1 = time.monotonic()
        if not audio_bytes:
            return

        bytes_per_sample = 2  # 16-bit
        num_samples = len(audio_bytes) // bytes_per_sample
        self._total_samples += num_samples

        end_ms = int(self._total_samples / 16000 * 1000)
        chunk_ms = end_ms - self._elapsed_ms
        logger.debug(
            "call_id=%s handle_audio start: bytes=%d chunk_ms=%d elapsed_ms=%d end_ms=%d",
            self.call_id,
            len(audio_bytes),
            chunk_ms,
            self._elapsed_ms,
            end_ms,
        )

        _t2 = _t1
        logger.debug("call_id=%s step1 is_speech=%s rnnoise_vad_prob=%.3f", self.call_id, is_speech, rnnoise_vad_prob)

        min_noise_vad_prob = self._noise_gate_rnnoise_vad_prob
        min_noise_speech_frames = self._noise_gate_min_speech_frames
        noise_gate_chars = self._noise_gate_filter_chars
        looks_like_noise = bool(is_speech and rnnoise_vad_prob < min_noise_vad_prob)
        self._noise_speech_streak = self._noise_speech_streak + 1 if looks_like_noise else 0

        # 1.5 打断检测（基于 step1 的 is_speech，不重复跑 VAD）
        if self._interrupt_enabled:
            in_ignore_window = time.monotonic() < self._interrupt_allow_after
            if in_ignore_window:
                logger.debug("call_id=%s step1.5 interrupt check skipped: in ignore window", self.call_id)
                # 仍需驱动 process_speech 以维护 VAD 内部状态，但不触发打断
                await loop.run_in_executor(self._vad_executor, self._vad.process_speech, is_speech)
            else:
                logger.debug("call_id=%s step1.5 interrupt check start", self.call_id)
                should_interrupt = await loop.run_in_executor(self._vad_executor, self._vad.process_speech, is_speech)
                logger.debug("call_id=%s step1.5 interrupt check result: should_interrupt=%s", self.call_id,
                             should_interrupt)
                if should_interrupt:
                    logger.info("call_id=%s step1.5 interrupt triggered", self.call_id)
                    self._create_safe_task(self._send_interrupt())
        else:
            logger.debug("call_id=%s step1.5 interrupt check skipped: interrupt_enabled=false", self.call_id)
        _t3 = time.monotonic()

        # 2. 流式 ASR 转写：音频帧必须完整送入以维持模型内部状态连续性
        text = ""
        if is_speech:
            if not self._in_speech and self._audio_lookback:
                lookback_audio = b"".join(self._audio_lookback)
                await loop.run_in_executor(self._asr_executor, self._asr.transcribe, lookback_audio)
                self._audio_lookback.clear()
                self._lookback_samples = 0
                logger.debug("call_id=%s step2 lookback frames fed to ASR", self.call_id)
            logger.debug("call_id=%s step2 asr transcribe start (is_speech=True)", self.call_id)
            text = await loop.run_in_executor(self._asr_executor, self._asr.transcribe, audio_bytes)
            if text and self._noise_speech_streak < min_noise_speech_frames and set(text) <= noise_gate_chars:
                logger.info(
                    "call_id=%s step2 noise-gate drop text=%r streak=%d rnnoise_vad_prob=%.3f",
                    self.call_id,
                    text,
                    self._noise_speech_streak,
                    rnnoise_vad_prob,
                )
                text = ""
        else:
            self._audio_lookback.append(audio_bytes)
            self._lookback_samples += len(audio_bytes) // 2
            max_lookback_ms = self._noise_gate_lookback_ms
            max_lookback_samples = max_lookback_ms * 16
            while self._lookback_samples > max_lookback_samples and self._audio_lookback:
                removed = self._audio_lookback.popleft()
                self._lookback_samples -= len(removed) // 2
            logger.debug("call_id=%s step2 asr transcribe skipped (noise dropped)", self.call_id)
        _t4 = time.monotonic()

        if text:
            logger.info(
                "call_id=%s handle_audio timing: denoise+vad=%dms interrupt=%dms asr=%dms total=%dms",
                self.call_id,
                int((_t1 - _t0) * 1000),
                int((_t3 - _t2) * 1000),
                int((_t4 - _t3) * 1000),
                int((_t4 - _t0) * 1000),
            )
            self._reset_no_answer_timer()
            logger.info("call_id=%s step2 no_answer timer reset", self.call_id)
            if not self._first_text_received:
                self._first_text_received = True
                self._match_timeout_task = asyncio.create_task(self._match_timer())
                logger.info("call_id=%s step2 first text received, match timer started", self.call_id)

            self._sentence_parts.append(text)
            self._all_segments.append({
                "text": text,
                "start_ms": self._elapsed_ms,
                "end_ms": end_ms,
            })
            text_preview = text if len(text) <= 120 else f"{text[:120]}..."
            logger.info("call_id=%s step2 asr text=%r", self.call_id, text_preview)

        else:
            logger.debug("call_id=%s step2 asr empty text", self.call_id)

        # 3. 停顿检测 → 句子结束判定
        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                logger.debug("call_id=%s step3 speech segment started", self.call_id)
            self._last_speech_end_ms = end_ms
            logger.debug("call_id=%s step3 update last_speech_end_ms=%d", self.call_id, end_ms)
        else:
            if self._in_speech and self._last_speech_end_ms is not None:
                silence_ms = end_ms - self._last_speech_end_ms
                logger.debug(
                    "call_id=%s step3 silence check: silence_ms=%d threshold_ms=%d has_sentence_parts=%s",
                    self.call_id,
                    silence_ms,
                    self._silence_max_ms,
                    bool(self._sentence_parts),
                )
                if silence_ms >= self._silence_max_ms and self._sentence_parts:
                    sentence = "".join(self._sentence_parts)

                    self._sentence_parts = []
                    self._sentence_start_ms = end_ms
                    self._in_speech = False
                    self._last_speech_end_ms = None
                    self._sentence_epoch += 1
                    sentence_epoch = self._sentence_epoch
                    self._reset_sentence_state()

                    logger.info("call_id=%s step3 sentence done (silence %dms  max_silence %dms): %s",
                                self.call_id, silence_ms, self._silence_max_ms, sentence)
                    logger.info("call_id=%s step3 intent task scheduled (epoch=%d)", self.call_id, sentence_epoch)
                    self._create_safe_task(self._send_transcript(sentence))
                    self._create_safe_task(self._call_intent(sentence, sentence_epoch))

        self._elapsed_ms = end_ms
        logger.debug("call_id=%s handle_audio end: elapsed_ms=%d", self.call_id, self._elapsed_ms)

    # ------------------------------------------------------------------ #
    #  Intent                                                              #
    # ------------------------------------------------------------------ #

    def _create_safe_task(self, coro: Awaitable[Any]) -> None:
        async def _runner() -> None:
            try:
                await coro
            except Exception as e:
                logger.error("call_id=%s background task failed: %s", self.call_id, e)

        asyncio.create_task(_runner())

    async def _call_intent(self, sentence: str, sentence_epoch: int | None = None) -> None:
        url = f"{cfg.get('intent_service_url')}/api/v1/recognize"
        payload: dict = {
            "text": sentence,
            "call_id": str(self.call_id),
            "model_id": self.model_id,
        }
        if self._word_count is not None:
            payload["word_count"] = self._word_count
        if self._question_similarity is not None:
            payload["question_similarity"] = self._question_similarity
        logger.debug("call_id=%s step3 intent payload=%s epoch=%s", self.call_id, payload, sentence_epoch)
        try:
            resp = await self._http_client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("call_id=%s intent call failed: %s", self.call_id, e)
            return

        if cfg.get("intent_epoch_guard_enabled",
                   True) and sentence_epoch is not None and sentence_epoch != self._sentence_epoch:
            logger.info(
                "call_id=%s intent result ignored: stale epoch result=%s current=%s",
                self.call_id,
                sentence_epoch,
                self._sentence_epoch,
            )
            return

        intent_id = data.get("intent_id", "intent_unknown")
        logger.info("call_id=%s intent_id=%s text=%r epoch=%s", self.call_id, intent_id, sentence, sentence_epoch)

        if intent_id != "intent_unknown":
            await self._send_callback(intent_id, sentence)

    # ------------------------------------------------------------------ #
    #  对外推送                                                             #
    # ------------------------------------------------------------------ #

    async def _send_callback(self, intent_id: str, text: str) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "INTENT",
            "intent_id": intent_id,
            "uuid": self.uuid,
            "text": text,
        }
        await self._post(cfg.get("business_callback_url"), payload, "intent callback")

    async def _send_transcript(self, text: str) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "transcript",
            "uuid": self.uuid,
            "text": text,
        }
        await self._post(cfg.get("transcript_url"), payload, "transcript", timeout=3.0)
        logger.info(f"call_id={self.call_id} transcript={text} 发送文本识别消息={cfg.get("transcript_url")}")

    async def _send_interrupt(self) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "INTERRUPT",
            "uuid": self.uuid,
        }
        await self._post(cfg.get("business_callback_url"), payload, "interrupt", timeout=3.0)

    async def _send_no_answer(self) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "NOANSWER",
            "uuid": self.uuid,
        }
        await self._post(cfg.get("business_callback_url"), payload, "no_answer", timeout=3.0)

    async def _send_fallback(self) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "TIMEOUT",
            "uuid": self.uuid,
        }
        await self._post(cfg.get("business_callback_url"), payload, "match_timeout fallback", timeout=3.0)

    async def _post(self, url: str, payload: dict, label: str, timeout: float = 5.0) -> None:
        try:
            resp = await self._http_client.post(
                url, json=payload, timeout=httpx.Timeout(timeout)
            )
            resp.raise_for_status()
            logger.info("call_id=%s %s sent", self.call_id, label)
        except Exception as e:
            logger.warning("call_id=%s %s failed: %s", self.call_id, label, e)

    # ------------------------------------------------------------------ #
    #  计时器                                                               #
    # ------------------------------------------------------------------ #

    async def _no_answer_timer(self) -> None:
        start_delay_s = self._timer_start_delay_s()
        if start_delay_s > 0:
            await asyncio.sleep(start_delay_s)

        timeout_s = self._no_answer_timeout_ms / 1000
        await asyncio.sleep(timeout_s)
        if not self._paused and not self._no_answer_sent:
            logger.info(
                "call_id=%s no_answer timeout (delay=%.1fs timeout=%.1fs)",
                self.call_id,
                start_delay_s,
                timeout_s,
            )
            await self._send_no_answer()
            self._no_answer_sent = True
            self._reset_sentence_state()

    async def _match_timer(self) -> None:
        start_delay_s = self._timer_start_delay_s()
        if start_delay_s > 0:
            await asyncio.sleep(start_delay_s)

        timeout_s = self._match_timeout_ms / 1000
        await asyncio.sleep(timeout_s)
        if not self._paused:
            logger.info(
                "call_id=%s match timeout (delay=%.1fs timeout=%.1fs)",
                self.call_id,
                start_delay_s,
                timeout_s,
            )
            await self._send_fallback()
            self._reset_sentence_state()

    def _reset_no_answer_timer(self) -> None:
        if self._no_answer_task:
            self._no_answer_task.cancel()
        self._no_answer_task = asyncio.create_task(self._no_answer_timer())

    # ------------------------------------------------------------------ #
    #  工具                                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_duration_ms(audio_bytes: bytes, sample_rate: int = 16000, bit_depth: int = 16) -> int:
        bytes_per_sample = bit_depth // 8
        num_samples = len(audio_bytes) // bytes_per_sample
        return int(num_samples / sample_rate * 1000)
