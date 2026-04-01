import asyncio
import logging
import time
from concurrent.futures import Executor
from typing import Any

import httpx

from asr.denoiser import RNNoiseFilter
from asr.engine import ASREngine
from asr.offline_engine import OfflineASREngine  # [新增] 导入离线引擎
from asr.vad import VADDetector
from config import nacos_config as cfg

logger = logging.getLogger(__name__)


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
    - executor: 外部注入的 ThreadPoolExecutor，将 CPU 密集推理移出事件循环
    """

    def __init__(self, call_id: int, uuid: int, model_id: int, http_client: httpx.AsyncClient, executor: Executor):
        self.call_id = call_id
        self.uuid = uuid
        self.model_id = model_id
        self._http_client = http_client
        self._executor = executor

        # 初始化时先用全局配置，start 事件后由 load_conf() 覆盖
        self._silence_max_ms: int = cfg.get("silence_max_ms", 800)
        self._no_answer_timeout_ms: int = cfg.get("no_answer_timeout_ms", 10000)
        self._match_timeout_ms: int = 10000
        self._interrupt_enabled: bool = cfg.get("interrupt_enabled", True)
        self._interrupt_threshold_ms: int = cfg.get("vad_interrupt_threshold_ms", 2000)
        self._interrupt_ignore_start_ms: int = 0   # 开启后前 N ms 禁止打断
        self._word_count: int = 2
        self._question_similarity: float | None = None

        self._vad = VADDetector(threshold_ms=self._interrupt_threshold_ms)
        self._asr = ASREngine()
        self._offline_asr = OfflineASREngine()  # [新增] 实例化离线引擎
        self._denoiser = RNNoiseFilter()  # 每路连接独立实例，线程安全
        self._paused = True

        # 当前句子文本缓冲与音频缓冲
        self._sentence_parts: list[str] = []
        self._sentence_audio_buffer = bytearray()  # [新增] 缓冲当前整句的音频
        self._sentence_start_ms: int = 0
        self._sentence_epoch: int = 0

        # 全量转写片段
        self._all_segments: list[dict[str, Any]] = []
        self._elapsed_ms: int = 0
        self._total_samples: int = 0

        # 停顿检测状态
        self._in_speech: bool = False
        self._last_speech_end_ms: int | None = None

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
        self._no_answer_timeout_ms = call_conf.get("noResponseTime") or self._no_answer_timeout_ms
        if "enable" in interrupt_cfg:
            self._interrupt_enabled = interrupt_cfg["enable"]
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
        self._sentence_audio_buffer.clear()  # [新增] 清空音频缓冲
        self._in_speech = False
        self._last_speech_end_ms = None
        self._first_text_received = False
        if self._match_timeout_task:
            self._match_timeout_task.cancel()
            self._match_timeout_task = None
        self._asr.reset()
        self._vad.reset()
        self._denoiser.reset()  # 清空余量，防止句间串扰

    async def close(self) -> None:
        """连接断开时调用：刷新模型内部缓冲区，处理最后一帧剩余文本。"""
        self.stop_timers()
        if self._paused:
            return
        loop = asyncio.get_running_loop()

        # [修改] 获取流式模型最后的 flush
        text = await loop.run_in_executor(self._executor, self._asr.transcribe, b"", True)
        if text:
            self._sentence_parts.append(text)

        final_sentence = ""

        # [新增] 如果缓冲区有残留音频，尝试用离线模型做最后一次高精度识别
        if self._sentence_audio_buffer:
            final_audio = bytes(self._sentence_audio_buffer)
            offline_text = await loop.run_in_executor(self._executor, self._offline_asr.transcribe, final_audio)
            if offline_text:
                final_sentence = offline_text

        # 降级：如果离线没结果，用在线模型拼凑的文本
        if not final_sentence and self._sentence_parts:
            final_sentence = "".join(self._sentence_parts)

        if final_sentence:
            self._sentence_parts = []
            logger.info("call_id=%s final flush: %s", self.call_id, final_sentence)
            await self._call_intent(final_sentence)

    async def handle_audio(self, audio_bytes: bytes) -> None:
        if self._paused:
            logger.debug("call_id=%s handle_audio skipped: paused", self.call_id)
            return

        loop = asyncio.get_running_loop()
        _t0 = time.monotonic()

        # 0. RNNoise 降噪预处理（executor 中执行，不阻塞事件循环）
        audio_bytes, _vad_prob = await loop.run_in_executor(
            self._executor, self._denoiser.process, audio_bytes
        )
        _t1 = time.monotonic()
        if not audio_bytes:
            # 余量未凑够一帧，等下一帧补齐，行为等价于静音帧
            return

        bytes_per_sample = 2  # 16-bit
        num_samples = len(audio_bytes) // bytes_per_sample
        self._total_samples += num_samples

        # chunk_ms = self._estimate_duration_ms(audio_bytes)
        # end_ms = self._elapsed_ms + chunk_ms

        end_ms = int(self._total_samples / 16000 * 1000)

        # 如果有些地方依然需要 chunk_ms
        chunk_ms = end_ms - self._elapsed_ms
        logger.debug(
            "call_id=%s handle_audio start: bytes=%d chunk_ms=%d elapsed_ms=%d end_ms=%d",
            self.call_id,
            len(audio_bytes),
            chunk_ms,
            self._elapsed_ms,
            end_ms,
        )

        # 1. 单次 VAD 语音判定（后续打断与停顿逻辑复用）
        is_speech = await loop.run_in_executor(self._executor, self._vad.is_speech, audio_bytes)
        _t2 = time.monotonic()
        logger.debug("call_id=%s step1 is_speech=%s", self.call_id, is_speech)

        if is_speech or self._in_speech:
            self._sentence_audio_buffer.extend(audio_bytes)

        # 1.5 打断检测（基于 step1 的 is_speech，不重复跑 VAD）
        if self._interrupt_enabled:
            in_ignore_window = time.monotonic() < self._interrupt_allow_after
            if in_ignore_window:
                logger.debug("call_id=%s step1.5 interrupt check skipped: in ignore window", self.call_id)
                # 仍需驱动 process_speech 以维护 VAD 内部状态，但不触发打断
                await loop.run_in_executor(self._executor, self._vad.process_speech, is_speech)
            else:
                logger.debug("call_id=%s step1.5 interrupt check start", self.call_id)
                should_interrupt = await loop.run_in_executor(self._executor, self._vad.process_speech, is_speech)
                logger.debug("call_id=%s step1.5 interrupt check result: should_interrupt=%s", self.call_id,
                             should_interrupt)
                if should_interrupt:
                    logger.info("call_id=%s step1.5 interrupt triggered", self.call_id)
                    asyncio.create_task(self._send_interrupt())
        else:
            logger.debug("call_id=%s step1.5 interrupt check skipped: interrupt_enabled=false", self.call_id)
        _t3 = time.monotonic()

        # 2. 流式 ASR 转写：音频帧必须完整送入以维持模型内部状态连续性
        text = ""
        if is_speech:
            # 只有 VAD 判定这帧里有讲话声，才喂给在线 Paraformer。
            logger.debug("call_id=%s step2 asr transcribe start (is_speech=True)", self.call_id)
            text = await loop.run_in_executor(self._executor, self._asr.transcribe, audio_bytes)
        else:
            # 纯噪音直接 Drop 掉，连推理都不做。
            # 这能彻底杜绝 ASR 强行在底噪里找发音造成的"蹦出毫无关联的杂字"（幻觉）现象。
            logger.debug("call_id=%s step2 asr transcribe skipped (noise dropped)", self.call_id)
        _t4 = time.monotonic()

        logger.debug(
            "call_id=%s handle_audio timing: denoise=%dms vad=%dms interrupt=%dms asr=%dms total=%dms",
            self.call_id,
            int((_t1 - _t0) * 1000),
            int((_t2 - _t1) * 1000),
            int((_t3 - _t2) * 1000),
            int((_t4 - _t3) * 1000),
            int((_t4 - _t0) * 1000),
        )


        if text:
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
                    # [修改] 提取原始流式识别的句子，并同时提取音频送入离线模型纠错
                    online_sentence = "".join(self._sentence_parts)
                    sentence_audio = bytes(self._sentence_audio_buffer)

                    _offline_start = time.monotonic()
                    logger.info("call_id=%s step3 offline ASR start: audio_bytes=%d", self.call_id, len(sentence_audio))
                    offline_sentence = await loop.run_in_executor(self._executor, self._offline_asr.transcribe,
                                                                  sentence_audio)
                    _offline_ms = int((time.monotonic() - _offline_start) * 1000)
                    logger.info("call_id=%s step3 offline ASR done: cost=%dms result=%r", self.call_id, _offline_ms, offline_sentence)

                    # 如果离线模型吐出了有效文本，则覆盖；否则降级保留原本的在线结果
                    sentence = offline_sentence if offline_sentence else online_sentence

                    self._sentence_parts = []
                    self._sentence_start_ms = end_ms
                    self._in_speech = False
                    self._last_speech_end_ms = None
                    self._sentence_epoch += 1
                    sentence_epoch = self._sentence_epoch
                    self._reset_sentence_state()

                    # [修改] 在原本的日志中额外增加 Online 和 Offline 的对比，方便 Debug 精度提升效果
                    logger.info("call_id=%s step3 sentence done (silence %dms): %s (Online raw: %s)",
                                self.call_id, silence_ms, sentence, online_sentence)
                    logger.info("call_id=%s step3 intent task scheduled (epoch=%d)", self.call_id, sentence_epoch)
                    asyncio.create_task(self._send_transcript(sentence))
                    asyncio.create_task(self._call_intent(sentence, sentence_epoch))

        self._elapsed_ms = end_ms
        logger.debug("call_id=%s handle_audio end: elapsed_ms=%d", self.call_id, self._elapsed_ms)

    # ------------------------------------------------------------------ #
    #  Intent                                                              #
    # ------------------------------------------------------------------ #

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
            "event": "intent",
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
            "event": "interrupt",
            "uuid": self.uuid,
        }
        await self._post(cfg.get("business_callback_url"), payload, "interrupt", timeout=3.0)

    async def _send_no_answer(self) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "no_answer",
            "uuid": self.uuid,
        }
        await self._post(cfg.get("business_callback_url"), payload, "no_answer", timeout=3.0)

    async def _send_fallback(self) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "match_timeout",
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
        timeout_s = self._no_answer_timeout_ms / 1000
        await asyncio.sleep(timeout_s)
        if not self._paused and not self._no_answer_sent:
            logger.info("call_id=%s no_answer timeout (%.1fs)", self.call_id, timeout_s)
            await self._send_no_answer()
            self._no_answer_sent = True
            self._reset_sentence_state()

    async def _match_timer(self) -> None:
        timeout_s = self._match_timeout_ms / 1000
        await asyncio.sleep(timeout_s)
        if not self._paused:
            logger.info("call_id=%s match timeout (%.1fs)", self.call_id, timeout_s)
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