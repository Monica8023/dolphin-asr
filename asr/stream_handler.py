import asyncio
import logging
import time
from typing import Any

import httpx

from asr.engine import transcribe
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
    """

    def __init__(self, call_id: int, uuid: int):
        self.call_id = call_id
        self.uuid = uuid
        self._vad = VADDetector(threshold_ms=cfg.get("vad_interrupt_threshold_ms", 2000))
        self._paused = False

        # 当前句子缓冲
        self._sentence_parts: list[str] = []
        self._sentence_start_ms: int = 0

        # 全量转写片段
        self._all_segments: list[dict[str, Any]] = []
        self._elapsed_ms: int = 0

        # 停顿检测状态
        self._in_speech: bool = False
        self._last_speech_end_ms: int | None = None

        # 计时器
        self._no_answer_task: asyncio.Task | None = None
        self._match_timeout_task: asyncio.Task | None = None
        self._first_text_received: bool = False

    def start_timers(self) -> None:
        """连接建立后调用，启动无应答计时器。"""
        self._no_answer_task = asyncio.create_task(self._no_answer_timer())

    def stop_timers(self) -> None:
        """连接断开或流程结束时调用。"""
        if self._no_answer_task:
            self._no_answer_task.cancel()
        if self._match_timeout_task:
            self._match_timeout_task.cancel()

    async def handle_audio(self, audio_bytes: bytes) -> None:
        if self._paused:
            return

        chunk_ms = self._estimate_duration_ms(audio_bytes)
        end_ms = self._elapsed_ms + chunk_ms
        is_speech = self._vad.is_speech(audio_bytes)

        # 1. 打断检测
        if cfg.get("interrupt_enabled", True):
            if self._vad.process(audio_bytes):
                asyncio.create_task(self._send_interrupt())

        # 2. ASR 转写
        text = transcribe(audio_bytes)
        if text:
            self._reset_no_answer_timer()
            if not self._first_text_received:
                self._first_text_received = True
                self._match_timeout_task = asyncio.create_task(self._match_timer())

            self._sentence_parts.append(text)
            self._all_segments.append({
                "text": text,
                "start_ms": self._elapsed_ms,
                "end_ms": end_ms,
            })
            logger.debug("call_id=%s asr: %s", self.call_id, text)

        # 3. 停顿检测 → 句子结束判定
        if is_speech:
            if not self._in_speech:
                self._in_speech = True
            self._last_speech_end_ms = end_ms
        else:
            if self._in_speech and self._last_speech_end_ms is not None:
                silence_ms = end_ms - self._last_speech_end_ms
                if silence_ms >= cfg.get("silence_max_ms", 800) and self._sentence_parts:
                    sentence = "".join(self._sentence_parts)
                    seg_start = self._sentence_start_ms
                    self._sentence_parts = []
                    self._sentence_start_ms = end_ms
                    self._in_speech = False
                    self._last_speech_end_ms = None
                    logger.info("call_id=%s sentence done (silence %dms): %s", self.call_id, silence_ms, sentence)
                    asyncio.create_task(self._call_intent(sentence, seg_start, end_ms))

        self._elapsed_ms = end_ms

    # ------------------------------------------------------------------ #
    #  Intent                                                              #
    # ------------------------------------------------------------------ #

    async def _call_intent(self, sentence: str, start_ms: int, end_ms: int) -> None:
        url = f"{cfg.get('intent_service_url')}/api/v1/recognize"
        payload = {"text": sentence, "callId": str(self.call_id)}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("call_id=%s intent call failed: %s", self.call_id, e)
            return

        intent_id = data.get("intent_id", "intent_unknown")
        logger.info("call_id=%s intent_id=%s text=%r", self.call_id, intent_id, sentence)

        if intent_id != "intent_unknown":
            self._paused = True
            self.stop_timers()
            await self._send_callback(intent_id)

    # ------------------------------------------------------------------ #
    #  对外推送                                                             #
    # ------------------------------------------------------------------ #

    async def _send_callback(self, intent_id: str) -> None:
        payload = {
            "call_id": self.call_id,
            "intent_id": intent_id,
            "uuid": self.uuid,
        }
        await self._post(cfg.get("business_callback_url"), payload, "intent callback")

    async def _send_interrupt(self) -> None:
        await self._post(cfg.get("interrupt_url"), {"call_id": self.call_id}, "interrupt", timeout=3.0)

    async def _send_no_answer(self) -> None:
        payload = {"call_id": self.call_id, "event": "no_answer", "timestamp": int(time.time() * 1000)}
        await self._post(cfg.get("business_callback_url"), payload, "no_answer", timeout=3.0)

    async def _send_fallback(self) -> None:
        payload = {
            "call_id": self.call_id,
            "event": "match_timeout",
            "transcript": "".join(s["text"] for s in self._all_segments),
            "timestamp": int(time.time() * 1000),
        }
        await self._post(cfg.get("business_callback_url"), payload, "match_timeout fallback", timeout=3.0)

    async def _post(self, url: str, payload: dict, label: str, timeout: float = 5.0) -> None:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            logger.info("call_id=%s %s sent", self.call_id, label)
        except Exception as e:
            logger.warning("call_id=%s %s failed: %s", self.call_id, label, e)

    # ------------------------------------------------------------------ #
    #  计时器                                                               #
    # ------------------------------------------------------------------ #

    async def _no_answer_timer(self) -> None:
        timeout_s = cfg.get("no_answer_timeout_ms", 10000) / 1000
        await asyncio.sleep(timeout_s)
        if not self._paused:
            logger.info("call_id=%s no_answer timeout (%.1fs)", self.call_id, timeout_s)
            self._paused = True
            await self._send_no_answer()

    async def _match_timer(self) -> None:
        timeout_s = cfg.get("match_timeout_ms", 15000) / 1000
        await asyncio.sleep(timeout_s)
        if not self._paused:
            logger.info("call_id=%s match timeout (%.1fs)", self.call_id, timeout_s)
            self._paused = True
            await self._send_fallback()

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
