import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import NamedTuple

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config import nacos_config as cfg
from log_config import setup_logging
from asr.engine import load_model
from asr.vad import load_vad_model
from asr.offline_engine import load_offline_model
from asr.stream_handler import StreamHandler

logger = logging.getLogger(__name__)

_AUDIO_QUEUE_SENTINEL = object()

class _Executors(NamedTuple):
    vad: ThreadPoolExecutor
    asr: ThreadPoolExecutor
    offline_asr: ThreadPoolExecutor


def _build_executors() -> _Executors:
    vad_workers = int(os.environ.get("VAD_WORKERS", str(cfg.get("vad_workers", 4))))
    asr_workers = int(os.environ.get("ASR_WORKERS", str(cfg.get("online_asr_workers", cfg.get("asr_workers", 8)))))
    offline_asr_workers = int(os.environ.get("OFFLINE_ASR_WORKERS", str(cfg.get("offline_asr_workers", cfg.get("asr_workers", 8)))))
    return _Executors(
        vad=ThreadPoolExecutor(max_workers=vad_workers, thread_name_prefix="vad"),
        asr=ThreadPoolExecutor(max_workers=asr_workers, thread_name_prefix="asr"),
        offline_asr=ThreadPoolExecutor(max_workers=offline_asr_workers, thread_name_prefix="offline-asr"),
    )


class IntentTestRequest(BaseModel):
    text: str
    call_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await cfg.init_config(
        nacos_server=os.environ.get("NACOS_SERVER", "nacos.register.service.com:8848"),
        nacos_namespace=os.environ.get("NACOS_NAMESPACE", "asr_test"),
        nacos_data_id=os.environ.get("NACOS_DATA_ID", "asr-server.yaml"),
        nacos_group=os.environ.get("NACOS_GROUP", "dolphin"),
    )
    load_model()
    load_vad_model()

    executors = _build_executors()

    # P0-fix-3: 全局 httpx 连接池，所有 StreamHandler 共用
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
        timeout=httpx.Timeout(5.0),
    )
    app.state.executor = executors

    # Redis 客户端（per-call 配置从 Redis 拉取）
    app.state.redis = aioredis.from_url(
        f"redis://{os.environ.get('REDIS_HOST', 'dev.redis.service.com')}:{os.environ.get('REDIS_PORT', '6379')}",
        password=os.environ.get("REDIS_PASSWORD") or None,
        db=int(os.environ.get("REDIS_DB", "9")),
        decode_responses=True,
    )

    logger.info("dolphin-asr service started.")
    yield
    logger.info("dolphin-asr service stopped.")


app = FastAPI(title="dolphin-asr", lifespan=lifespan)


async def _load_model_conf(redis_client: aioredis.Redis, model_id: int) -> dict:
    """从 Redis 拉取 ai_model:{model_id}:conf；key 不存在或异常时返回空 dict。"""
    key = f"ai_model:{model_id}:conf"
    try:
        raw = await redis_client.get(key)
        if raw:
            return json.loads(raw)
        logger.info("model_id=%s redis key not found: %s, using global defaults", model_id, key)
    except Exception as e:
        logger.warning("model_id=%s redis get conf failed: %s, using global defaults", model_id, e)
    return {}


async def _handle_ws_event(websocket: WebSocket, handler: StreamHandler, raw: str, call_id: str) -> bool:
    """处理文本事件帧，返回是否需要结束 ws 循环。"""
    preview = raw if len(raw) <= 200 else f"{raw[:200]}..."
    logger.info(
        "call_id=%s ws text frame received: raw_len=%d preview=%r",
        call_id,
        len(raw),
        preview,
    )
    try:
        data = json.loads(raw)
    except ValueError as e:
        logger.warning(
            "call_id=%s invalid event frame: raw_len=%d preview=%r err=%s",
            call_id,
            len(raw),
            preview,
            e,
        )
        return False

    event = data.get("control")
    if event == "start":
        logger.info("call_id=%s event=start: ASR resumed", call_id)
        call_conf = await _load_model_conf(websocket.app.state.redis, handler.model_id)
        handler.load_conf(call_conf)
        handler.resume()
        return False

    if event == "resume":
        logger.info("call_id=%s event=resume: ASR resumed, no_answer timer reset", call_id)
        handler.resume()
        return False

    if event == "pause":
        logger.info("call_id=%s event=pause: ASR paused", call_id)
        handler.pause()
        return False

    if event == "stop":
        logger.info("call_id=%s event=stop: closing websocket", call_id)
        handler.pause()
        await websocket.close()
        return True

    logger.warning("call_id=%s unknown event=%r", call_id, event)
    return False


async def _drain_audio_queue(audio_queue: asyncio.Queue[bytes | object]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
        except asyncio.QueueEmpty:
            return


async def _audio_consumer_loop(handler: StreamHandler, audio_queue: asyncio.Queue[bytes | object]) -> None:
    while True:
        item = await audio_queue.get()
        try:
            if item is _AUDIO_QUEUE_SENTINEL:
                return
            try:
                await handler.handle_audio(item)
            except Exception as e:
                logger.error("Audio consumer frame handling failed: %s", e, exc_info=True)
        finally:
            audio_queue.task_done()


def _enqueue_sentinel(audio_queue: asyncio.Queue[bytes | object]) -> None:
    try:
        audio_queue.put_nowait(_AUDIO_QUEUE_SENTINEL)
        return
    except asyncio.QueueFull:
        pass

    try:
        audio_queue.get_nowait()
        audio_queue.task_done()
    except asyncio.QueueEmpty:
        return

    try:
        audio_queue.put_nowait(_AUDIO_QUEUE_SENTINEL)
    except asyncio.QueueFull:
        logger.warning("Failed to enqueue audio queue sentinel: queue still full")


@app.websocket("/ws/asr")
async def ws_asr(websocket: WebSocket, call_id: str, uuid: str, model_id: int = 0):
    await websocket.accept()
    handler = StreamHandler(
        call_id=call_id,
        uuid=uuid,
        model_id=model_id,
        http_client=websocket.app.state.http_client,
        vad_executor=websocket.app.state.executor.vad,
        asr_executor=websocket.app.state.executor.asr,
    )
    queue_maxsize = max(1, int(cfg.get("audio_queue_maxsize", 64)))
    audio_queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=queue_maxsize)
    consumer_task = asyncio.create_task(_audio_consumer_loop(handler, audio_queue))
    stream_active = False

    logger.debug("WebSocket connected: call_id=%s", call_id)
    handler.emit_monitor_event("ws_open", text="websocket accepted", model_id=model_id)
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            text = message.get("text")
            audio_bytes = message.get("bytes")

            if msg_type == "websocket.disconnect":
                logger.debug("WebSocket message indicates disconnect: call_id=%s", call_id)
                break

            if text is not None:
                try:
                    event = json.loads(text).get("control")
                except ValueError:
                    event = None

                should_close = await _handle_ws_event(websocket, handler, text, call_id)

                if event in {"start", "resume"}:
                    stream_active = True
                    await _drain_audio_queue(audio_queue)
                elif event in {"pause", "stop"}:
                    stream_active = False
                    await _drain_audio_queue(audio_queue)

                if event in {"start", "resume", "pause", "stop"}:
                    handler.emit_monitor_event(f"ws_{event}", text=f"control={event}")

                if should_close:
                    break
                continue

            if audio_bytes is not None:
                input_sample_rate = int(getattr(handler, "_audio_input_sample_rate", cfg.get("audio_input_sample_rate", 16000)))
                input_codec = "pcm16"
                bytes_per_sample = 2
                num_samples = len(audio_bytes) // bytes_per_sample
                chunk_ms = int(num_samples / input_sample_rate * 1000) if input_sample_rate > 0 else 0
                logger.debug(
                    "call_id=%s ws audio frame received: bytes=%d samples=%d sample_rate=%d codec=%s chunk_ms=%d",
                    call_id,
                    len(audio_bytes),
                    num_samples,
                    input_sample_rate,
                    input_codec,
                    chunk_ms,
                )
                if not stream_active:
                    continue
                try:
                    audio_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    try:
                        dropped = audio_queue.get_nowait()
                        audio_queue.task_done()
                        if dropped is _AUDIO_QUEUE_SENTINEL:
                            await audio_queue.put(_AUDIO_QUEUE_SENTINEL)
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        audio_queue.put_nowait(audio_bytes)
                    except asyncio.QueueFull:
                        logger.debug("call_id=%s audio frame dropped due to full queue", call_id)
                continue

            logger.warning("call_id=%s ws message ignored: neither text nor bytes present", call_id)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: call_id=%s", call_id)
    except Exception as e:
        logger.error("WebSocket error call_id=%s: %s", call_id, e)
    finally:
        handler.emit_monitor_event("ws_close", text="websocket closed")
        consumer_task.cancel()
        with suppress(asyncio.CancelledError):
            await consumer_task
        return
