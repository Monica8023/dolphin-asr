import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

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
from asr.enhancer import load_enhancer_model

logger = logging.getLogger(__name__)

class _Executors:
    def __init__(self, vad: ThreadPoolExecutor, asr: ThreadPoolExecutor):
        self.vad = vad
        self.asr = asr
        # self.enhancer = enhancer


def _build_executors() -> _Executors:
    vad_workers = int(os.environ.get("VAD_WORKERS", str(cfg.get("vad_workers", 4))))
    asr_workers = int(os.environ.get("ASR_WORKERS", str(cfg.get("asr_workers", 8))))
    return _Executors(
        vad=ThreadPoolExecutor(max_workers=vad_workers, thread_name_prefix="vad"),
        asr=ThreadPoolExecutor(max_workers=asr_workers, thread_name_prefix="asr"),
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
        nacos_data_id=os.environ.get("NACOS_DATA_ID", "asr-server-test.yaml"),
        nacos_group=os.environ.get("NACOS_GROUP", "dolphin"),
    )
    load_model()
    load_vad_model()
    load_offline_model()
    load_enhancer_model()

    # 每个 worker 进程在 lifespan 内独立创建 executor，避免 fork 后继承父进程线程池
    executors = _build_executors()
    logger.info("dolphin-asr worker pid=%d executors: vad=%d asr=%d", os.getpid(),
                executors.vad._max_workers, executors.asr._max_workers)

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

    await app.state.http_client.aclose()
    await app.state.redis.aclose()
    executors.vad.shutdown(wait=False)
    executors.asr.shutdown(wait=False)
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

    control = data.get("control")
    if control == "start":
        logger.info("call_id=%s control=start: ASR resumed", call_id)
        call_conf = await _load_model_conf(websocket.app.state.redis, handler.model_id)
        handler.load_conf(call_conf)
        handler.resume()
        return False

    if control == "pause":
        logger.info("call_id=%s control=pause: ASR paused", call_id)
        handler.pause()
        return False

    if control == "stop":
        logger.info("call_id=%s control=stop: closing websocket", call_id)
        handler.pause()
        await websocket.close()
        return True

    logger.warning("call_id=%s unknown control=%r", call_id, control)
    return False


@app.websocket("/ws/asr")
async def ws_asr(websocket: WebSocket, call_id: str, uuid: str , model_id: int = 0):
    await websocket.accept()
    handler = StreamHandler(
        call_id=call_id,
        uuid=uuid,
        model_id=model_id,
        http_client=websocket.app.state.http_client,
        vad_executor=websocket.app.state.executor.vad,
        asr_executor=websocket.app.state.executor.asr,
    )
    # handler.start_processing()
    logger.info("WebSocket connected: call_id=%s", call_id)
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            raw = message.get("text")
            audio_bytes = message.get("bytes")
            # logger.info("headers : %s", str(websocket.headers))


            if msg_type == "websocket.disconnect":
                logger.info("WebSocket message indicates disconnect: call_id=%s", call_id)
                break

            if raw is not None:
                # logger.info("call_id=%s routing ws text frame to event handler", call_id)
                should_close = await _handle_ws_event(websocket, handler, raw, call_id)
                if should_close:
                    break
                continue

            if audio_bytes is not None:
                # logger.info("call_id=%s routing ws binary frame to audio handler: bytes=%d", call_id, len(audio_bytes))
                await handler.handle_audio(audio_bytes)
                continue

            logger.warning("call_id=%s ws message ignored: neither text nor bytes present", call_id)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: call_id=%s", call_id)
    except Exception as e:
        logger.error("WebSocket error call_id=%s: %s", call_id, e)
    finally:
        await handler.close()
