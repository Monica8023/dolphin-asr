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

logger = logging.getLogger(__name__)

# P0-fix-1: 全局线程池，供 run_in_executor 执行 CPU 密集型推理
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("ASR_WORKERS", str(cfg.get("asr_workers", 8)))))


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
    load_offline_model()

    # P0-fix-3: 全局 httpx 连接池，所有 StreamHandler 共用
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
        timeout=httpx.Timeout(5.0),
    )
    app.state.executor = _executor

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
    _executor.shutdown(wait=False)
    logger.info("dolphin-asr service stopped.")


app = FastAPI(title="dolphin-asr", lifespan=lifespan)


@app.post("/test/intent")
async def test_intent(req: IntentTestRequest) -> dict[str, Any]:
    logger.info(f"Test intent {req.text} callId {req.call_id}")
    if os.environ.get("TEST_INTENT_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="not found")


    url = f"{cfg.get('intent_service_url')}/api/v1/recognize"
    payload = {"text": req.text, "call_id": req.call_id}

    preview = req.text if len(req.text) <= 100 else f"{req.text[:100]}..."
    logger.info("intent test input: call_id=%s text=%r", req.call_id, preview)

    try:
        resp = await app.state.http_client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException as e:
        logger.warning("intent test timeout: call_id=%s err=%s", req.call_id, e)
        raise HTTPException(status_code=504, detail="intent service timeout") from e
    except httpx.HTTPStatusError as e:
        logger.warning("intent test http error: call_id=%s status=%s", req.call_id, e.response.status_code)
        raise HTTPException(status_code=502, detail="intent service http error") from e
    except httpx.HTTPError as e:
        logger.warning("intent test request error: call_id=%s err=%s", req.call_id, e)
        raise HTTPException(status_code=502, detail="intent service request error") from e

    logger.info("intent test output: call_id=%s intent_id=%s", data.get("call_id"), data.get("intent_id", "intent_unknown"))
    return {"request": payload, "response": data}


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


async def _handle_ws_event(websocket: WebSocket, handler: StreamHandler, raw: str, call_id: int) -> bool:
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

    event = data.get("event")
    if event == "start":
        logger.info("call_id=%s event=start: ASR resumed", call_id)
        call_conf = await _load_model_conf(websocket.app.state.redis, handler.model_id)
        handler.load_conf(call_conf)
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


@app.websocket("/ws/asr")
async def ws_asr(websocket: WebSocket, call_id: int, uuid: int = 0, model_id: int = 0):
    await websocket.accept()
    handler = StreamHandler(
        call_id=call_id,
        uuid=uuid,
        model_id=model_id,
        http_client=websocket.app.state.http_client,
        executor=websocket.app.state.executor,
    )
    logger.info("WebSocket connected: call_id=%s", call_id)
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            text = message.get("text")
            audio_bytes = message.get("bytes")
            unknown_keys = sorted(k for k in message.keys() if k not in {"type", "text", "bytes"})

            # logger.info(
            #     "call_id=%s ws message: type=%s text_none=%s text_len=%d bytes_none=%s bytes_len=%d extra_keys=%s",
            #     call_id,
            #     msg_type,
            #     text is None,
            #     len(text) if text is not None else 0,
            #     audio_bytes is None,
            #     len(audio_bytes) if audio_bytes is not None else 0,
            #     unknown_keys,
            # )

            if msg_type == "websocket.disconnect":
                logger.info("WebSocket message indicates disconnect: call_id=%s", call_id)
                break

            if text is not None:
                # logger.info("call_id=%s routing ws text frame to event handler", call_id)
                should_close = await _handle_ws_event(websocket, handler, text, call_id)
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
