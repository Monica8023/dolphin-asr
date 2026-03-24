import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config import nacos_config as cfg
from log_config import setup_logging
from asr.engine import load_model
from asr.vad import load_vad_model
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

    # P0-fix-3: 全局 httpx 连接池，所有 StreamHandler 共用
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
        timeout=httpx.Timeout(5.0),
    )
    app.state.executor = _executor

    logger.info("dolphin-asr service started.")
    yield

    await app.state.http_client.aclose()
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


@app.websocket("/ws/asr")
async def ws_asr(websocket: WebSocket, call_id: int, uuid: int = 0):
    await websocket.accept()
    handler = StreamHandler(
        call_id=call_id,
        uuid=uuid,
        http_client=websocket.app.state.http_client,
        executor=websocket.app.state.executor,
    )
    handler.start_timers()
    logger.info("WebSocket connected: call_id=%s", call_id)
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            await handler.handle_audio(audio_bytes)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: call_id=%s", call_id)
    except Exception as e:
        logger.error("WebSocket error call_id=%s: %s", call_id, e)
    finally:
        await handler.close()
