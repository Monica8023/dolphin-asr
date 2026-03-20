import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from config import nacos_config as cfg
from log_config import setup_logging
from asr.engine import load_model
from asr.stream_handler import StreamHandler

logger = logging.getLogger(__name__)


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
    logger.info("dolphin-asr service started.")
    yield
    logger.info("dolphin-asr service stopped.")


app = FastAPI(title="dolphin-asr", lifespan=lifespan)


@app.websocket("/ws/asr")
async def ws_asr(websocket: WebSocket, call_id: int, uuid: int = 0):
    await websocket.accept()
    handler = StreamHandler(call_id=call_id, uuid=uuid)
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
        handler.stop_timers()
