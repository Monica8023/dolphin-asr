"""
/ws/asr 接口功能演示客户端

用法：
    python demo_ws_client.py [--host 127.0.0.1] [--port 8080] [--call-id 1001]
                             [--intent-port 8808] [--mock-port 9000]
                             [--intent-text 我要转账] [--intent-id intent_transfer]

演示内容：
    1. 启动本地 Mock HTTP 服务（:9000，接收 callback / interrupt 推送）
    2. 启动本地 Mock Intent 服务（:8808，模拟意图识别，返回命中结果）
    3. 连接 WebSocket /ws/asr?call_id=<id>&uuid=<id>
    4. 发送模拟 PCM 音频帧（有声帧触发 VAD，静音帧触发停顿检测 → Intent 调用）
    5. Intent 服务返回命中意图 → 服务端推送 callback → demo 打印结果

依赖：
    pip install websockets
"""

import argparse
import asyncio
import json
import logging
import math
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

# ──────────────────────────────────────────────
# 全局配置（由 argparse 填充）
# ──────────────────────────────────────────────

MOCK_PORT = 19000      # callback / interrupt 接收端口
INTENT_PORT = 18808    # mock intent 服务端口
INTENT_TEXT = "我要转账"   # 模拟 ASR 识别出的文本（注入到 engine.transcribe）
INTENT_ID = "intent_transfer"  # mock intent 服务返回的意图 ID

received_events: list[dict] = []


# ──────────────────────────────────────────────
# Mock HTTP 服务：callback / interrupt
# ──────────────────────────────────────────────

class MockCallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            data = {"raw": body.decode(errors="replace")}

        received_events.append({"path": self.path, "data": data})
        logger.info("[CALLBACK] POST %s  %s", self.path, json.dumps(data, ensure_ascii=False))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"code":0}')

    def log_message(self, fmt, *args):
        pass


# ──────────────────────────────────────────────
# Mock Intent 服务：/api/v1/recognize
# ──────────────────────────────────────────────

class MockIntentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except Exception:
            req = {}

        text = req.get("text", "")
        # 只要文本非空就返回命中意图（模拟真实意图识别）
        if text:
            intent_id = INTENT_ID
            logger.info("[INTENT]  text=%r  → intent_id=%s", text, intent_id)
        else:
            intent_id = "intent_unknown"
            logger.info("[INTENT]  text=''  → intent_unknown")

        resp = json.dumps({"intent_id": intent_id, "text": text}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):
        pass


def _start(handler_cls, port: int, label: str):
    server = HTTPServer(("0.0.0.0", port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("%s listening on :%d", label, port)


# ──────────────────────────────────────────────
# PCM 音频帧生成
# ──────────────────────────────────────────────

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2                  # 16-bit PCM


def make_silence_frame() -> bytes:
    return bytes(FRAME_BYTES)


def make_speech_frame(freq: float = 440.0, amplitude: int = 8000) -> bytes:
    """正弦波，RMS ≈ 5657，远超 VAD 阈值 500"""
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        for i in range(FRAME_SAMPLES)
    ]
    return struct.pack(f"{FRAME_SAMPLES}h", *samples)


# ──────────────────────────────────────────────
# 注入 engine.transcribe（让占位实现返回文本）
# ──────────────────────────────────────────────

def _patch_engine(text: str):
    """
    engine.transcribe 是占位实现，始终返回 ""。
    这里在 demo 进程内直接替换，让有声帧期间返回指定文本，
    模拟真实 ASR 识别结果，从而触发 Intent 调用流程。
    """
    import asr.engine as engine_mod

    _call_count = {"n": 0}

    def fake_transcribe(audio_bytes: bytes) -> str:
        # 每 10 帧（200ms）返回一次文本，避免刷屏
        _call_count["n"] += 1
        if _call_count["n"] % 10 == 0:
            return text
        return ""

    engine_mod.transcribe = fake_transcribe
    logger.info("[PATCH] engine.transcribe patched → returns %r every 10 frames", text)


# ──────────────────────────────────────────────
# WebSocket 客户端
# ──────────────────────────────────────────────

async def run_demo(host: str, port: int, call_id: int):
    url = f"ws://{host}:{port}/ws/asr?call_id={call_id}&uuid={call_id}"
    logger.info("Connecting to %s", url)

    try:
        async with websockets.connect(url, ping_interval=None) as ws:
            logger.info("Connected.")

            # Phase 1：1s 有声帧 → ASR 产生文本，启动 match_timeout 计时器
            logger.info("--- Phase 1: 1s speech (ASR text generated) ---")
            for _ in range(50):   # 50 * 20ms = 1s
                await ws.send(make_speech_frame())
                await asyncio.sleep(0.02)

            # Phase 2：1s 静音 → 停顿检测触发（silence_max_ms=800ms）→ 整句推 Intent
            logger.info("--- Phase 2: 1s silence (sentence end → Intent call) ---")
            for _ in range(50):
                await ws.send(make_silence_frame())
                await asyncio.sleep(0.02)

            # Phase 3：等待 Intent 响应 + callback 推送
            logger.info("--- Phase 3: waiting for intent callback (up to 3s) ---")
            for _ in range(30):
                await asyncio.sleep(0.1)
                if any(ev["path"] == "/callback" for ev in received_events):
                    logger.info("Intent callback received, done.")
                    break

            logger.info("Closing connection.")

    except ConnectionRefusedError:
        logger.error("Connection refused — is the server running at %s:%d?", host, port)
        return
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        return

    print("\n" + "=" * 55)
    print(f"Demo complete. Received {len(received_events)} event(s):")
    for i, ev in enumerate(received_events, 1):
        print(f"  [{i}] {ev['path']}  {json.dumps(ev['data'], ensure_ascii=False)}")
    print("=" * 55)


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    global MOCK_PORT, INTENT_PORT, INTENT_TEXT, INTENT_ID

    parser = argparse.ArgumentParser(description="dolphin-asr /ws/asr demo")
    parser.add_argument("--host", default="127.0.0.1", help="ASR server host")
    parser.add_argument("--port", type=int, default=8080, help="ASR server port")
    parser.add_argument("--call-id", type=int, default=1001)
    parser.add_argument("--mock-port", type=int, default=9000, help="callback/interrupt mock port")
    parser.add_argument("--intent-port", type=int, default=8808, help="mock intent service port")
    parser.add_argument("--intent-text", default="我要转账", help="text injected into engine.transcribe")
    parser.add_argument("--intent-id", default="intent_transfer", help="intent_id returned by mock intent service")
    args = parser.parse_args()

    MOCK_PORT = args.mock_port
    INTENT_PORT = args.intent_port
    INTENT_TEXT = args.intent_text
    INTENT_ID = args.intent_id

    _patch_engine(INTENT_TEXT)
    _start(MockCallbackHandler, MOCK_PORT, "Mock callback/interrupt server")
    _start(MockIntentHandler, INTENT_PORT, "Mock intent server")

    asyncio.run(run_demo(args.host, args.port, args.call_id))


if __name__ == "__main__":
    main()
