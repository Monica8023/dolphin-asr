# 企业级高并发 ASR 架构优化报告

> 生成日期：2026-03-20
> 分析范围：dolphin-asr 项目全量代码

---

## 优化优先级汇总

| 优先级 | 问题 | 预期收益 |
|--------|------|---------|
| P0 | Event Loop 阻塞 | 消除全局卡顿，直接影响所有连接 |
| P0 | HTTP 连接池 | 消除端口耗尽风险 |
| P0 | ASR Batching | GPU 吞吐率提升 3-10x |
| P1 | VAD numpy 优化 | 单帧处理速度提升 10x+ |
| P1 | Task 强引用管理 | 消除静默任务丢失 |
| P1 | 配置读锁优化 | 减少高频锁竞争 |
| P1 | 全局连接限流 | 防雪崩过载保护 |
| P2 | Prometheus 监控 | 线上可观测性 |
| P2 | 接入层/计算层解耦 | 独立弹性扩容 |

**建议实施顺序：** P0 三项 → P1 五项 → 接入真实 ASR 模型（`engine.py` / `vad.py`）→ P2 架构演进。

---

## P0 — 必须立即修复（阻断性问题）

### 1. CPU 密集型推理阻塞 Event Loop

**问题：** `StreamHandler.handle_audio()` 在 asyncio 事件循环中直接调用同步方法 `transcribe()` 和 `is_speech()`，会冻结整个事件循环，导致所有 WebSocket 连接的音频帧接收挂起，计时器延误。

**修复：**
```python
# 当前：阻塞调用
text = self._engine.transcribe(audio_bytes)

# 修改：放入线程池
loop = asyncio.get_running_loop()
text = await loop.run_in_executor(None, self._engine.transcribe, audio_bytes)
```

VAD 同理：
```python
is_speech = await loop.run_in_executor(None, self._vad.is_speech, audio_bytes)
```

---

### 2. ASR 模型缺乏 GPU Dynamic Batching

**问题：** 当接入 Whisper/Paraformer 等真实模型时，每条流单独推理，GPU 吞吐率极低，并发高时 OOM。缺乏显式加锁还会导致线程安全问题。

**修复方向：**
- 前端缓冲音频帧 → 后台 Worker 队列做批量推理（Dynamic Batching）
- 或将 ASR 独立为 Triton Inference Server / Ray Serve，通过 gRPC 调用

```
音频帧 → asyncio.Queue → BatchWorker(批量凑帧) → 模型推理(batch_size=8~32) → 结果分发
```

---

### 3. HTTP 客户端未复用 — 连接池雪崩

**问题：** `_call_intent` / `_post` 每次 `async with httpx.AsyncClient()` 创建新客户端，无法复用 TCP 连接。高并发下产生大量 TCP 三次握手和 `TIME_WAIT`，可能耗尽系统端口。

**修复：** 在 `lifespan` 中创建全局单例客户端：
```python
# main.py lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
        timeout=httpx.Timeout(5.0)
    )
    yield
    await app.state.http_client.aclose()
```

`StreamHandler` 通过 `request.app.state.http_client` 复用，不再自行创建。

---

## P1 — 高优先级（稳定性与性能关键）

### 4. VAD 纯 Python 计算性能低下

**问题：** `vad.py` 中使用 Python 生成器 `sum(s * s for s in samples)` 计算 RMS，高并发下 CPU 消耗过大，且受 GIL 限制。

**修复：**
```python
# 当前
samples = struct.unpack_from(f"<{num_samples}h", audio_bytes)
rms = math.sqrt(sum(s * s for s in samples) / num_samples)

# 优化：numpy 向量化（10x+ 加速）
import numpy as np
samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
rms = float(np.sqrt(np.mean(samples ** 2)))
```

进阶：接入 Silero VAD（ONNX Runtime）或 WebRTC VAD（C 扩展库 `webrtcvad`）。

---

### 5. asyncio.Task 未持有强引用 — 任务泄漏风险

**问题：** `create_task()` 后未保留引用，GC 可能在任务运行中途回收，导致意图回调或打断请求静默丢失。连接断开时后台网络请求也可能继续占用资源引发幽灵报错。

**修复：**
```python
class StreamHandler:
    def __init__(self):
        self._bg_tasks: set[asyncio.Task] = set()

    def _fire(self, coro) -> asyncio.Task:
        """创建后台任务并持有强引用，防止 GC 回收。"""
        t = asyncio.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)
        return t

    async def close(self):
        """连接断开时统一取消所有后台任务。"""
        for t in list(self._bg_tasks):
            t.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()
```

---

### 6. 配置读取锁竞争

**问题：** `cfg.get()` 使用 `threading.Lock()`，但每帧音频处理中多次调用配置读取，高并发下数千协程频繁争锁，造成性能损耗。

**修复：** 利用 GIL 对字典整体替换的原子性，取消读锁，仅在写时加锁：
```python
# nacos_config.py
def get(self, key, default=None):
    # 读操作无需加锁，字典引用替换在 CPython 中是原子的
    return self._config.get(key, default)

def _update_config(self, new_config: dict):
    # 写操作加锁，保证 _config 引用切换的可见性
    with self._lock:
        self._config = {**self._config, **new_config}
```

或在 `StreamHandler.__init__` 时缓存不变配置，仅对热更新字段注册回调。

---

### 7. 缺乏全局连接数限制 — 无过载保护

**问题：** 服务没有限制全局并发数上限，请求洪峰下 GPU 显存 OOM 会导致所有连接崩溃。

**修复：**
```python
# main.py
_conn_sem = asyncio.Semaphore(200)  # 最大并发连接数，按 GPU 显存调整

@app.websocket("/ws/asr")
async def ws_endpoint(ws: WebSocket, call_id: int):
    if _conn_sem.locked():
        await ws.accept()
        await ws.close(code=1013)  # 1013 = Try Again Later
        return
    async with _conn_sem:
        await handle_stream(ws, call_id)
```

---

## P2 — 架构演进（规模化必要投入）

### 8. 缺乏可观测性 — 线上问题难定位

引入 `prometheus_client`，对外暴露 `/metrics` 接口：

```python
from prometheus_client import Gauge, Histogram, Counter, make_asgi_app

# 核心埋点
active_connections = Gauge("asr_active_connections", "当前活跃 WebSocket 连接数")
asr_latency = Histogram("asr_inference_seconds", "ASR 推理耗时", buckets=[.05,.1,.25,.5,1,2.5,5])
intent_latency = Histogram("intent_api_seconds", "Intent API 回调延迟")
timeout_counter = Counter("asr_timeout_total", "超时事件计数", ["event_type"])
# event_type: no_answer, match_timeout, intent_hit, vad_interrupt
```

| 指标类型 | 埋点内容 |
|---------|---------|
| Gauge | 当前活跃 WebSocket 连接数 |
| Histogram | ASR 推理耗时、Intent API 回调延迟 |
| Counter | no_answer / match_timeout / intent_hit 触发次数 |
| Counter | VAD 打断触发次数 |

---

### 9. 接入层与计算层解耦 — 支持独立扩容

**当前问题：** WebSocket 长连接（高带宽 CPU 节点）和 ASR 推理（GPU 节点）强绑定在同一进程，扩容时必须按整机（含 GPU）扩容，且 ASR 推理卡顿会牵连同进程的所有连接。

**目标架构：**

```
               ┌─────────────────────────────────┐
Client ──WS──► │  Audio API Gateway (CPU 节点)    │
               │  main.py + StreamHandler         │
               │  状态机 / 计时器 / 回调           │
               │  水平扩展：无状态，可任意扩容      │
               └──────────┬──────────────────────┘
                          │ gRPC 流式 (音频帧)
               ┌──────────▼──────────────────────┐
               │  ASR Inference Service (GPU 节点) │
               │  Triton Inference Server          │
               │  / Ray Serve / 自研 Worker        │
               │  Dynamic Batching + 多实例        │
               │  GPU 独立扩缩容                   │
               └─────────────────────────────────┘
```

**收益：**
- CPU 节点与 GPU 节点可按需独立扩缩容，降低成本
- ASR 推理故障不影响 WebSocket 连接层（可熔断降级）
- GPU 利用率通过 Dynamic Batching 显著提升

---

## 附：待实现模块接口建议

### `asr/engine.py` — 真实 ASR 接入

```python
class ASREngine:
    def load_model(self) -> None:
        """加载模型到 GPU，推荐 half precision (fp16) 降低显存。"""
        ...

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        同步推理接口，调用方应通过 run_in_executor 异步化。
        返回识别文本，无结果返回空字符串。
        """
        ...
```

### `asr/vad.py` — 真实 VAD 接入

```python
class VADDetector:
    def is_speech(self, audio_bytes: bytes) -> bool:
        """
        判断音频帧是否包含人声。
        推荐：silero-vad (ONNX) 或 webrtcvad (C 扩展)。
        帧长度需符合模型要求（silero: 512/1024/1536 samples @ 16kHz）。
        """
        ...
```
