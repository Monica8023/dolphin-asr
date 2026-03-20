# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8080

# 启动（开发模式，热重载）
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# 代码检查
ruff check .

# 类型检查
mypy .
```

## 配置

配置优先级：Nacos > `config/config.yaml` > `config/nacos_config.py` 中的 `DEFAULT_CONFIG`。

Nacos 通过环境变量启用：

| 环境变量 | 说明 |
|----------|------|
| `NACOS_SERVER` | Nacos 地址，不设则跳过 Nacos，直接读本地 config.yaml |
| `NACOS_NAMESPACE` | 命名空间（默认 `asr_test`） |
| `NACOS_DATA_ID` | Data ID（默认 `asr-server.yaml`） |
| `NACOS_GROUP` | Group（默认 `dolphin`） |

本地修改配置只需编辑 `config/config.yaml`，无需改代码。

主要配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `intent_service_url` | `http://127.0.0.1:8808` | 意图识别服务地址 |
| `business_callback_url` | `http://127.0.0.1:9000/callback` | 业务层回调地址（意图命中、无应答、匹配超时均推此地址） |
| `interrupt_url` | `http://127.0.0.1:9000/interrupt` | 打断请求地址 |
| `interrupt_enabled` | `true` | 打断功能开关，关闭后不发打断请求 |
| `vad_interrupt_threshold_ms` | `2000` | 连续说话超过此时长触发无意图打断 |
| `silence_max_ms` | `800` | 静音超过此时长判定一句话结束，触发 Intent 调用 |
| `no_answer_timeout_ms` | `10000` | 连接后超过此时长未收到有效文本，推送 `event=no_answer` |
| `match_timeout_ms` | `15000` | 首次收到文本后超过此时长未命中意图，推送 `event=match_timeout` |
| `log_path` | `./log` | 日志目录 |
| `log_level` | `INFO` | 日志级别 |

## 架构概览

每路 WebSocket 连接（`/ws/asr?call_id=<int>`）独立持有一个 `StreamHandler` 实例，处理流程：

```
WebSocket 连接建立
    │
    └─► start_timers()：启动无应答计时器（no_answer_timeout_ms）

每帧音频
    │
    ├─► VADDetector.process()        [interrupt_enabled=true 时生效]
    │       └─ 连续说话 >= vad_interrupt_threshold_ms → POST interrupt_url
    │
    ├─► engine.transcribe()          ← 占位，待接入真实模型
    │       └─ 有文本 → 复位无应答计时器
    │                → 首次文本：启动匹配时长计时器（match_timeout_ms）
    │
    ├─► 停顿检测：静音 >= silence_max_ms 且有缓冲文本
    │       └─ 整句推 Intent /api/v1/recognize
    │               ├─ intent_unknown → 继续
    │               └─ 命中意图 → _paused=True → stop_timers() → POST business_callback_url
    │
    └─► 计时器到期（asyncio.Task）
            ├─ no_answer_timeout_ms 到期 → POST event=no_answer
            └─ match_timeout_ms 到期   → POST event=match_timeout
```

**关键设计点：**

- `StreamHandler._paused`：意图命中或超时后置为 `True`，后续音频帧直接丢弃。
- VAD、Intent 调用、计时器均通过 `asyncio.create_task()` 异步触发，不阻塞音频接收循环。
- `VADDetector._interrupted` 确保同一段连续说话只触发一次打断。
- Intent 以整句（停顿判定后）为单位调用，而非逐帧调用。
- 配置全局单例在 `config/nacos_config.py`，所有模块通过 `cfg.get(key)` 读取，支持 Nacos 热更新。

## Nacos 热更新

`config/nacos_config.py` 使用 `nacos-sdk-python` v3（import 路径为 `v2.nacos`，非 `nacos`），通过 gRPC 连接 Nacos 服务端（端口 `HTTP+1000`，即 8848 → 9848）。

热更新双保险机制：
- **gRPC push**：Nacos 服务端主动推送，实时生效（需 9848 端口可达）
- **轮询兜底**：每 30s 主动拉取一次（`poll_interval_s` 可配置），gRPC 不通时保底

配置变更时日志格式：
```
Config changed [nacos-poll]: business_callback_url: 'old' -> 'new'
Config changed [nacos-grpc-push]: silence_max_ms: 800 -> 1200
```
`source` 标签说明来源：`nacos-init`（启动加载）、`nacos-grpc-push`（推送）、`nacos-poll`（轮询）、`local-fallback`（本地降级）。无变化时只打 DEBUG 级别，不刷屏。

`main.py` lifespan 中 `setup_logging()` 必须在 `init_config()` 之前调用，否则启动日志会因 handler 未注册而丢失。

## 待实现

- `asr/engine.py` 中的 `load_model()` 和 `transcribe()`：接入真实 ASR 模型。
- `asr/vad.py` 中的 `is_speech()`：替换为真实 VAD（如 silero-vad）。
