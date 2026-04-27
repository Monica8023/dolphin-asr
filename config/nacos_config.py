import asyncio
import logging
import os
import threading
from typing import Any

import yaml

logger = logging.getLogger("dolphin.nacos_config")

DEFAULT_CONFIG = {
    "intent_service_url": "http://127.0.0.1:8808",
    "business_callback_url": "http://127.0.0.1:9000/callback",
    "interrupt_url": "http://127.0.0.1:9000/interrupt",
    "transcript_url": "http://127.0.0.1:9000/transcript",
    "vad_interrupt_threshold_ms": 2000,
    "interrupt_enabled": True,
    "silence_max_ms": 350,
    "no_answer_timeout_ms": 10000,
    "match_timeout_ms": 15000,
    "log_path": "./log",
    "log_level": "info",
    "asr_model_path": "D:/model/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "vad_model_path": "D:/model/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "enhancer_model_path": "D:/model/iic/speech_zipenhancer_ans_multiloss_16k_base",
    "offline_asr_model_path":"D:/model/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "asr_device": "cpu",
    "asr_chunk_size": [0, 6, 2],
    "asr_encoder_chunk_look_back": 2,
    "asr_decoder_chunk_look_back": 0,
    "intent_epoch_guard_enabled": True,
    "asr_workers": 8,
    "vad_workers": 4,
    "vad_energy_threshold": 500,
    "vad_gate_asr": False,
    "audio_queue_maxsize": 64,
}

_config: dict[str, Any] = {}
_lock = threading.Lock()


def _load_local_fallback() -> dict[str, Any]:
    fallback_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(fallback_path):
        with open(fallback_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded config from local fallback: %s", fallback_path)
        return data
    return {}


def _apply_config(data: dict[str, Any], source: str = "") -> None:
    new = {**DEFAULT_CONFIG, **data}
    tag = f" [{source}]" if source else ""
    with _lock:
        old = dict(_config)
        _config.clear()
        _config.update(new)

    if not old:
        logger.info("Config initialized%s", tag)
        return

    diffs = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
    if diffs:
        logger.info("Config hot-reload%s — %d item(s) changed:", tag, len(diffs))
        for k, (before, after) in diffs.items():
            logger.info("  %-30s %r  ->  %r", k, before, after)
    else:
        logger.debug("Config reloaded%s: no changes.", tag)


async def init_config(
    nacos_server: str = "nacos.register.service.com:8848",
    nacos_namespace: str = "asr_test",
    nacos_data_id: str = "asr-server.yaml",
    nacos_group: str = "dolphin",
    poll_interval_s: int = 30,
) -> None:
    """Initialize config from Nacos v2, falling back to local config.yaml.
    同时启动轮询兜底，每 poll_interval_s 秒主动拉取一次，防止 gRPC 推送不可达。
    """
    if nacos_server:
        try:
            from v2.nacos import NacosConfigService, ClientConfigBuilder, ConfigParam  # type: ignore

            client_config = (
                ClientConfigBuilder()
                .server_address(nacos_server)
                .namespace_id(nacos_namespace)
                .build()
            )
            client_config.disable_use_config_cache = True

            svc = await NacosConfigService.create_config_service(client_config)

            raw = await svc.get_config(ConfigParam(data_id=nacos_data_id, group=nacos_group))
            if raw:
                _apply_config(yaml.safe_load(raw) or {}, source="nacos-init")
                logger.info("Nacos config loaded: data_id=%s group=%s", nacos_data_id, nacos_group)
            else:
                logger.warning("Nacos returned empty config, using local fallback.")
                _apply_config(_load_local_fallback(), source="local-fallback")

            async def _on_change(tenant, data_id, group, content):
                if not content:
                    return
                try:
                    updated = yaml.safe_load(content) or {}
                    if not updated:
                        return
                    _apply_config(updated, source="nacos-grpc-push")
                except Exception as e:
                    logger.error("Failed to reload Nacos config: %s", e)

            await svc.add_listener(nacos_data_id, nacos_group, _on_change)

            # 轮询兜底：防止 gRPC 推送不可达时配置无法更新
            async def _poll_loop():
                while True:
                    await asyncio.sleep(poll_interval_s)
                    try:
                        latest, _ = await svc.grpc_client_proxy.query_config(nacos_data_id, nacos_group)
                        if latest:
                            new_data = yaml.safe_load(latest) or {}
                            with _lock:
                                current = dict(_config)
                            merged = {**DEFAULT_CONFIG, **new_data}
                            if merged != current:
                                _apply_config(new_data, source="nacos-poll")
                    except Exception as e:
                        logger.warning("Nacos poll failed: %s", e)

            asyncio.create_task(_poll_loop())
            return
        except Exception as e:
            logger.warning("Nacos unavailable (%s), falling back to local config.", e)

    _apply_config(_load_local_fallback(), source="local-fallback")


def get(key: str, default: Any = None) -> Any:
    with _lock:
        return _config.get(key, DEFAULT_CONFIG.get(key, default))
