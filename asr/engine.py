"""ASR engine — FunASR ParaformerStreaming (online ONNX) integration."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

_model = None


def load_model() -> None:
    """启动时加载一次全局 ASR 模型，所有连接共享同一模型实例（线程安全只读）。"""
    global _model
    import funasr.models.paraformer_streaming.model  # noqa: F401 — 确保 ParaformerStreaming 注册到 tables
    from funasr import AutoModel
    from config import nacos_config as cfg

    model_path = cfg.get("asr_model_path")
    device = cfg.get("asr_device", "cpu")
    logger.info("Loading ASR model from %s on device=%s", model_path, device)
    _model = AutoModel(model=model_path, device=device, disable_update=True)
    logger.info("ASR model loaded.")


class ASREngine:
    """每路 WebSocket 连接独立一个实例，持有流式推理所需的 cache 状态。"""

    def __init__(self) -> None:
        self._cache: dict = {}

    def transcribe(self, audio_bytes: bytes, is_final: bool = False) -> str:
        """
        喂入一帧 PCM16 音频，返回当前帧的识别文本（可能为空）。

        audio_bytes: 16kHz, 16-bit PCM, mono, little-endian
        is_final: 连接断开时传 True，刷新模型内部缓冲区
        """
        if _model is None:
            return ""
        if not audio_bytes and not is_final:
            return ""

        from config import nacos_config as cfg

        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if audio_bytes
            else np.zeros(0, dtype=np.float32)
        )

        chunk_size = cfg.get("asr_chunk_size", [0, 10, 5])
        encoder_look_back = cfg.get("asr_encoder_chunk_look_back", 4)
        decoder_look_back = cfg.get("asr_decoder_chunk_look_back", 1)

        result = _model.generate(
            input=audio_np,
            cache=self._cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_look_back,
            decoder_chunk_look_back=decoder_look_back,
            disable_pbar=True,
        )
        if result and result[0].get("text"):
            return result[0]["text"].strip()
        return ""

    def reset(self) -> None:
        """重置流式状态（通话结束或重新开始时调用）。"""
        self._cache = {}
