"""ASR engine — FunASR Paraformer Offline integration."""
import logging
import numpy as np

logger = logging.getLogger(__name__)

_offline_model = None


def load_offline_model() -> None:
    """启动时加载一次全局离线 ASR 模型。"""
    global _offline_model
    import funasr.models.paraformer.model
    from funasr import AutoModel
    from config import nacos_config as cfg

    # 推荐填写 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' 或其 onnx 版本
    model_path = cfg.get("offline_asr_model_path")
    device = cfg.get("asr_device", "cpu")

    logger.info("Loading Offline ASR model from %s on device=%s", model_path, device)
    _offline_model = AutoModel(model=model_path, device=device, disable_update=True)
    logger.info("Offline ASR model loaded.")


class OfflineASREngine:
    """无状态的离线推理引擎，可以直接复用全局模型。"""

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        输入一整句完整的 PCM16 音频，返回高精度的识别文本。
        """
        if _offline_model is None or not audio_bytes:
            return ""

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            result = _offline_model.generate(
                input=audio_np,
                disable_pbar=True,
            )
            if result and len(result) > 0 and result[0].get("text"):
                return "".join(result[0]["text"].split())
        except Exception as e:
            logger.error("Offline ASR inference failed: %s", e)

        return ""