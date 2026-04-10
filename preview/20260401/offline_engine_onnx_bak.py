"""ASR engine — FunASR Paraformer Offline integration."""
import logging
import numpy as np
from config import nacos_config as cfg

logger = logging.getLogger(__name__)

_offline_model = None


def load_offline_model() -> None:
    """启动时加载一次全局离线 ASR 模型。"""
    global _offline_model
    import funasr.models.paraformer.model
    from funasr import AutoModel
    from funasr_onnx import Paraformer
    from config import nacos_config as cfg

    # 推荐填写 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' 或其 onnx 版本
    # model_path = cfg.get("offline_asr_model_path")
    model_path = cfg.get("offline_asr_model_path")
    device = cfg.get("asr_device", "cpu")

    logger.info("Loading Offline ASR model from %s on device=%s", model_path, device)
    _offline_model = Paraformer(
        model_dir=model_path,
        batch_size=1,
        quantize=True,  # 请检查你的模型目录下是否有 model_quant.onnx，如果有，请将此处设为 True
        device_id=0  # 默认 -1 代表 CPU 推理。如果想用 GPU 加速请设为 0
    )
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
            # 【核心修改点 1】：直接调用模型实例，并用列表 [] 包裹输入以适配 batch 维度
            result = _offline_model(audio_np)

            # 【核心修改点 2】：安全解析 funasr_onnx 的返回格式
            # funasr_onnx 通常返回 [['识别出的文本']] 或者 ['识别出的文本']
            if result and len(result) > 0 and isinstance(result[0], dict):
                # 提取 preds 字段的内容
                raw_text = result[0].get("preds", "")

                if raw_text:
                    # 去除字与字之间的空格，将 "你 好 你 好" 变成 "你好你好"
                    final_text = "".join(raw_text.split())

                    logger.info("Offline ASR result: %s", final_text)
                    return final_text

        except Exception as e:
            logger.error("Offline ASR inference failed: %s", e)

        return ""
