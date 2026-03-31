import struct
import time
import logging

import numpy as np

logger = logging.getLogger(__name__)

_vad_model = None


def load_vad_model() -> None:
    """Load FunASR FSMN-VAD model. Path is read from config."""
    global _vad_model
    from funasr import AutoModel
    from config import nacos_config as cfg

    model_path = cfg.get("vad_model_path")
    device = cfg.get("asr_device", "cpu")
    logger.info("Loading VAD model from %s on device=%s", model_path, device)
    _vad_model = AutoModel(model=model_path, device=device, disable_update=True)
    logger.info("VAD model loaded.")


class VADDetector:
    """Tracks continuous speech duration per call and triggers interrupt when threshold exceeded."""

    def __init__(self, threshold_ms: int = 2000):
        self.threshold_ms = threshold_ms
        self._speech_start: float | None = None
        self._interrupted = False
        self._vad_cache: dict = {}

    def reset(self) -> None:
        self._speech_start = None
        self._interrupted = False
        self._vad_cache = {}

    def is_speech(self, audio_bytes: bytes) -> bool:
        """判断当前帧是否为语音。优先使用 FSMN-VAD，不可用时降级为能量阈值。"""
        if len(audio_bytes) < 2:
            return False

        if _vad_model is not None:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            chunk_ms = max(1, len(audio_np) // 16)
            try:
                result = _vad_model.generate(
                    input=audio_np,
                    cache=self._vad_cache,
                    is_final=False,
                    chunk_size=chunk_ms,
                    disable_pbar=True,
                )
                # result[0]["value"] 为语音时间戳列表，非空即为语音帧
                return bool(result and result[0].get("value"))
            except Exception as e:
                logger.warning("FSMN-VAD inference failed, fallback to energy: %s", e)

        # 能量阈值降级
        from config import nacos_config as cfg
        energy_threshold = cfg.get("vad_energy_threshold", 500)
        samples = struct.unpack_from(f"{len(audio_bytes) // 2}h", audio_bytes)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms > energy_threshold

    def process_speech(self, speech: bool) -> bool:
        """基于已计算出的 speech 状态更新打断计时，返回是否触发打断。"""
        if speech:
            if self._speech_start is None:
                self._speech_start = time.monotonic()
                self._interrupted = False
                logger.debug("VAD: speech started")

            elapsed_ms = (time.monotonic() - self._speech_start) * 1000
            if elapsed_ms >= self.threshold_ms and not self._interrupted:
                self._interrupted = True
                logger.info(
                    "VAD: continuous speech %.0f ms >= threshold %d ms, triggering interrupt",
                    elapsed_ms,
                    self.threshold_ms,
                )
                return True
        else:
            if self._speech_start is not None:
                logger.debug("VAD: speech ended")
            self._speech_start = None
            self._interrupted = False

        return False

    def process(self, audio_bytes: bytes) -> bool:
        """
        喂入一帧音频，返回 True 表示本次触发打断（连续说话超过阈值，且只触发一次）。
        """
        return self.process_speech(self.is_speech(audio_bytes))
