import time
import logging
import struct

logger = logging.getLogger(__name__)


class VADDetector:
    """Tracks continuous speech duration per call and triggers interrupt when threshold exceeded."""

    def __init__(self, threshold_ms: int = 2000):
        self.threshold_ms = threshold_ms
        self._speech_start: float | None = None
        self._interrupted = False

    def reset(self) -> None:
        self._speech_start = None
        self._interrupted = False

    def is_speech(self, audio_bytes: bytes) -> bool:
        """判断当前帧是否为语音（能量阈值，待替换为真实 VAD）。"""
        if len(audio_bytes) < 2:
            return False
        samples = struct.unpack_from(f"{len(audio_bytes) // 2}h", audio_bytes)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms > 500

    def process(self, audio_bytes: bytes) -> bool:
        """
        喂入一帧音频，返回 True 表示本次触发打断（连续说话超过阈值，且只触发一次）。
        """
        speech = self.is_speech(audio_bytes)

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
                    elapsed_ms, self.threshold_ms,
                )
                return True
        else:
            if self._speech_start is not None:
                logger.debug("VAD: speech ended")
            self._speech_start = None
            self._interrupted = False

        return False
