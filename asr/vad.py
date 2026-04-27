import struct
import time
import logging

import numpy as np

logger = logging.getLogger(__name__)

_vad_model = None
_VAD_SAMPLE_RATE = 16000
_MIN_FEATURE_WINDOW_SAMPLES = 400  # 16kHz 下 25ms，至少要能凑出 1 帧特征
_SHORT_CHUNK_THRESHOLD_SAMPLES = 640  # 40ms 以下视为短包，走滑动窗无状态 VAD
_SHORT_VAD_WINDOW_SAMPLES = 640  # 最近 40ms 窗口


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
        self._is_speaking = False
        self._cache_frame_count: int = 0
        self._cache_reset_interval_frames: int = 6000
        self._recent_audio = bytearray()

    def reset(self) -> None:
        self._speech_start = None
        self._interrupted = False
        self._vad_cache = {}
        self._is_speaking = False
        self._cache_frame_count = 0
        self._recent_audio.clear()

    @staticmethod
    def _extract_events(result) -> list:
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return first.get("value") or []
        return []

    def _run_streaming_vad(self, audio_bytes: bytes) -> bool:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_ms = max(1, len(audio_np) // 16)
        result = _vad_model.generate(
            input=audio_np,
            cache=self._vad_cache,
            is_final=False,
            chunk_size=chunk_ms,
            disable_pbar=True,
        )
        self._cache_frame_count += 1
        if self._cache_frame_count >= self._cache_reset_interval_frames:
            self._vad_cache = {}
            self._cache_frame_count = 0

        events = self._extract_events(result)
        if events:
            for event in events:
                start, end = event[0], event[1]
                if start != -1 and end == -1:
                    self._is_speaking = True
                elif start == -1 and end != -1:
                    self._is_speaking = False
        return self._is_speaking

    def _run_short_chunk_vad(self, audio_bytes: bytes) -> bool:
        self._vad_cache = {}
        self._cache_frame_count = 0

        self._recent_audio.extend(audio_bytes)
        max_recent_bytes = _SHORT_VAD_WINDOW_SAMPLES * 2
        if len(self._recent_audio) > max_recent_bytes:
            del self._recent_audio[:-max_recent_bytes]

        buffered_samples = len(self._recent_audio) // 2
        if buffered_samples < _MIN_FEATURE_WINDOW_SAMPLES:
            logger.debug(
                "VAD buffering short frame: buffered_samples=%d min_required=%d",
                buffered_samples,
                _MIN_FEATURE_WINDOW_SAMPLES,
            )
            return self._is_speaking

        window_audio_bytes = bytes(self._recent_audio)
        audio_np = np.frombuffer(window_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_ms = max(1, len(audio_np) // 16)
        result = _vad_model.generate(
            input=audio_np,
            cache={},
            is_final=False,
            chunk_size=chunk_ms,
            disable_pbar=True,
        )
        self._is_speaking = bool(self._extract_events(result))
        return self._is_speaking

    def is_speech(self, audio_bytes: bytes) -> bool:
        """解析 FunASR 输出事件，更新 _is_speaking 状态"""
        if len(audio_bytes) < 2:
            return self._is_speaking

        if _vad_model is not None:
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
                if not audio_bytes:
                    return self._is_speaking

            num_samples = len(audio_bytes) // 2
            try:
                if num_samples < _SHORT_CHUNK_THRESHOLD_SAMPLES:
                    return self._run_short_chunk_vad(audio_bytes)
                return self._run_streaming_vad(audio_bytes)

            except Exception as e:
                self._vad_cache = {}
                self._cache_frame_count = 0
                logger.warning("FSMN-VAD inference failed, fallback to energy: %s", e)

        # 能量降级逻辑
        from config import nacos_config as cfg
        energy_threshold = cfg.get("vad_energy_threshold", 1500)
        samples = struct.unpack_from(f"{len(audio_bytes) // 2}h", audio_bytes)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        # 能量判断只能代表当前帧，你可以用它临时覆盖状态
        self._is_speaking = rms > energy_threshold
        return self._is_speaking

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
