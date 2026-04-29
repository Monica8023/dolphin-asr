import struct
import logging

import numpy as np

logger = logging.getLogger(__name__)

_vad_model = None
_VAD_SAMPLE_RATE = 16000
_SHORT_CHUNK_WINDOW_MS = 60
_SHORT_CHUNK_WINDOW_SAMPLES = _VAD_SAMPLE_RATE * _SHORT_CHUNK_WINDOW_MS // 1000  # 960 samples


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

    def __init__(self, threshold_ms: int = 2000, silence_tolerance_ms: int = 200):
        self.threshold_ms = int(threshold_ms)
        self.silence_tolerance_ms = max(0, int(silence_tolerance_ms))
        self._speech_active = False
        self._speech_elapsed_ms = 0
        self._speech_gap_ms = 0
        self._last_progress_log_ms = 0
        self._interrupted = False
        self._vad_cache: dict = {}
        self._is_speaking = False
        self._cache_frame_count: int = 0
        self._cache_reset_interval_frames: int = 6000
        self._short_chunk_history = bytearray()

    def reset(self) -> None:
        self.reset_interrupt_state()
        self.reset_detection_state()

    def reset_interrupt_state(self) -> None:
        """重置打断连续说话累计状态。"""
        self._speech_active = False
        self._speech_elapsed_ms = 0
        self._speech_gap_ms = 0
        self._last_progress_log_ms = 0
        self._interrupted = False

    def reset_detection_state(self) -> None:
        """重置 FSMN-VAD 检测缓存，不影响打断连续说话累计。"""
        self._vad_cache = {}
        self._is_speaking = False
        self._cache_frame_count = 0
        self._short_chunk_history.clear()

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
        self._short_chunk_history.extend(audio_bytes)
        max_history_bytes = _SHORT_CHUNK_WINDOW_SAMPLES * 2
        if len(self._short_chunk_history) > max_history_bytes:
            del self._short_chunk_history[:-max_history_bytes]

        buffered_samples = len(self._short_chunk_history) // 2
        if buffered_samples < _SHORT_CHUNK_WINDOW_SAMPLES:
            logger.debug(
                "VAD buffering short chunk: buffered_samples=%d min_required=%d",
                buffered_samples,
                _SHORT_CHUNK_WINDOW_SAMPLES,
            )
            return self._is_speaking

        window_audio = bytes(self._short_chunk_history)
        audio_np = np.frombuffer(window_audio, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_ms = max(1, len(audio_np) // 16)
        result = _vad_model.generate(
            input=audio_np,
            cache={},
            is_final=False,
            chunk_size=chunk_ms,
            disable_pbar=True,
        )
        events = self._extract_events(result)
        if events:
            for event in events:
                start, end = event[0], event[1]
                if start != -1 and end == -1:
                    self._is_speaking = True
                elif start == -1 and end != -1:
                    self._is_speaking = False
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
                if num_samples < _SHORT_CHUNK_WINDOW_SAMPLES:
                    return self._run_short_chunk_vad(audio_bytes)
                return self._run_streaming_vad(audio_bytes)

            except Exception as e:
                self._vad_cache = {}
                self._cache_frame_count = 0
                if num_samples < _SHORT_CHUNK_WINDOW_SAMPLES:
                    logger.warning("FSMN-VAD short-chunk inference failed, fallback to energy: %s", e)
                else:
                    logger.warning("FSMN-VAD inference failed, fallback to energy: %s", e)

        # 能量降级逻辑
        from config import nacos_config as cfg
        energy_threshold = cfg.get("vad_energy_threshold", 1500)
        samples = struct.unpack_from(f"{len(audio_bytes) // 2}h", audio_bytes)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        # 能量判断只能代表当前帧，你可以用它临时覆盖状态
        self._is_speaking = rms > energy_threshold
        return self._is_speaking

    def process_speech(self, speech: bool, chunk_ms: int = 0, allow_trigger: bool = True) -> bool:
        """基于已计算出的 speech 状态累计音频时长，返回是否触发打断。"""
        if speech:
            if not self._speech_active:
                self._speech_active = True
                self._speech_elapsed_ms = 0
                self._speech_gap_ms = 0
                self._last_progress_log_ms = 0
                self._interrupted = False
                logger.debug("VAD: speech started")

            self._speech_gap_ms = 0
            self._speech_elapsed_ms += max(0, int(chunk_ms or 0))
            if (
                allow_trigger
                and not self._interrupted
                and self._speech_elapsed_ms - self._last_progress_log_ms >= 500
            ):
                self._last_progress_log_ms = self._speech_elapsed_ms
                logger.info(
                    "VAD: continuous speech progress %d/%d ms",
                    self._speech_elapsed_ms,
                    self.threshold_ms,
                )
            if allow_trigger and self._speech_elapsed_ms >= self.threshold_ms and not self._interrupted:
                self._interrupted = True
                logger.info(
                    "VAD: continuous speech %d ms >= threshold %d ms, triggering interrupt",
                    self._speech_elapsed_ms,
                    self.threshold_ms,
                )
                return True
        else:
            gap_ms = max(0, int(chunk_ms or 0))
            if self._speech_active and gap_ms and self._speech_gap_ms + gap_ms <= self.silence_tolerance_ms:
                self._speech_gap_ms += gap_ms
                logger.debug(
                    "VAD: speech gap tolerated %d/%d ms",
                    self._speech_gap_ms,
                    self.silence_tolerance_ms,
                )
                return False
            if self._speech_active:
                logger.debug("VAD: speech ended after %d ms", self._speech_elapsed_ms)
            self._speech_active = False
            self._speech_elapsed_ms = 0
            self._speech_gap_ms = 0
            self._last_progress_log_ms = 0
            self._interrupted = False

        return False

    def process(self, audio_bytes: bytes) -> bool:
        """
        喂入一帧音频，返回 True 表示本次触发打断（连续说话超过阈值，且只触发一次）。
        """
        chunk_ms = int((len(audio_bytes) // 2) / _VAD_SAMPLE_RATE * 1000)
        return self.process_speech(self.is_speech(audio_bytes), chunk_ms)
