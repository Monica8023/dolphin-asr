"""RNNoise 实时音频降噪预处理。

使用 pyrnnoise 底层 C 接口（pyrnnoise.rnnoise），绕过有兼容性问题的 audiolab 依赖。
RNNoise 原生采样率为 48kHz，内部做 16kHz ↔ 48kHz 重采样（scipy.signal.resample_poly）。

安装：pip install pyrnnoise
"""
import logging

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

# RNNoise 帧大小（48kHz 下 480 samples = 10ms）
_RNNOISE_FRAME_SIZE = 480
_RNNOISE_RATE = 48000
_ASR_RATE = 16000
# 重采样比：48k/16k = 3/1
_UP = 3
_DOWN = 1


class RNNoiseFilter:
    """
    RNNoise 实时降噪滤波器，每路 WebSocket 连接独立一个实例。

    输入：16kHz PCM16 音频帧（任意长度）
    输出：(降噪后 16kHz PCM16, 平均 VAD 概率 0~1)

    内部流程：
      16kHz PCM16 → 升采样至 48kHz float32 → RNNoise 逐帧处理
      → 降采样回 16kHz → 输出 PCM16 + 平均 VAD 概率
    """

    def __init__(self) -> None:
        try:
            # 直接加载底层 rnnoise 模块，绕过 pyrnnoise/__init__.py 中有兼容性问题的 audiolab 依赖
            import importlib.util, os
            pkg_dir = os.path.dirname(importlib.util.find_spec("pyrnnoise").origin)
            spec = importlib.util.spec_from_file_location("pyrnnoise.rnnoise", os.path.join(pkg_dir, "rnnoise.py"))
            _rnnoise = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_rnnoise)
            self._create = _rnnoise.create
            self._process_mono_frame = _rnnoise.process_mono_frame
            self._destroy = _rnnoise.destroy
            assert _rnnoise.FRAME_SIZE == _RNNOISE_FRAME_SIZE
        except (ImportError, AttributeError, AssertionError) as e:
            raise RuntimeError("pyrnnoise 未安装或版本不兼容，请执行：pip install pyrnnoise") from e

        self._state = self._create()
        # 16kHz 帧余量（int16 numpy）
        self._remainder_16k = np.zeros(0, dtype=np.int16)
        # 48kHz 帧余量（float32 numpy，尚未凑满 480 samples）
        self._remainder_48k = np.zeros(0, dtype=np.float32)

    def __del__(self) -> None:
        if hasattr(self, "_state") and self._state is not None:
            self._destroy(self._state)
            self._state = None

    def process(self, audio_bytes: bytes) -> tuple[bytes, float]:
        """
        输入 16kHz PCM16 音频，返回 (降噪后 16kHz PCM16, 平均 VAD 概率)。
        帧不对齐时暂存余量，下次调用补齐。末尾不足一帧输出时延迟到下次。
        """
        if not audio_bytes:
            return b"", 0.0

        # 1. 转换输入
        new_16k = np.frombuffer(audio_bytes, dtype=np.int16)
        combined_16k = (
            np.concatenate((self._remainder_16k, new_16k))
            if self._remainder_16k.size
            else new_16k
        )

        # 2. 升采样至 48kHz（int16 → float32，range -32768~32767）
        combined_48k_float = resample_poly(combined_16k.astype(np.float32), _UP, _DOWN)
        combined_48k = np.concatenate((self._remainder_48k, combined_48k_float))

        # 3. 逐帧处理（480 samples @ 48kHz = 10ms）
        out_frames_48k: list[np.ndarray] = []
        vad_probs: list[float] = []
        offset = 0
        while offset + _RNNOISE_FRAME_SIZE <= combined_48k.size:
            frame = combined_48k[offset: offset + _RNNOISE_FRAME_SIZE].astype(np.float32)
            # process_mono_frame 期望 int16 范围的 float32
            denoised_int16, vad_prob = self._process_mono_frame(self._state, frame.astype(np.int16))
            out_frames_48k.append(denoised_int16.astype(np.float32))
            vad_probs.append(float(vad_prob))
            offset += _RNNOISE_FRAME_SIZE

        self._remainder_48k = combined_48k[offset:].copy()

        if not out_frames_48k:
            # 暂存 16k 余量，等待下次凑帧
            self._remainder_16k = combined_16k
            return b"", 0.0

        # 4. 降采样回 16kHz
        out_48k = np.concatenate(out_frames_48k)
        out_16k_float = resample_poly(out_48k, _DOWN, _UP)
        out_16k = np.clip(out_16k_float, -32768, 32767).astype(np.int16)

        # 5. 计算本批对应的 16kHz 余量
        # 已消耗的 48k samples = offset，对应 16k samples = offset / 3
        consumed_16k = int(round(offset / _UP))
        self._remainder_16k = combined_16k[consumed_16k:] if consumed_16k < combined_16k.size else np.zeros(0, dtype=np.int16)

        avg_vad = float(np.mean(vad_probs))
        logger.debug(
            "rnnoise process: in=%d bytes out=%d samples vad_prob=%.3f",
            len(audio_bytes), len(out_16k), avg_vad,
        )
        return out_16k.tobytes(), avg_vad

    def reset(self) -> None:
        """通话结束或句子重置时清空内部状态。"""
        if self._state is not None:
            self._destroy(self._state)
        import importlib.util, os
        pkg_dir = os.path.dirname(importlib.util.find_spec("pyrnnoise").origin)
        spec = importlib.util.spec_from_file_location("pyrnnoise.rnnoise", os.path.join(pkg_dir, "rnnoise.py"))
        _rnnoise = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_rnnoise)
        self._state = _rnnoise.create()
        self._remainder_16k = np.zeros(0, dtype=np.int16)
        self._remainder_48k = np.zeros(0, dtype=np.float32)
