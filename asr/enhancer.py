"""FRCRN 语音增强降噪模块。

使用 FunASR iic/speech_frcrn_ans_cirm_16k 模型，
对 16kHz PCM16 音频做端到端语音增强（ANS）。

入口统一上采样后，本模块不再做任何采样率转换，直接处理 16kHz。
"""
import contextlib
import io
import logging
import numpy as np
from config import nacos_config as cfg

logger = logging.getLogger(__name__)

_enhancer_model = None

def load_enhancer_model() -> None:
    """启动时加载一次全局 FRCRN 模型，所有连接共享同一模型实例。"""
    global _enhancer_model
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    model_path = cfg.get("enhancer_model_path", "iic/speech_frcrn_ans_cirm_16k")
    device = cfg.get("asr_device", "cpu")

    logger.info("Loading FRCRN Enhancer model from %s on device=%s", model_path, device)
    _enhancer_model = pipeline(
        Tasks.acoustic_noise_suppression,
        model=model_path,
        device=device,
    )
    logger.info("FRCRN Enhancer model loaded.")

def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """创建 WAV 文件头。"""
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    header = bytearray()
    header.extend(b'RIFF')
    header.extend(riff_chunk_size.to_bytes(4, byteorder='little'))
    header.extend(b'WAVE')
    header.extend(b'fmt ')
    header.extend(fmt_chunk_size.to_bytes(4, byteorder='little'))
    header.extend((1).to_bytes(2, byteorder='little'))
    header.extend(num_channels.to_bytes(2, byteorder='little'))
    header.extend(sample_rate.to_bytes(4, byteorder='little'))
    header.extend(byte_rate.to_bytes(4, byteorder='little'))
    header.extend(block_align.to_bytes(2, byteorder='little'))
    header.extend(bits_per_sample.to_bytes(2, byteorder='little'))
    header.extend(b'data')
    header.extend(data_chunk_size.to_bytes(4, byteorder='little'))

    return bytes(header) + dataflow


class SpeechEnhancer:
    """带缓冲的语音增强引擎，复用全局 FRCRN 模型实例。

    输入/输出均为 16kHz PCM16（由 stream_handler 入口统一上采样后传入）。
    内部不做任何采样率转换，直接送入 FRCRN 推理。

    缓冲阈值：_BUFFER_THRESHOLD_SAMPLES samples @ 16kHz
    默认 3200 samples = 200ms，在延迟与推理效率之间取平衡。
    """

    # 3200 samples @ 16kHz = 200ms
    _BUFFER_THRESHOLD_SAMPLES = int(cfg.get("enhancer_buffer_samples", 3200))

    def __init__(self) -> None:
        self._buffer = bytearray()

    def reset(self) -> None:
        """句子结束 / 通话重置时清空缓冲，防止跨句串扰。"""
        self._buffer.clear()

    def enhance(self, audio_bytes: bytes) -> bytes:
        """
        喂入一帧 16kHz PCM16 音频，返回增强后的 16kHz PCM16 音频。

        缓冲未满时返回 b""；达到阈值后批量推理并返回结果。
        模型未加载或推理失败时原样返回，保证通话链路绝不中断。
        """
        if _enhancer_model is None or not audio_bytes:
            return audio_bytes

        self._buffer.extend(audio_bytes)

        if len(self._buffer) // 2 < self._BUFFER_THRESHOLD_SAMPLES:
            return b""

        batch_bytes = bytes(self._buffer)
        self._buffer.clear()

        try:
            # 直接构建 16kHz WAV Header 送入 FRCRN（无需采样率转换）
            _devnull = io.StringIO()
            with contextlib.redirect_stdout(_devnull):
                result = _enhancer_model(
                    create_wav_header(batch_bytes, sample_rate=16000)
                )
            enhanced = result['output_pcm']

            if isinstance(enhanced, np.ndarray):
                enhanced_np = enhanced.flatten().astype(np.float32)
            else:
                enhanced_np = np.frombuffer(enhanced, dtype=np.int16).astype(np.float32)

            return np.clip(enhanced_np, -32768, 32767).astype(np.int16).tobytes()

        except Exception as e:
            logger.warning("FRCRN inference failed, returning original audio: %s", e)

        return batch_bytes
