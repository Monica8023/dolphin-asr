"""FRCRN 语音增强降噪模块。

使用 FunASR iic/speech_frcrn_ans_cirm_16k 模型，
对 16kHz PCM16 音频做端到端语音增强（ANS）。

高并发优化说明：
- 采用 FunASR 原生 AutoModel 引擎
- 纯内存 Numpy 矩阵运算，零 IO 开销，无临时文件
- 完全规避 ModelScope Pipeline 的类型校验壁垒
"""
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

    # 改为 FRCRN 模型路径
    model_path = cfg.get("enhancer_model_path", "iic/speech_frcrn_ans_cirm_16k")
    device = cfg.get("asr_device", "cpu")

    logger.info("Loading FRCRN Enhancer model from %s on device=%s", model_path, device)

    # 恢复使用 FunASR 的 AutoModel 加载
    _enhancer_model = pipeline(
        Tasks.acoustic_noise_suppression,
        model=model_path,
        device=device,
    )
    logger.info("FRCRN Enhancer model loaded.")

def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """
    创建WAV文件头的字节串。

    :param dataflow: 音频bytes数据（以字节为单位）。
    :param sample_rate: 采样率，默认16000。
    :param num_channels: 声道数，默认1（单声道）。
    :param bits_per_sample: 每个样本的位数，默认16。
    :return: WAV文件头的字节串和音频bytes数据。
    """
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    # 使用 bytearray 构建字节串
    header = bytearray()

    # RIFF/WAVE header
    header.extend(b'RIFF')
    header.extend(riff_chunk_size.to_bytes(4, byteorder='little'))
    header.extend(b'WAVE')

    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(fmt_chunk_size.to_bytes(4, byteorder='little'))
    header.extend((1).to_bytes(2, byteorder='little'))  # Audio format (1 is PCM)
    header.extend(num_channels.to_bytes(2, byteorder='little'))
    header.extend(sample_rate.to_bytes(4, byteorder='little'))
    header.extend(byte_rate.to_bytes(4, byteorder='little'))
    header.extend(block_align.to_bytes(2, byteorder='little'))
    header.extend(bits_per_sample.to_bytes(2, byteorder='little'))

    # data subchunk
    header.extend(b'data')
    header.extend(data_chunk_size.to_bytes(4, byteorder='little'))

    return bytes(header) + dataflow

class SpeechEnhancer:
    """无状态语音增强引擎，可直接复用全局模型实例。

    输入：16kHz PCM16 音频（任意长度，建议分块）
    输出：增强后的 16kHz PCM16 音频
    """

    def enhance(self, audio_bytes: bytes) -> bytes:
        """
        对整段 PCM16 音频做语音增强降噪。

        audio_bytes: 16kHz, 16-bit PCM, mono, little-endian
        返回：增强后相同格式的 PCM16 bytes；模型未加载或失败时返回原始音频。
        """
        if _enhancer_model is None or not audio_bytes:
            return audio_bytes

        # 1. 极速内存解析：将二进制直接映射为 float32 的 numpy 数组
        # FunASR 内部的声学特征提取器要求输入的是归一化到 [-1.0, 1.0] 的 float32 数据
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            # 2. 纯内存推理：FunASR 完美接受 ndarray 作为 input
            result = _enhancer_model(create_wav_header(audio_bytes, sample_rate=16000, num_channels=1, bits_per_sample=16))
            return result['output_pcm']

        except Exception as e:
            logger.warning("FRCRN inference failed, returning original audio: %s", e)

        # 任何环节失败，原样退回声音，保证通话链路绝对不断
        return audio_bytes