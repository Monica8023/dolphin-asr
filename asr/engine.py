"""ASR engine — sherpa-onnx streaming Zipformer (Transducer) integration."""
import logging

import numpy as np
import sherpa_onnx
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

_recognizer: sherpa_onnx.OnlineRecognizer | None = None


def load_model() -> None:
    """启动时加载一次全局 OnlineRecognizer，所有连接共享（只读，线程安全）。"""
    global _recognizer
    from config import nacos_config as cfg

    model_dir = cfg.get("asr_model_path")
    #fp16
    # encoder = cfg.get("asr_encoder", f"{model_dir}/encoder.fp16.onnx")
    # decoder = cfg.get("asr_decoder", f"{model_dir}/decoder.fp16.onnx")
    # joiner = cfg.get("asr_joiner", f"{model_dir}/joiner.fp16.onnx")

    #int8
    encoder = cfg.get("asr_encoder", f"{model_dir}/encoder.int8.onnx")
    decoder = cfg.get("asr_decoder", f"{model_dir}/decoder.onnx")
    joiner = cfg.get("asr_joiner", f"{model_dir}/joiner.int8.onnx")
    BPE_VOCAB = f"{model_dir}/bpe.vocab"
    tokens = cfg.get("asr_tokens", f"{model_dir}/tokens.txt")
    num_threads = cfg.get("asr_num_threads", 2)
    feature_dim = cfg.get("asr_feature_dim", 80)
    provider = cfg.get("asr_provider", "cuda")

    logger.info(
        "Loading Zipformer model: encoder=%s decoder=%s joiner=%s tokens=%s "
        "num_threads=%d feature_dim=%d provider=%s",
        encoder, decoder, joiner, tokens, num_threads, feature_dim, provider,
    )
    _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=feature_dim,
        provider=provider,

        # decoding_method="modified_beam_search",
        # hotwords_file="/home/zhulieai/xwl/ASR/asr/hotword.txt",
        # hotwords_score=1.5,

        # modeling_unit="cjkchar+bpe",
        # bpe_vocab=BPE_VOCAB
    )
    logger.info("Zipformer model loaded.")


class ASREngine:
    """每路 WebSocket 连接独立一个实例，持有 OnlineStream 流式解码状态。"""

    def __init__(self) -> None:
        self._stream: sherpa_onnx.OnlineStream | None = None
        self._last_text: str = ""
        if _recognizer is not None:
            self._stream = _recognizer.create_stream()

    def transcribe(self, audio_bytes: bytes, is_final: bool = False) -> str:
        """
        喂入一帧 PCM16 音频，返回本次新增的识别文本（可能为空）。

        audio_bytes: 16kHz, 16-bit PCM, mono, little-endian
        is_final: 连接断开时传 True，取出剩余缓冲文本后重置流
        """
        if _recognizer is None or self._stream is None:
            return ""
        if not audio_bytes and not is_final:
            return ""

        if audio_bytes:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            self._stream.accept_waveform(16000, samples)

        while _recognizer.is_ready(self._stream):
            _recognizer.decode_stream(self._stream)

        current_text: str = _recognizer.get_result(self._stream)

        # get_result 返回自上次 reset 以来的累计文本，取增量部分返回
        new_text = current_text[len(self._last_text):]
        self._last_text = current_text

        if is_final:
            self._last_text = ""
            _recognizer.reset(self._stream)

        return new_text

    def reset(self) -> None:
        """每句话结束后调用，重置流式解码状态准备识别下一句。"""
        if _recognizer is not None and self._stream is not None:
            _recognizer.reset(self._stream)
        self._last_text = ""
