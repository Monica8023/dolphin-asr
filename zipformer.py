import sys
import queue
import numpy as np
import sounddevice as sd
import sherpa_onnx

# ================= 配置区 =================
# 请替换为你解压后的实际模型路径
MODEL_DIR = "./model_dir/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"

# 模型文件路径配置
# 如果你最终下载的是 fp16/标准版，文件名可能没有 .int8，请按实际情况修改
ENCODER = f"{MODEL_DIR}/encoder.int8.onnx"
DECODER = f"{MODEL_DIR}/decoder.onnx"
JOINER = f"{MODEL_DIR}/joiner.int8.onnx"
TOKENS = f"{MODEL_DIR}/tokens.txt"

SAMPLE_RATE = 16000  # 模型要求的标准采样率


# ==========================================

def create_recognizer():
    """初始化流式识别器"""
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=TOKENS,
        encoder=ENCODER,
        decoder=DECODER,
        joiner=JOINER,
        num_threads=1,  # CPU 推理线程数，int8 下通常 1-2 即可跑满流式
        sample_rate=SAMPLE_RATE,
        feature_dim=80,  # fbank 特征维度，zipformer 默认 80
        # enable_endpoint=False,  # 开启自带的 VAD 截断断句
        rule1_min_trailing_silence=1.2,  # 停顿多久算一句话结束 (秒)
        provider="cpu"  # 若用 fp16 且有显卡，可改为 "cuda"
    )
    return recognizer


def main():
    recognizer = create_recognizer()
    stream = recognizer.create_stream()

    # 使用队列来在音频回调线程和主线程之间安全地传递数据
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        """sounddevice 的音频回调函数，将采集到的音频放入队列"""
        if status:
            print(status, file=sys.stderr)
        # 将音频数据展平为一维 float32 数组
        audio_queue.put(indata[:, 0].flatten())

    print("\n[INFO] 模型加载完毕。")
    print("[INFO] 请开始说话 (按 Ctrl+C 退出)...\n")

    # 打开麦克风流
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
        try:
            last_text = ""
            while True:
                # 获取音频片段并喂给模型
                samples = audio_queue.get()
                stream.accept_waveform(SAMPLE_RATE, samples)

                # 循环解码当前所有的可用 Chunk
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                # 获取实时识别结果
                is_endpoint = recognizer.is_endpoint(stream)
                text = recognizer.get_result(stream)

                # 终端输出效果优化 (流式覆盖当前行)
                if text and text != last_text:
                    # 打印流式中间结果
                    print(f"\r实时: {text}", end="", flush=True)
                    last_text = text

                # 如果模型判定当前句子结束 (触发 VAD)
                if is_endpoint:
                    if text:
                        print(f"\r[最终结果]: {text}")
                        # 在这里可以触发你的意图识别逻辑
                        # send_to_intent_system(text)

                    # 重置状态，准备听下一句话
                    recognizer.reset(stream)
                    last_text = ""

        except KeyboardInterrupt:
            print("\n[INFO] 停止识别。")


if __name__ == "__main__":
    main()