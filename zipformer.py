import sys
import os
import wave
import time
import glob
import queue
import numpy as np
import sounddevice as sd
import sherpa_onnx
from asr.vad import load_vad_model

# ================= 配置区 =================
# ASR 模型路径配置 (请替换为你的实际路径)
MODEL_DIR = "D:/model/zipformer/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
ENCODER = f"{MODEL_DIR}/encoder.int8.onnx"
DECODER = f"{MODEL_DIR}/decoder.onnx"
JOINER = f"{MODEL_DIR}/joiner.int8.onnx"
TOKENS = f"{MODEL_DIR}/tokens.txt"
SAMPLE_RATE = 16000

# VAD 模型路径及参数配置
# 必须使用 sherpa-onnx 适配版本的 silero_vad.onnx
VAD_MODEL = "D:/model/vad/silero_vad.onnx"
VAD_THRESHOLD = 0.3
VAD_MIN_SILENCE_DURATION = 0.15  # 句尾静音多久算一句话结束 (秒)
VAD_MIN_SPEECH_DURATION = 0.25  # 最短有效语音长度，过滤短暂杂音 (秒)
VAD_WINDOW_SIZE = 512  # 16kHz 下的固定窗口点数 (核心限制)


# ==========================================


def create_recognizer():
    """初始化流式 ASR 识别器"""
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=TOKENS,
        encoder=ENCODER,
        decoder=DECODER,
        joiner=JOINER,
        num_threads=1,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        provider="cpu"
    )
    return recognizer


def create_vad():
    """初始化 VAD 端点检测器"""
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = VAD_MODEL
    vad_config.silero_vad.threshold = VAD_THRESHOLD
    vad_config.silero_vad.min_silence_duration = VAD_MIN_SILENCE_DURATION
    vad_config.silero_vad.min_speech_duration = VAD_MIN_SPEECH_DURATION
    vad_config.sample_rate = SAMPLE_RATE
    vad_config.silero_vad.window_size = VAD_WINDOW_SIZE

    # buffer_size_in_seconds 控制内部队列上限，防止内存溢出
    return sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)


def read_wav(filename):
    """读取 wav 文件并转换为模型需要的 float32 numpy 数组"""
    with wave.open(filename, 'rb') as f:
        if f.getframerate() != SAMPLE_RATE:
            print(f"[警告] {filename} 的采样率为 {f.getframerate()}Hz，模型预期为 {SAMPLE_RATE}Hz。")
        if f.getnchannels() != 1:
            print(f"[警告] {filename} 包含多个声道，当前仅读取单声道数据！")

        frames = f.readframes(f.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return audio


def process_single_file(recognizer, vad, filename):
    """使用 VAD 过滤静音并进行离线文件识别"""
    try:
        audio_data = read_wav(filename)
    except Exception as e:
        print(f"[错误] 读取文件 {filename} 失败: {e}")
        return

    start_time = time.time()
    # vad.clear()  # 每次处理新文件前清空 VAD 状态

    offset = 0
    full_text = []

    # 将文件切分成 VAD 要求的固定窗口大小喂入
    while offset + VAD_WINDOW_SIZE <= len(audio_data):
        chunk = audio_data[offset: offset + VAD_WINDOW_SIZE]
        vad.accept_waveform(chunk)

        # 消费 VAD 切分好的干净语音片段
        while not vad.empty():
            segment = vad.front

            # 针对每段干净的语音创建独立的 ASR 识别流
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, segment.samples)
            stream.input_finished()

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream)
            if text:
                full_text.append(text)

            vad.pop()  # 弹出该片段

        offset += VAD_WINDOW_SIZE

    # 强制刷出最后残留的未成句片段
    vad.flush()
    while not vad.empty():
        segment = vad.front
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, segment.samples)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = recognizer.get_result(stream)
        if text:
            full_text.append(text)
        vad.pop()

    end_time = time.time()

    audio_duration = len(audio_data) / SAMPLE_RATE
    process_time = end_time - start_time
    rtf = process_time / audio_duration if audio_duration > 0 else 0
    final_result = "，".join(full_text)

    print(f"文件: {filename}")
    print(f"识别结果: {final_result}")
    print(f"音频时长: {audio_duration:.2f}s | 推理耗时: {process_time:.2f}s | RTF: {rtf:.3f}")
    print("-" * 50)


def run_batch_file(recognizer, vad, input_path):
    """批量或单文件处理逻辑"""
    if os.path.isfile(input_path):
        process_single_file(recognizer, vad, input_path)
    elif os.path.isdir(input_path):
        wav_files = glob.glob(os.path.join(input_path, "*.wav"))
        if not wav_files:
            print(f"[提示] 在目录 {input_path} 中没有找到 .wav 文件。")
            return
        print(f"[INFO] 找到 {len(wav_files)} 个音频文件，开始批量处理...\n" + "=" * 50)
        for wf in wav_files:
            process_single_file(recognizer, vad, wf)
    else:
        print(f"[错误] 路径不存在: {input_path}")


def run_microphone(recognizer, vad):
    """结合 VAD 的麦克风端点检测识别逻辑"""
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].flatten())

    print("\n[INFO] 麦克风已开启，基于 VAD 自动切分句子 (按 Ctrl+C 退出)...\n")
    print("[提示] 您可以连续说话，代码会在您停顿后自动输出整句识别结果。")

    # 用于暂存不足 VAD_WINDOW_SIZE 的碎音频
    audio_buffer = np.array([], dtype=np.float32)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
        try:
            # vad.clear()
            while True:
                samples = audio_queue.get()
                audio_buffer = np.concatenate((audio_buffer, samples))

                # 必须凑齐 VAD 要求的窗口大小再喂数据
                while len(audio_buffer) >= VAD_WINDOW_SIZE:
                    chunk = audio_buffer[:VAD_WINDOW_SIZE]
                    audio_buffer = audio_buffer[VAD_WINDOW_SIZE:]

                    vad.accept_waveform(chunk)

                    # 当 VAD 检测到一段完整的话（讲话+停顿）时，会推入队列
                    while not vad.empty():
                        segment = vad.front

                        # 创建一个新的识别流来处理这句干净的音频
                        stream = recognizer.create_stream()
                        stream.accept_waveform(SAMPLE_RATE, segment.samples)
                        stream.input_finished()

                        # 瞬间解码这整段有效音频
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)

                        text = recognizer.get_result(stream)
                        if text:
                            print(f"[VAD 切分段落]: {text}")

                        # 消费完毕，将片段移出 VAD 队列
                        vad.pop()

        except KeyboardInterrupt:
            print("\n[INFO] 麦克风识别已停止。")


def main():
    print("[INFO] 正在加载 ASR 模型与 VAD 模型，请稍候...")
    recognizer = create_recognizer()
    vad = create_vad()

    if len(sys.argv) < 2:
        print("=" * 50)
        print("用法说明:")
        print("1. 麦克风端点识别: python zipformer.py mic")
        print("2. 离线单文件识别: python zipformer.py file ./test.wav")
        print("3. 离线文件夹批量: python zipformer.py file ./audio_dir")
        print("=" * 50)
        return

    mode = sys.argv[1].lower()

    if mode == "mic":
        run_microphone(recognizer, vad)
    elif mode == "file":
        if len(sys.argv) < 3:
            print("[错误] 请提供音频文件或文件夹路径！示例: python zipformer.py file ./test.wav")
        else:
            run_batch_file(recognizer, vad, sys.argv[2])
    else:
        print(f"[错误] 未知模式: {mode}")


if __name__ == "__main__":
    main()