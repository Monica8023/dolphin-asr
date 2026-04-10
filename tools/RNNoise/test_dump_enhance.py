import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from asr.denoiser import _load_rnnoise, RNNoiseFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0, help="录制时长（秒）")
    ap.add_argument("--raw", default="raw_mic.wav", help="输出原始 WAV 路径")
    ap.add_argument("--enh", default="enh_mic.wav", help="输出增强 WAV 路径")
    ap.add_argument("--samplerate", type=int, default=16000, help="采样率，默认16k")
    args = ap.parse_args()

    sr = args.samplerate
    print(f"开始录制 {args.seconds}s ... 请说话")
    audio = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    audio = np.squeeze(audio)  # (N,1) -> (N,)

    # 保存原始
    sf.write(args.raw, audio, sr, subtype="PCM_16")

    # 增强
    rnnoise = RNNoiseFilter()
    enh_bytes, _vad_prob = rnnoise.process(audio)
    enh_np = np.frombuffer(enh_bytes, dtype=np.int16)
    sf.write(args.enh, enh_np, sr, subtype="PCM_16")

    print(f"原始 WAV: {args.raw}")
    print(f"增强 WAV: {args.enh}")
    print("用 Audacity 导入两条 WAV，波形/试听对比。")


if __name__ == "__main__":
    main()
