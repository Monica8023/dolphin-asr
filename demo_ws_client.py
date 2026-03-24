"""
/ws/asr 本地验证客户端（文件流 + 合成流 + 麦克风实时流）

示例：
  # 1) 用 wav 文件（自动转 16k/mono/s16le）
  python demo_ws_client.py --host 127.0.0.1 --port 8080 --call-id 1001 --wav-file ./sample.wav

  # 2) 用已准备好的 pcm（16kHz, 16-bit, mono, little-endian）
  python demo_ws_client.py --host 127.0.0.1 --port 8080 --call-id 1001 --pcm-file ./sample.pcm

  # 3) 不带文件，发送 2 秒正弦 + 1 秒静音（用于打断/停顿链路联调）
  python demo_ws_client.py --host 127.0.0.1 --port 8080 --call-id 1001 --synthetic-seconds 2 --synthetic-silence-seconds 1

  # 4) 麦克风实时流
  python demo_ws_client.py --mic --host 127.0.0.1 --port 8080 --call-id 1001 --chunk-ms 60

依赖：
  pip install websockets
  # 使用 --wav-file 时需系统可用 ffmpeg
  # 使用 --mic 时需安装 sounddevice
"""

import argparse
import asyncio
import logging
import math
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le
DEFAULT_CHUNK_MS = 60

logger = logging.getLogger("ws-demo")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_ws_url(host: str, port: int, call_id: int, uuid: int) -> str:
    return f"ws://{host}:{port}/ws/asr?call_id={call_id}&uuid={uuid}"


def frame_bytes(chunk_ms: int, sample_rate: int = SAMPLE_RATE) -> int:
    return sample_rate * chunk_ms // 1000 * BYTES_PER_SAMPLE


def wav_to_pcm_s16le_mono_16k(wav_file: Path, ffmpeg_bin: str) -> Path:
    fd, tmp_path = tempfile.mkstemp(prefix="ws_demo_", suffix=".pcm")
    os.close(fd)
    pcm_path = Path(tmp_path)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(wav_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        str(pcm_path),
    ]
    logger.info("Converting wav -> pcm via ffmpeg: %s", wav_file)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pcm_path


def generate_speech_frame(chunk_ms: int, freq_hz: float = 440.0, amplitude: int = 9000) -> bytes:
    samples = SAMPLE_RATE * chunk_ms // 1000
    vals = [
        int(amplitude * math.sin(2.0 * math.pi * freq_hz * i / SAMPLE_RATE))
        for i in range(samples)
    ]
    return struct.pack(f"{samples}h", *vals)


def generate_silence_frame(chunk_ms: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    return bytes(frame_bytes(chunk_ms, sample_rate=sample_rate))


def list_microphone_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise RuntimeError("sounddevice 未安装，请先执行: pip install sounddevice") from e

    devices = sd.query_devices()
    try:
        default_input = sd.default.device[0]
    except (TypeError, IndexError):
        default_input = sd.default.device if isinstance(sd.default.device, int) else None

    print("\n可用音频设备（输入通道>0）:")
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        mark = " (default)" if idx == default_input else ""
        print(
            f"  [{idx}] {dev.get('name')}"
            f" | in={dev.get('max_input_channels')}"
            f" | sr={dev.get('default_samplerate')}{mark}"
        )


def _normalize_device_arg(device: Optional[str]) -> Optional[int | str]:
    if device is None:
        return None
    text = device.strip()
    if text == "":
        return None
    if text.isdigit():
        return int(text)
    return text


async def stream_pcm_file(
    ws,
    pcm_file: Path,
    chunk_ms: int,
    realtime: bool,
) -> tuple[int, int]:
    sent_frames = 0
    sent_bytes = 0
    chunk_size = frame_bytes(chunk_ms)

    with pcm_file.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            await ws.send(data)
            sent_frames += 1
            sent_bytes += len(data)
            if realtime:
                await asyncio.sleep(chunk_ms / 1000)

    return sent_frames, sent_bytes


async def stream_synthetic(
    ws,
    chunk_ms: int,
    realtime: bool,
    speech_seconds: float,
    silence_seconds: float,
) -> tuple[int, int]:
    sent_frames = 0
    sent_bytes = 0

    speech_frames = int((speech_seconds * 1000) // chunk_ms)
    silence_frames = int((silence_seconds * 1000) // chunk_ms)

    logger.info("Synthetic stream: speech_frames=%d silence_frames=%d", speech_frames, silence_frames)

    for _ in range(speech_frames):
        data = generate_speech_frame(chunk_ms)
        await ws.send(data)
        sent_frames += 1
        sent_bytes += len(data)
        if realtime:
            await asyncio.sleep(chunk_ms / 1000)

    for _ in range(silence_frames):
        data = generate_silence_frame(chunk_ms)
        await ws.send(data)
        sent_frames += 1
        sent_bytes += len(data)
        if realtime:
            await asyncio.sleep(chunk_ms / 1000)

    return sent_frames, sent_bytes


def _resample_mono(mono_f32: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性插值重采样（不引入额外依赖）。mono_f32: float32 一维数组。"""
    if src_rate == dst_rate:
        return mono_f32
    src_len = len(mono_f32)
    dst_len = int(src_len * dst_rate / src_rate)
    if dst_len == 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, src_len - 1, dst_len, dtype=np.float64)
    lo = src_idx.astype(np.int64)
    hi = np.clip(lo + 1, 0, src_len - 1)
    frac = (src_idx - lo).astype(np.float32)
    return mono_f32[lo] * (1.0 - frac) + mono_f32[hi] * frac


async def stream_microphone(
    ws,
    chunk_ms: int,
    device: Optional[str],
    duration_seconds: Optional[float],
    device_sample_rate: int,
    channels: int,
) -> tuple[int, int]:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise RuntimeError("sounddevice 未安装，请先执行: pip install sounddevice") from e

    if channels < 1:
        raise ValueError("--mic-channels 必须 >= 1")

    selected_device = _normalize_device_arg(device)

    # 自动探测设备原生采样率
    if device_sample_rate == SAMPLE_RATE:
        dev_info = sd.query_devices(selected_device, kind="input") if selected_device is not None else sd.query_devices(kind="input")
        native_rate = int(dev_info.get("default_samplerate", SAMPLE_RATE))
        if native_rate != SAMPLE_RATE:
            logger.info(
                "Device native sample_rate=%d != target %d, will resample in callback",
                native_rate, SAMPLE_RATE,
            )
    else:
        native_rate = device_sample_rate

    # 按设备原生采样率计算每次回调读取的帧数（对应 chunk_ms 时长）
    native_samples_per_chunk = native_rate * chunk_ms // 1000
    # 目标帧长（16kHz, chunk_ms）
    target_samples_per_chunk = SAMPLE_RATE * chunk_ms // 1000
    target_frame_bytes = target_samples_per_chunk * BYTES_PER_SAMPLE

    # 重采样后的 PCM 碎片缓冲（处理重采样后帧长不整除的情况）
    pcm_buffer = bytearray()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    loop = asyncio.get_running_loop()
    dropped_frames = 0

    def audio_callback(indata, frames, _time_info, status):
        nonlocal dropped_frames
        if status:
            logger.warning("Microphone status: %s", status)

        # 转 float32，downmix 到单声道
        data_f32 = indata.astype(np.float32, copy=False)
        if data_f32.ndim == 2 and data_f32.shape[1] > 1:
            mono_f32 = data_f32.mean(axis=1)
        else:
            mono_f32 = data_f32.ravel()

        # 能量监控：RMS 转换为 int16 量纲（0~32767），方便对照 VAD 阈值 500
        rms = float(np.sqrt(np.mean(mono_f32 ** 2))) * 32767
        logger.debug("mic rms=%.0f (vad threshold ~500)", rms)

        # 重采样到 16kHz
        resampled = _resample_mono(mono_f32, native_rate, SAMPLE_RATE)

        # 转 int16 little-endian bytes，追加到缓冲
        pcm_chunk = (resampled * 32767.0).clip(-32768, 32767).astype("<i2").tobytes()
        pcm_buffer.extend(pcm_chunk)

        # 按 target_frame_bytes 切帧放入队列
        def _flush_buffer() -> None:
            nonlocal dropped_frames
            while len(pcm_buffer) >= target_frame_bytes:
                frame = bytes(pcm_buffer[:target_frame_bytes])
                del pcm_buffer[:target_frame_bytes]
                if queue.full():
                    dropped_frames += 1
                else:
                    queue.put_nowait(frame)

        loop.call_soon_threadsafe(_flush_buffer)

    stream_kwargs = {
        "samplerate": native_rate,
        "blocksize": native_samples_per_chunk,
        "dtype": "float32",
        "channels": channels,
        "callback": audio_callback,
    }
    if selected_device is not None:
        stream_kwargs["device"] = selected_device

    logger.info(
        "Microphone streaming started: device=%s native_rate=%d target_rate=%d channels=%d chunk_ms=%d",
        selected_device if selected_device is not None else "default",
        native_rate,
        SAMPLE_RATE,
        channels,
        chunk_ms,
    )

    sent_frames = 0
    sent_bytes = 0
    started = time.monotonic()

    with sd.InputStream(**stream_kwargs):
        while True:
            if duration_seconds is not None and (time.monotonic() - started) >= duration_seconds:
                break
            data = await asyncio.wait_for(queue.get(), timeout=1.5)
            await ws.send(data)
            sent_frames += 1
            sent_bytes += len(data)

    logger.info(
        "Microphone streaming ended: frames=%d bytes=%d dropped_frames=%d",
        sent_frames,
        sent_bytes,
        dropped_frames,
    )
    return sent_frames, sent_bytes


def validate_source_args(args: argparse.Namespace) -> None:
    source_count = int(bool(args.mic)) + int(bool(args.wav_file)) + int(bool(args.pcm_file))
    if source_count > 1:
        raise ValueError("--mic / --wav-file / --pcm-file 只能选择一种输入源")


async def run(args: argparse.Namespace) -> None:
    validate_source_args(args)

    if args.list_mic_devices:
        list_microphone_devices()
        return

    ws_url = build_ws_url(args.host, args.port, args.call_id, args.uuid)
    logger.info("Connecting: %s", ws_url)

    pcm_path: Path | None = None
    temp_pcm: Path | None = None

    if args.pcm_file:
        pcm_path = Path(args.pcm_file)
        if not pcm_path.exists():
            raise FileNotFoundError(f"PCM file not found: {pcm_path}")

    if args.wav_file:
        wav_path = Path(args.wav_file)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")
        temp_pcm = wav_to_pcm_s16le_mono_16k(wav_path, args.ffmpeg_bin)
        pcm_path = temp_pcm

    try:
        async with websockets.connect(ws_url, ping_interval=None, max_size=None) as ws:
            logger.info("WebSocket connected")

            if args.mic:
                frames, total_bytes = await stream_microphone(
                    ws=ws,
                    chunk_ms=args.chunk_ms,
                    device=args.mic_device,
                    duration_seconds=args.mic_duration_seconds,
                    device_sample_rate=args.mic_sample_rate,
                    channels=args.mic_channels,
                )
            elif pcm_path:
                logger.info("Streaming PCM file: %s", pcm_path)
                frames, total_bytes = await stream_pcm_file(ws, pcm_path, args.chunk_ms, args.realtime)
            else:
                frames, total_bytes = await stream_synthetic(
                    ws=ws,
                    chunk_ms=args.chunk_ms,
                    realtime=args.realtime,
                    speech_seconds=args.synthetic_seconds,
                    silence_seconds=args.synthetic_silence_seconds,
                )

            # 默认行为：文件流不补尾静音；mic/合成流补 1000ms（可用 --tail-silence-ms 覆盖）
            effective_tail_silence_ms = args.tail_silence_ms
            if effective_tail_silence_ms is None:
                effective_tail_silence_ms = 0 if pcm_path else 1000

            if effective_tail_silence_ms > 0:
                tail_frames = effective_tail_silence_ms // args.chunk_ms
                logger.info("Sending tail silence: %d ms (%d frames)", effective_tail_silence_ms, tail_frames)
                for _ in range(tail_frames):
                    data = generate_silence_frame(args.chunk_ms, sample_rate=args.mic_sample_rate if args.mic else SAMPLE_RATE)
                    await ws.send(data)
                    frames += 1
                    total_bytes += len(data)
                    if args.realtime:
                        await asyncio.sleep(args.chunk_ms / 1000)

            # 默认行为：文件流不额外等待；mic/合成流等待 1000ms（可用 --wait-after-send-ms 覆盖）
            effective_wait_after_send_ms = args.wait_after_send_ms
            if effective_wait_after_send_ms is None:
                effective_wait_after_send_ms = 0 if pcm_path else 1000

            if effective_wait_after_send_ms > 0:
                logger.info("Waiting %d ms for server side flush/intent", effective_wait_after_send_ms)
                await asyncio.sleep(effective_wait_after_send_ms / 1000)

            stream_ms = frames * args.chunk_ms
            logger.info(
                "Done. frames=%d bytes=%d approx_audio_ms=%d",
                frames,
                total_bytes,
                stream_ms,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if temp_pcm and temp_pcm.exists():
            temp_pcm.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="/ws/asr 本地语音流验证客户端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--call-id", type=int, default=1001)
    parser.add_argument("--uuid", type=int, default=0)

    parser.add_argument("--pcm-file", help="16kHz/16-bit/mono/little-endian PCM 文件路径")
    parser.add_argument("--wav-file", help="WAV 文件路径（将自动用 ffmpeg 转 PCM）")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg 可执行文件路径")

    parser.add_argument("--mic", action="store_true", help="启用麦克风实时流")
    parser.add_argument("--mic-device", help="麦克风设备 ID 或名称", default= "1")
    parser.add_argument("--mic-duration-seconds", type=float, help="麦克风采集时长（秒），不填表示直到 Ctrl+C")
    parser.add_argument("--mic-sample-rate", type=int, default=16000, help="麦克风采样率（默认 16000）")
    parser.add_argument("--mic-channels", type=int, default=1, help="麦克风输入通道数（默认 1）")
    parser.add_argument("--list-mic-devices", action="store_true", help="列出可用麦克风设备后退出")

    parser.add_argument("--chunk-ms", type=int, default=DEFAULT_CHUNK_MS)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")

    parser.add_argument("--synthetic-seconds", type=float, default=2.0)
    parser.add_argument("--synthetic-silence-seconds", type=float, default=1.0)

    parser.add_argument("--tail-silence-ms", type=int, default=None, help="尾静音时长；默认文件流=0ms，mic/合成流=1000ms")
    parser.add_argument("--wait-after-send-ms", type=int, default=None, help="发送后额外等待；默认文件流=0ms，mic/合成流=1000ms")

    parser.add_argument("--log-level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
