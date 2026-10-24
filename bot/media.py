from __future__ import annotations

import asyncio
import io
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path


async def run_media_command(*args: str, data: bytes | None = None) -> bytes:
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **options,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(data), timeout=40)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError("Медиа не декодируется: " + stderr.decode("utf-8", errors="replace")[-300:])
    return stdout


async def sample_video(data: bytes) -> list[bytes]:
    # Seekable input is required for MP4 whose moov atom is at the end.
    with tempfile.TemporaryDirectory(prefix="yss-frames-") as directory:
        root = Path(directory)
        source = root / "input.media"
        source.write_bytes(data)
        try:
            await run_media_command(
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-t", "12", "-vf", "fps=1/3,scale=512:-2",
                "-frames:v", "4", str(root / "frame-%02d.jpg"),
            )
        except RuntimeError:
            # Sub-second clips can leave the sparse sampler with no output packets.
            if any(root.glob("frame-*.jpg")):
                raise
        frames = [path.read_bytes() for path in sorted(root.glob("frame-*.jpg"))]
        # Very short stickers may not produce a frame at 1/3 fps.
        if not frames:
            await run_media_command(
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-frames:v", "1", "-vf", "scale=512:-2",
                str(root / "first.jpg"),
            )
            frames = [(root / "first.jpg").read_bytes()]
        return frames


def wrap_pcm(data: bytes, mime_type: str) -> bytes:
    if data.startswith(b"RIFF") or "l16" not in mime_type.lower() and "pcm" not in mime_type.lower():
        return data
    match = re.search(r"rate=(\d+)", mime_type)
    rate = int(match.group(1)) if match else 24000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(data)
    return buffer.getvalue()


class SpeechFallback:
    def __init__(self) -> None:
        self._model = None
        self._lock = asyncio.Lock()

    async def transcribe(self, data: bytes) -> str:
        pcm = await run_media_command(
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-t", "90", "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "pipe:1", data=data,
        )
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, pcm)

    def _transcribe_sync(self, pcm: bytes) -> str:
        import numpy as np
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(
                os.getenv("LOCAL_STT_MODEL", "base"), device="cpu", compute_type="int8",
                cpu_threads=2, download_root="data/models",
            )
        # FFmpeg already decoded mono 16 kHz; pass samples to avoid decoding twice.
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(samples, beam_size=3, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            return "[Разборчивой речи не обнаружено; по расшифровке нельзя определить звуки или эмоции.]"
        return "[Локальная расшифровка первых 90 секунд; возможны ошибки имён.]\n" + text

    async def speak(self, text: str) -> bytes:
        import edge_tts

        async def collect() -> bytes:
            speech = edge_tts.Communicate(text, os.getenv("FALLBACK_TTS_VOICE", "ru-RU-DmitryNeural"))
            chunks = [chunk["data"] async for chunk in speech.stream() if chunk["type"] == "audio"]
            if not chunks:
                raise RuntimeError("Резервная озвучка не вернула аудио")
            return b"".join(chunks)

        return await asyncio.wait_for(collect(), timeout=35)
