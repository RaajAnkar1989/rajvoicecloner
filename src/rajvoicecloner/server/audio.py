"""Audio output-format handling for the API.

Supports ElevenLabs-style ``output_format`` identifiers:
``mp3_44100_128``, ``pcm_16000``, ``pcm_24000``, ``pcm_44100``, ``pcm_48000``,
``wav_44100``, ``wav_48000`` ... Unknown formats raise ``ValueError``.
"""

import io
import functools
import struct
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr

DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


@dataclass(frozen=True)
class OutputFormat:
    codec: str  # "mp3" | "pcm" | "wav"
    sample_rate: int
    bitrate_kbps: int | None = None

    @property
    def media_type(self) -> str:
        if self.codec == "mp3":
            return "audio/mpeg"
        if self.codec == "wav":
            return "audio/wav"
        return "application/octet-stream"  # raw pcm

    @property
    def file_extension(self) -> str:
        return {"mp3": "mp3", "wav": "wav", "pcm": "pcm"}[self.codec]


def parse_output_format(value: str | None) -> OutputFormat:
    raw = (value or DEFAULT_OUTPUT_FORMAT).strip().lower()
    parts = raw.split("_")
    codec = parts[0]
    if codec not in ("mp3", "pcm", "wav"):
        raise ValueError(f"Unsupported output_format: {value}")
    try:
        sample_rate = int(parts[1]) if len(parts) > 1 else 44100
        bitrate = int(parts[2]) if len(parts) > 2 else None
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Malformed output_format: {value}") from exc
    return OutputFormat(codec=codec, sample_rate=sample_rate, bitrate_kbps=bitrate)


def resample(wav: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return wav
    # soxr is already pulled in by librosa and is noticeably faster for the
    # repeated mono resampling done during streaming.
    return soxr.resample(wav.astype(np.float32), src_rate, dst_rate)


def float_to_pcm16(wav: np.ndarray) -> bytes:
    clipped = np.clip(wav, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


@functools.lru_cache(maxsize=1)
def _mp3_supported() -> bool:
    return "MP3" in sf.available_formats()


def encode(wav: np.ndarray, src_rate: int, fmt: OutputFormat) -> tuple[bytes, OutputFormat]:
    """Encode a float waveform into the requested format.

    Returns the encoded bytes plus the format actually used (mp3 silently
    falls back to wav when libsndfile lacks an mp3 encoder).
    """
    wav = resample(wav, src_rate, fmt.sample_rate)
    if fmt.codec == "pcm":
        return float_to_pcm16(wav), fmt

    codec = fmt.codec
    if codec == "mp3" and not _mp3_supported():
        codec = "wav"
        fmt = OutputFormat(codec="wav", sample_rate=fmt.sample_rate)

    buf = io.BytesIO()
    if codec == "mp3":
        sf.write(buf, wav, fmt.sample_rate, format="MP3")
    else:
        sf.write(buf, wav, fmt.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), fmt


def wav_stream_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """RIFF/WAVE header with an unknown (max) data size, for chunked streaming."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0xFFFFFFFF - 36
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 0xFFFFFFFF),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample),
            b"data",
            struct.pack("<I", data_size),
        ]
    )
