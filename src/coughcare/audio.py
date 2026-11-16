"""Audio loading, quality checks, cough segmentation and waveform helpers.

COUGHVID clips are Opus audio inside WebM containers. ``libsndfile`` does not
reliably open that combination, so PyAV (which bundles FFmpeg) is the primary
decoder. ``soundfile`` remains a fallback for WAV/FLAC/OGG files.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .config import (
    COUGH_WINDOW_SECONDS,
    MAX_DURATION_SECONDS,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)


class AudioError(ValueError):
    """Raised when a file cannot be decoded as usable audio."""


@dataclass
class AudioQuality:
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    silent_ratio: float
    issues: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return not any(issue.startswith("error:") for issue in self.issues)

    @property
    def warnings(self) -> list[str]:
        return [issue for issue in self.issues if issue.startswith("warning:")]


def _read_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    if hasattr(source, "read"):
        pos = source.tell() if hasattr(source, "tell") else None
        data = source.read()
        if pos is not None and hasattr(source, "seek"):
            try:
                source.seek(pos)
            except (OSError, ValueError):
                pass
        return data if isinstance(data, bytes) else bytes(data)
    raise AudioError("Unsupported audio source; pass a path, bytes, or file-like object.")


def _decode_with_av(data: bytes, sr: int) -> tuple[np.ndarray, int]:
    import av

    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            raise AudioError("This file has no audio track.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().astype(np.float32).reshape(-1))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().astype(np.float32).reshape(-1))
    if not chunks:
        raise AudioError("The audio track is empty.")
    return np.concatenate(chunks), sr


def _decode_with_soundfile(data: bytes, sr: int) -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    y, native_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size == 0:
        raise AudioError("The audio track is empty.")
    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr, res_type="soxr_hq")
    return y.astype(np.float32), sr


def load_audio(
    source: str | Path | bytes | BinaryIO,
    sr: int = SAMPLE_RATE,
    max_seconds: float = MAX_DURATION_SECONDS,
) -> tuple[np.ndarray, int]:
    """Decode any supported audio source to mono float32 at ``sr``.

    Returns ``(samples, sample_rate)``. Raises :class:`AudioError` with a
    human-readable message on failure.
    """
    data = _read_bytes(source)
    if not data:
        raise AudioError("The audio file is empty. Please record or upload again.")

    errors: list[str] = []
    y: np.ndarray | None = None
    try:
        y, sr = _decode_with_av(data, sr)
    except Exception as exc:  # noqa: BLE001 - fall back to libsndfile
        errors.append(f"PyAV: {exc}")
        log.debug("PyAV decode failed: %s", exc)
        try:
            y, sr = _decode_with_soundfile(data, sr)
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"soundfile: {exc2}")
            log.warning("Audio decode failed (%s)", "; ".join(errors))
            raise AudioError(
                "We could not read that audio. Try a WAV, MP3, M4A, OGG or WebM clip."
            ) from exc2

    if y is None or y.size == 0:
        raise AudioError("The audio track is empty. Please record or upload again.")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if max_seconds and y.size > int(max_seconds * sr):
        # Keep the loudest contiguous region rather than silently cutting the tail.
        y = extract_cough_window(y, sr, window_s=min(max_seconds, y.size / sr))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:  # integer-scaled audio from unusual files
        y = y / peak
    if peak < 1e-6:
        raise AudioError("The recording is silent. Check your microphone and try again.")
    return y, sr


def probe_duration(source: str | Path | bytes | BinaryIO) -> float:
    """Read duration, decoding frames only when the container omits it.

    COUGHVID WebM files often report no container/stream duration, so a
    header-only probe is not enough for them. Full decode is still fast for
    these short clips.
    """
    import av

    opened = None
    try:
        if isinstance(source, (str, Path)):
            opened = av.open(str(source))
        else:
            opened = av.open(io.BytesIO(_read_bytes(source)))
        with opened as container:
            if container.duration:
                return float(container.duration / av.time_base)
            if container.streams.audio:
                stream = container.streams.audio[0]
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
                samples = 0
                rate = 0
                for frame in container.decode(stream):
                    samples += int(frame.samples)
                    rate = int(frame.sample_rate or rate or 0)
                if samples and rate:
                    return float(samples / rate)
    except Exception as exc:  # noqa: BLE001 - fall back below
        log.debug("Duration probe failed for %s: %s", source, exc)

    try:
        import soundfile as sf

        if isinstance(source, (str, Path)):
            info = sf.info(str(source))
        else:
            info = sf.info(io.BytesIO(_read_bytes(source)))
        return float(info.frames / info.samplerate)
    except Exception as exc:  # noqa: BLE001
        raise AudioError("Could not determine the audio duration.") from exc


def assess_quality(y: np.ndarray, sr: int) -> AudioQuality:
    """Compute simple quality indicators and user-facing warnings."""
    duration = float(len(y) / sr) if sr else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    peak_dbfs = 20 * np.log10(max(peak, 1e-9))
    rms_dbfs = 20 * np.log10(max(rms, 1e-9))
    clipping_ratio = float(np.mean(np.abs(y) >= 0.999)) if y.size else 0.0

    frame = max(1, int(0.05 * sr))
    usable = (len(y) // frame) * frame
    if usable:
        frames = y[:usable].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        silent_ratio = float(np.mean(frame_rms < 10 ** (-45 / 20)))
    else:
        silent_ratio = 1.0

    issues: list[str] = []
    if duration < 0.3:
        issues.append("error: The clip is shorter than 0.3 seconds.")
    elif duration < 1.0:
        issues.append("warning: Clips under 1 second are too short to analyse reliably.")
    if peak_dbfs < -45:
        issues.append("error: The recording is almost silent. Move closer and try again.")
    elif peak_dbfs < -30:
        issues.append("warning: The recording is quiet. Move your phone a little closer.")
    if clipping_ratio > 0.02:
        issues.append("warning: The audio is clipping. Lower your input volume and re-record.")
    if duration >= 1.0 and silent_ratio > 0.85:
        issues.append("warning: Mostly silence detected. Cough once, close to the microphone.")
    return AudioQuality(
        duration_s=round(duration, 3),
        peak_dbfs=round(float(peak_dbfs), 2),
        rms_dbfs=round(float(rms_dbfs), 2),
        clipping_ratio=round(clipping_ratio, 4),
        silent_ratio=round(silent_ratio, 3),
        issues=issues,
    )


def extract_cough_window(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_s: float = COUGH_WINDOW_SECONDS,
) -> np.ndarray:
    """Return the highest-energy contiguous window of ``window_s`` seconds.

    A short pre-roll is kept so the onset of the cough is not clipped. If the
    clip is shorter than the window it is returned unchanged (and later padded
    by the feature extractor).
    """
    n = len(y)
    win = int(round(window_s * sr))
    if n <= win:
        return y
    frame = max(1, int(0.05 * sr))
    n_frames = n // frame
    if n_frames < 1:
        return y
    energies = np.mean(np.square(y[: n_frames * frame].reshape(n_frames, frame)), axis=1)
    # Smooth over ~150 ms so a single transient does not dominate.
    kernel = np.ones(3, dtype=np.float32) / 3
    energies = np.convolve(energies, kernel, mode="same")
    best = int(np.argmax(energies))
    center = int((best + 0.5) * frame)
    start = int(np.clip(center - win * 0.08, 0, n - win))
    return y[start : start + win]


def waveform_peaks(y: np.ndarray, n_points: int = 320) -> list[float]:
    """Downsample a waveform to normalized peak magnitudes for display."""
    if y.size == 0:
        return [0.0] * n_points
    n_points = max(8, int(n_points))
    edges = np.linspace(0, y.size, n_points + 1, dtype=int)
    peaks = np.array(
        [np.max(np.abs(y[a:b])) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])],
        dtype=np.float32,
    )
    top = float(peaks.max())
    if top > 0:
        peaks = peaks / top
    return [round(float(v), 4) for v in peaks]


def to_wav_bytes(y: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Encode mono float audio as 16-bit WAV bytes for the browser player."""
    import soundfile as sf

    buffer = io.BytesIO()
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.98:
        y = y * (0.98 / peak)
    sf.write(buffer, np.asarray(y, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
