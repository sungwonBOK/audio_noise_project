"""Waveform operations shared by training, evaluation and file inference."""
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path, sr, seconds, start=0.0, channel=None):
    """Read only the requested interval; never normalize each inference window."""
    with sf.SoundFile(str(path)) as f:
        f.seek(min(round(start * f.samplerate), len(f)))
        x = f.read(round(seconds * f.samplerate), dtype="float32", always_2d=True)
        old_sr = f.samplerate
    if not len(x):
        return np.zeros(round(seconds * sr), dtype=np.float32)
    x = x.mean(axis=1) if channel is None else x[:, channel]
    if old_sr != sr:
        g = math.gcd(old_sr, sr)
        x = resample_poly(x, sr // g, old_sr // g).astype(np.float32)
    n = round(seconds * sr)
    return np.pad(x[:n], (0, max(0, n - len(x)))).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def limit_peak(x):
    return (x / max(1.0, float(np.max(np.abs(x))) / 0.95)).astype(np.float32)


def mix_at_snr(target, noise, snr_db):
    """SNR is defined by RMS of the full padded analysis window, not active frames."""
    rt, rn = rms(target), rms(noise)
    if min(rt, rn) < 1e-10:
        raise ValueError("Cannot mix a silent target/noise at a defined SNR")
    scaled_noise = noise * (rt / (rn * 10 ** (snr_db / 20)))
    return limit_peak(target + scaled_noise)


def max_frame_dbfs(x, sr, frame_seconds=0.05, hop_seconds=0.025):
    """A short loud event must not disappear into a whole-window RMS average."""
    frame, hop = max(1, round(frame_seconds * sr)), max(1, round(hop_seconds * sr))
    peak = max((rms(x[i:i + frame]) for i in range(0, len(x), hop)), default=0.0)
    return 20 * math.log10(max(peak, 1e-12))


def window_intervals(duration, window, hop):
    if not 0 < hop <= window:
        raise ValueError("Require 0 < hop_seconds <= window_seconds")
    if duration <= 0:
        return
    # Stop once a window reaches EOF; avoid multiple heavily padded trailing windows.
    i = 0
    while True:
        start = i * hop
        end = min(start + window, duration)
        yield start, end
        if end >= duration - 1e-9:
            break
        i += 1


def noise_at_level(noise, level, snr_db):
    """Scale noise so that RMS(noise) = level / 10^(snr/20)."""
    rn = rms(noise)
    if rn < 1e-10:
        raise ValueError("Cannot mix a silent noise at a defined SNR")
    return noise * (level / (rn * 10 ** (snr_db / 20)))


def render_long_window(row, cfg, mode="mixed"):
    """One window of a long clean recording (rows made by prepare.py --long-labels).

    The whole file shares one gain (its loudest labeled window -> target_rms), and the noise
    level is set against that file-level target level, not against this window: background
    noise stays constant while the target event comes and goes, and event-free windows work.
    """
    sr, seconds = cfg["sample_rate"], cfg["window_seconds"]
    x = load_audio(row["target_path"], sr, seconds, row["target_start"]) * (row["target_rms"] / row["ref_rms"])
    if row["kind"] == "mixed" and mode == "mixed":
        x = x + noise_at_level(load_audio(row["noise_path"], sr, seconds), row["target_rms"], row["snr_db"])
    return limit_peak(x)


def render_recipe(row, cfg, mode="mixed"):
    sr, seconds = cfg["sample_rate"], cfg["window_seconds"]
    n = round(sr * seconds)
    if row.get("source") == "long":
        return render_long_window(row, cfg, mode)
    if row["kind"] == "silence":
        return np.zeros(n, dtype=np.float32)
    x = load_audio(row["target_path"] or row["noise_path"], sr, seconds)
    if row["target_path"]:
        if rms(x) < 1e-10:
            raise ValueError(f"Silent target: {row['target_path']}")
        x = x * (row["target_rms"] / rms(x))
        if row["kind"] == "mixed" and mode == "mixed":
            noise = load_audio(row["noise_path"], sr, seconds)
            return mix_at_snr(x, noise, row["snr_db"])
    return limit_peak(x)


def resolve_path(value):
    return str(Path(value).expanduser().resolve())
