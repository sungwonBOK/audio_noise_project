"""

학습 데이터를 매번 새로 섞는 도우미 모듈 (SNR, 시간 위치 등). 검증 및 테스트는 이전과 동일

TRAIN 분할만 cfg["augment"]["enabled"]가 true일 때 이 모듈을 통과합니다.


아래는 클로드 설명

Training-only, on-the-fly waveform augmentation.

Validation and test keep using audio_utils.render_recipe (fixed recipes), so
scores stay comparable with earlier runs. Only the TRAIN split goes through
this module when cfg["augment"]["enabled"] is true.

Each training presentation is rendered from (seed, epoch, slot index):
- every epoch draws new noise, SNR, gain and time positions (fresh mixing),
- the same epoch is exactly reproducible,
- a config without an "augment" section behaves exactly as before.

What is randomized per presentation
1. Target onset: the digital zero padding of the clip is trimmed and the
   remaining content is placed at a random offset. Part of it may fall outside
   the window, but at least `min_target_energy_keep` of its energy stays
   inside, so the positive label remains valid.
2. Continuous SNR: uniform in `snr_range_db` instead of the fixed 0/5/10 list.
3. Fresh noise: a new non-target file from the TRAIN noise pool every time.
4. Partial time overlap: with probability `partial_overlap_prob` the noise
   covers only a random part of the window. Otherwise the noise is looped to
   cover the whole window, as before.
5. Overlapping crops of long sources: a file longer than the window is split
   into overlapping crop slots (`long_crop_hop_seconds` apart). Every slot is
   one training item per epoch, with a random start jittered inside its slot.
   For a long TARGET file without annotations, a crop is accepted only if its
   loudest frame is within `long_target_peak_db` of the file's loudest frame
   (assumption: the target is the loudest content of its own recording).
6. Windows of long clean recordings (rows with source == "long", made by
   pseudo_label.py + prepare.py --long-labels) already carry WINDOW labels, so they
   are never re-cropped or time-shifted (the label belongs to that exact window).
   They still get a fresh gain, noise, SNR and partial overlap every epoch; the noise
   level follows audio_utils.long_noise_reference (the window's own RMS if a target is
   labeled 1, else the file's target level).
"""
import math
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from audio_utils import load_audio, rms, limit_peak, long_noise_reference, max_frame_dbfs

DEFAULTS = {
    "enabled": False,
    "snr_range_db": [-5.0, 15.0],
    "target_rms_range": None,          # None -> cfg["foreground_rms_range"]
    "negative_gain_db": [-6.0, 6.0],
    "time_shift": True,
    "min_target_energy_keep": 0.7,
    "trim_abs_threshold": 1e-5,
    "shift_tries": 8,
    "partial_overlap_prob": 0.3,
    "partial_overlap_min_fraction": 0.3,
    "long_crop_hop_seconds": 1.0,
    "max_crops_per_source": 30,
    "long_target_peak_db": 20.0,
    "crop_tries": 8,
    "spec_augment": {"enabled": False, "time_drop_width": 32, "time_stripes_num": 2,
                     "freq_drop_width": 8, "freq_stripes_num": 2},
}


def augment_settings(cfg):
    user = dict(cfg.get("augment") or {})
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown augment settings: {sorted(unknown)}")
    spec_user = dict(user.pop("spec_augment", None) or {})
    unknown = set(spec_user) - set(DEFAULTS["spec_augment"])
    if unknown:
        raise ValueError(f"Unknown spec_augment settings: {sorted(unknown)}")
    s = {**DEFAULTS, **user, "spec_augment": {**DEFAULTS["spec_augment"], **spec_user}}
    if s["target_rms_range"] is None:
        s["target_rms_range"] = list(cfg["foreground_rms_range"])
    for key in ("snr_range_db", "target_rms_range", "negative_gain_db"):
        lo, hi = s[key]
        if lo > hi:
            raise ValueError(f"augment.{key} must be [low, high]")
    if s["target_rms_range"][0] <= 0:
        raise ValueError("augment.target_rms_range must be positive")
    if not 0 < s["min_target_energy_keep"] <= 1:
        raise ValueError("augment.min_target_energy_keep must be in (0, 1]")
    if not 0 <= s["partial_overlap_prob"] <= 1 or not 0 < s["partial_overlap_min_fraction"] <= 1:
        raise ValueError("augment partial overlap settings must be probabilities/fractions")
    hop = s["long_crop_hop_seconds"]
    if hop is not None and hop <= 0:
        raise ValueError("augment.long_crop_hop_seconds must be positive or null")
    if s["max_crops_per_source"] < 1 or s["shift_tries"] < 1 or s["crop_tries"] < 1:
        raise ValueError("augment counts must be >= 1")
    sa = s["spec_augment"]
    if sa["enabled"] and min(sa["time_drop_width"], sa["freq_drop_width"]) < 1:
        raise ValueError("spec_augment drop widths must be >= 1")
    return s


@lru_cache(maxsize=None)
def duration_seconds(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate


@lru_cache(maxsize=None)
def source_profile(path, sr, frame_seconds=0.05):
    """Loudest frame level (dBFS) and its time in a whole source file."""
    dur = duration_seconds(path)
    x = load_audio(path, sr, dur)
    frame = max(1, round(frame_seconds * sr))
    levels = [rms(x[i:i + frame]) for i in range(0, max(1, len(x)), frame)]
    k = int(np.argmax(levels))
    return 20 * math.log10(max(levels[k], 1e-12)), (k + 0.5) * frame / sr


def crop_slots(path, window, s):
    """Start-time ranges [lo, hi] of overlapping crops covering a source."""
    if not path:
        return [(0.0, 0.0)]
    max_start = duration_seconds(path) - window
    if max_start <= 1e-6:
        return [(0.0, 0.0)]
    hop = s["long_crop_hop_seconds"]
    k = 1 if hop is None else int(math.floor(max_start / hop + 1e-9)) + 1
    k = min(k, s["max_crops_per_source"])
    width = max_start / k
    return [(i * width, (i + 1) * width) for i in range(k)]


def trim_padding(x, threshold):
    idx = np.flatnonzero(np.abs(x) > threshold)
    if not len(idx):
        return x[:0]
    return x[idx[0]:idx[-1] + 1]


def place(content, offset, n):
    """Put content at offset in an n-sample window (parts outside are cut)."""
    out = np.zeros(n, dtype=np.float32)
    a, b = max(0, offset), min(n, offset + len(content))
    if a < b:
        out[a:b] = content[a - offset:b - offset]
    return out


def shift_target(x, rng, s, n):
    """Random onset; returns (window, onset_sample). Keeps enough target energy inside."""
    c = trim_padding(x, s["trim_abs_threshold"])
    if not len(c):
        raise ValueError("Silent target after trimming padding")
    if len(c) >= n:
        k = int(rng.integers(n))
        return np.roll(x[:n], k).astype(np.float32), k
    total = float(np.sum(np.square(c, dtype=np.float64)))
    for _ in range(s["shift_tries"]):
        offset = int(rng.integers(-len(c) + 1, n))
        out = place(c, offset, n)
        if float(np.sum(np.square(out, dtype=np.float64))) >= s["min_target_energy_keep"] * total:
            return out, offset
    offset = int(rng.integers(0, n - len(c) + 1))
    return place(c, offset, n), offset


def shift_noise(x, rng, s, n):
    """Noise-only window: random position, never cut (all of it stays inside)."""
    c = trim_padding(x, s["trim_abs_threshold"])
    if not len(c):
        return x[:n].astype(np.float32)
    if len(c) >= n:
        return np.roll(x[:n], int(rng.integers(n))).astype(np.float32)
    return place(c, int(rng.integers(0, n - len(c) + 1)), n)


def loop_to_length(x, rng, s, n):
    """Noise bed for mixing: trimmed noise looped from a random phase."""
    c = trim_padding(x, s["trim_abs_threshold"])
    if len(c) < 2:
        c = x
    start = int(rng.integers(len(c)))
    return np.resize(np.roll(c, -start), n).astype(np.float32)


def load_crop(path, slot, sr, seconds, rng, s, require_event=False):
    dur = duration_seconds(path)
    if dur <= seconds + 1e-6:
        return load_audio(path, sr, seconds), 0.0
    lo, hi = slot
    peak_db, peak_time = source_profile(path, sr) if require_event else (None, None)
    for _ in range(s["crop_tries"]):
        start = float(rng.uniform(lo, hi))
        x = load_audio(path, sr, seconds, start)
        if not require_event or max_frame_dbfs(x, sr) >= peak_db - s["long_target_peak_db"]:
            return x, start
    start = float(np.clip(peak_time - seconds / 2, 0, dur - seconds))
    return load_audio(path, sr, seconds, start), start


def mix_noise(target, noise_bed, snr_db, rng, s, reference=None):
    """SNR = full-window target RMS (or `reference`) vs noise RMS over the noise span."""
    n = len(target)
    if rng.random() < s["partial_overlap_prob"]:
        m = int(round(rng.uniform(s["partial_overlap_min_fraction"], 1.0) * n))
        m = max(1, min(n, m))
        start = int(rng.integers(0, n - m + 1))
        span = (start, start + m)
    else:
        span = (0, n)
    segment = noise_bed[:span[1] - span[0]]
    rt, rn = (rms(target) if reference is None else reference), rms(segment)
    if min(rt, rn) < 1e-10:
        raise ValueError("Cannot mix a silent target/noise at a defined SNR")
    noise = np.zeros(n, dtype=np.float32)
    noise[span[0]:span[1]] = segment * (rt / (rn * 10 ** (snr_db / 20)))
    return limit_peak(target + noise), span


def render_augmented(row, slot, cfg, s, mode, rng, noise_pool):
    """Returns (waveform, info). Labels are never changed by augmentation."""
    sr, seconds = cfg["sample_rate"], cfg["window_seconds"]
    n = round(sr * seconds)
    info = {"kind": row["kind"]}
    if row.get("source") == "long":
        level = float(rng.uniform(*s["target_rms_range"]))
        x = load_audio(row["target_path"], sr, seconds, row["target_start"]) * (level / row["ref_rms"])
        info["crop_start"] = row["target_start"]
        if row["kind"] == "mixed" and mode == "mixed":
            noise_path = noise_pool[int(rng.integers(len(noise_pool)))]
            raw, _ = load_crop(noise_path, (0.0, max(0.0, duration_seconds(noise_path) - seconds)),
                               sr, seconds, rng, s)
            info.update(noise_path=noise_path, snr_db=float(rng.uniform(*s["snr_range_db"])))
            wave, info["noise_span"] = mix_noise(x, loop_to_length(raw, rng, s, n), info["snr_db"],
                                                 rng, s, reference=long_noise_reference(row, x, level))
            return wave, info
        return limit_peak(x), info
    if row["kind"] == "silence":
        return np.zeros(n, dtype=np.float32), info
    if row["target_path"]:
        x, info["crop_start"] = load_crop(row["target_path"], slot, sr, seconds, rng, s, require_event=True)
        if rms(x) < 1e-10:
            raise ValueError(f"Silent target crop: {row['target_path']}")
        if s["time_shift"]:
            x, info["onset_sample"] = shift_target(x, rng, s, n)
        x = x * (float(rng.uniform(*s["target_rms_range"])) / rms(x))
        if row["kind"] == "mixed" and mode == "mixed":
            if not noise_pool:
                raise ValueError("Mixed training rows need a non-empty train noise pool")
            noise_path = noise_pool[int(rng.integers(len(noise_pool)))]
            raw, _ = load_crop(noise_path, (0.0, max(0.0, duration_seconds(noise_path) - seconds)),
                               sr, seconds, rng, s)
            info.update(noise_path=noise_path, snr_db=float(rng.uniform(*s["snr_range_db"])))
            wave, info["noise_span"] = mix_noise(x, loop_to_length(raw, rng, s, n), info["snr_db"], rng, s)
            return wave, info
        return limit_peak(x), info
    x, info["crop_start"] = load_crop(row["noise_path"], slot, sr, seconds, rng, s)
    if s["time_shift"]:
        x = shift_noise(x, rng, s, n)
    info["gain_db"] = float(rng.uniform(*s["negative_gain_db"]))
    return limit_peak(x * 10 ** (info["gain_db"] / 20)), info


class AugmentedRecipeDataset(Dataset):
    """Train-split dataset that re-renders every item each epoch (call set_epoch)."""

    def __init__(self, rows, cfg, mode="mixed"):
        self.rows, self.cfg, self.mode = rows, cfg, mode
        self.settings = augment_settings(cfg)
        self.seed, self.epoch = int(cfg["seed"]), 0
        window = cfg["window_seconds"]
        self.slots = [(i, lo, hi) for i, r in enumerate(rows)
                      for lo, hi in ([(r["target_start"], r["target_start"])] if r.get("source") == "long"
                                     else crop_slots(r["target_path"] or r["noise_path"], window, self.settings))]
        self.noise_pool = sorted({r["noise_path"] for r in rows if r.get("noise_path")})
        if mode == "mixed" and any(r["kind"] == "mixed" for r in rows) and not self.noise_pool:
            raise ValueError("No train noise files for on-the-fly mixing")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.slots)

    def render(self, index):
        row_index, lo, hi = self.slots[index]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        return render_augmented(self.rows[row_index], (lo, hi), self.cfg, self.settings,
                                self.mode, rng, self.noise_pool)

    def __getitem__(self, index):
        wave, _ = self.render(index)
        labels = self.rows[self.slots[index][0]]["labels"]
        return torch.from_numpy(wave), torch.tensor(labels, dtype=torch.float32)
