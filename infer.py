"""File demo: overlapping analysis windows, original timestamps, optional RMS gate."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_utils import load_audio, max_frame_dbfs, window_intervals
from common import device_for, load_checkpoint, read_json, sha256, synchronize, write_json
from model import SoundDetector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rms-db", type=float, default=None)
    p.add_argument("--hop-seconds", type=float, default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    setup_start = time.perf_counter()
    ckpt = load_checkpoint(a.checkpoint)
    cfg = ckpt["config"]
    threshold_info = read_json(a.thresholds)
    if threshold_info["checkpoint_sha256"] != sha256(a.checkpoint) or threshold_info["classes"] != cfg["classes"]:
        raise ValueError("Thresholds do not match checkpoint/class order")
    thresholds = np.asarray(threshold_info["thresholds"])
    device = device_for(a.device or cfg["device"])
    model = SoundDetector()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    with torch.inference_mode():
        model(torch.zeros(1, round(cfg["sample_rate"] * cfg["window_seconds"]), device=device))
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    started = time.perf_counter()
    info = sf.info(a.audio)
    if info.frames == 0:
        raise ValueError("Empty audio file")
    duration = info.frames / info.samplerate
    records = []
    timing = dict(audio_read_resample_seconds=0.0, rms_seconds=0.0, model_seconds=0.0,
                  model_windows=0, skipped_windows=0)
    for start, end in window_intervals(duration, cfg["window_seconds"], a.hop_seconds if a.hop_seconds is not None else cfg["hop_seconds"]):
        t = time.perf_counter()
        wave = load_audio(a.audio, cfg["sample_rate"], cfg["window_seconds"], start)
        timing["audio_read_resample_seconds"] += time.perf_counter() - t
        skip, level = False, None
        if a.rms_db is not None:
            t = time.perf_counter()
            level = max_frame_dbfs(wave, cfg["sample_rate"], cfg["rms_frame_seconds"], cfg["rms_hop_seconds"])
            skip = level < a.rms_db
            timing["rms_seconds"] += time.perf_counter() - t
        probs = np.zeros(5)
        if not skip:
            synchronize(device)
            t = time.perf_counter()
            with torch.inference_mode():
                probs = model(torch.from_numpy(wave).unsqueeze(0).to(device)).sigmoid()[0].cpu().numpy()
            synchronize(device)
            timing["model_seconds"] += time.perf_counter() - t
            timing["model_windows"] += 1
        else:
            timing["skipped_windows"] += 1
        if not np.isfinite(probs).all():
            raise RuntimeError("Non-finite prediction")
        detected = [cfg["class_names_ko"][i] for i in range(5) if probs[i] >= thresholds[i]]
        row = dict(start_seconds=start, end_seconds=end, rms_skipped=skip,
                   max_frame_dbfs=level, detected=" + ".join(detected),
                   **{c: float(probs[i]) for i, c in enumerate(cfg["classes"])})
        records.append(row)
        print(f"{start:7.2f} ~ {end:7.2f}s: {row['detected'] or ('RMS 제외' if skip else '목표음 미검출')}")
    timing["analysis_pipeline_seconds"] = time.perf_counter() - started
    timing["setup_and_warmup_seconds"] = setup_seconds
    timing["audio_duration_seconds"] = duration
    timing["real_time_factor"] = timing["analysis_pipeline_seconds"] / duration
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "windows.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    write_json(out / "result.json", {"audio": str(Path(a.audio).resolve()), "classes": cfg["classes"],
               "checkpoint_sha256": threshold_info["checkpoint_sha256"], "thresholds": thresholds.tolist(),
               "rms_dbfs": a.rms_db, "timing": timing, "windows": records,
               "note": "Timestamps are analysis-window bounds, not exact event onset/offset. Multi-target overlaps are outside training/evaluation scope."})
    print(timing)


if __name__ == "__main__":
    main()
