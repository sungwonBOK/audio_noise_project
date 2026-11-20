"""Window-level pseudo labels from the ORIGINAL AudioSet Cnn14 (a teacher we never trained).

사람용 안내는 PLAY.md를 보세요.

Three subcommands:
  calibrate  Run the teacher on clean ESC-50 clips of the train/val folds (test fold untouched)
             and choose, per class, two thresholds:
               pos_th: lowest threshold whose teacher-positive clips are >= --precision correct
               neg_th: threshold that leaves at most --miss of the true positives below it
  check      Score those labels on the held-out ESC-50 test fold (true labels known).
  label      Slide the same windows as infer.py over CLEAN recordings in
             <audio-root>/<split>/<class>/ and write one row per window and class:
               1 = teacher prob >= pos_th, 0 = prob < neg_th, -1 = uncertain
             (-1 is ignored by the training loss and by the metrics; a person may overwrite it).

Why clean recordings: the teacher is reliable on clean audio (ESC-50 AP 0.94-1.0) and much
weaker under noise, so labels are made BEFORE noise is mixed in by prepare.py/train.py.
The folder name (<class>) is only a sanity check; the teacher's window labels are the labels.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_utils import load_audio, rms, window_intervals
from common import ROOT, config, sha256, write_json
from model import SoundDetector

# AudioSet index per target class (same "strict" mapping as analysis/baseline_pretrained_cnn14.py).
# Broader mappings were worse on validation (e.g. adding "Door"/"Wood" made knock fire on noise).
AUDIOSET_INDEX = {"car_horn": 308,          # Vehicle horn, car horn, honking
                  "siren": 396,             # Siren
                  "door_wood_knock": 359,   # Knock
                  "crying_baby": 23,        # Baby cry, infant cry
                  "glass_breaking": 443}    # Shatter
SPLITS = ("train", "val", "test")
NO_TARGET = "none"
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
DEFAULT_CALIBRATION = ROOT / "runs" / "teacher_calibration" / "calibration.json"


class Teacher:
    def __init__(self, weights, batch, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        model = SoundDetector()
        model.load_pretrained(weights)
        self.backbone = model.backbone.eval().to(self.device)
        self.batch = batch
        self.weights_sha256 = sha256(weights)
        names = {}
        with (ROOT / "data" / "audioset_class_labels_indices.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                names[int(r["index"])] = r["display_name"]
        self.names = names

    def __call__(self, waves):
        """List of equal-length waveforms -> (N, 527) AudioSet probabilities."""
        out = []
        with torch.inference_mode():
            for i in range(0, len(waves), self.batch):
                x = torch.from_numpy(np.stack(waves[i:i + self.batch])).to(self.device)
                out.append(self.backbone(x)["clipwise_output"].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 527), np.float32)


def to_classes(p527, classes):
    return np.stack([p527[:, AUDIOSET_INDEX[c]] for c in classes], axis=1)


def apply_thresholds(p, pos_th, neg_th):
    """1 / 0 / -1(uncertain) per window and class."""
    labels = np.full(p.shape, -1, dtype=np.int64)
    labels[p < np.asarray(neg_th)] = 0
    labels[p >= np.asarray(pos_th)] = 1
    return labels


def pick_thresholds(y, p, precision, miss, min_selected=5):
    pos_th, neg_th, details = [], [], []
    grid = np.round(np.arange(0.005, 1.0, 0.005), 3)
    for i in range(y.shape[1]):
        yy, pp = y[:, i].astype(bool), p[:, i]
        if not yy.any() or yy.all():
            raise ValueError(f"class {i}: calibration needs positive and negative clips")
        chosen = None
        for t in grid:
            sel = yy[pp >= t]
            if len(sel) >= min_selected and sel.mean() >= precision:
                chosen = float(t)
                break
        if chosen is None:
            raise ValueError(f"class {i}: teacher never reaches precision {precision}")
        low = float(np.quantile(pp[yy], miss, method="lower"))
        low = min(low, chosen)
        pos_th.append(chosen)
        neg_th.append(low)
        details.append({"positives": int(yy.sum()), "negatives": int((~yy).sum()),
                        "positives_labeled_1": float((pp[yy] >= chosen).mean()),
                        "positives_labeled_0": float((pp[yy] < low).mean()),
                        "negatives_labeled_0": float((pp[~yy] < low).mean()),
                        "negatives_labeled_1": float((pp[~yy] >= chosen).mean())})
    return pos_th, neg_th, details


def esc_clips(esc, folds, classes):
    with (Path(esc) / "meta" / "esc50.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["fold"]) in folds]
    y = np.array([[int(r["category"] == c) for c in classes] for r in rows])
    return rows, y


def score_esc(teacher, esc, rows, cfg):
    sr, seconds = cfg["sample_rate"], cfg["window_seconds"]
    probs, t0 = [], time.time()
    step = 64
    for i in range(0, len(rows), step):
        waves = [load_audio(Path(esc) / "audio" / r["filename"], sr, seconds) for r in rows[i:i + step]]
        probs.append(to_classes(teacher(waves), cfg["classes"]))
        print(f"  {min(i + step, len(rows))}/{len(rows)} clips  {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(probs)


def label_report(y, labels, classes):
    report = {}
    for i, c in enumerate(classes):
        pos, neg = y[:, i] == 1, y[:, i] == 0
        report[c] = {f"{name}_labeled_{v}": float((labels[mask, i] == v).mean())
                     for name, mask in (("positives", pos), ("negatives", neg)) for v in (1, 0, -1)}
        decided = labels[:, i] >= 0
        correct = (labels[decided, i] == y[decided, i]).mean() if decided.any() else float("nan")
        report[c]["accuracy_of_decided_labels"] = float(correct)
    return report


def cmd_calibrate(a, cfg):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"Already exists (keep it or choose a new --out): {out}")
    folds = cfg["train_folds"] + cfg["val_folds"]
    rows, y = esc_clips(a.esc50, folds, cfg["classes"])
    teacher = Teacher(a.weights, a.batch, a.device)
    print(f"Teacher on {len(rows)} clean ESC-50 clips, folds {folds}", flush=True)
    p = score_esc(teacher, a.esc50, rows, cfg)
    pos_th, neg_th, details = pick_thresholds(y, p, a.precision, a.miss)
    result = {"classes": cfg["classes"], "audioset_index": [AUDIOSET_INDEX[c] for c in cfg["classes"]],
              "audioset_names": [teacher.names[AUDIOSET_INDEX[c]] for c in cfg["classes"]],
              "pos_th": pos_th, "neg_th": neg_th, "precision_target": a.precision, "miss_target": a.miss,
              "calibration_folds": folds, "per_class": dict(zip(cfg["classes"], details)),
              "teacher_weights_sha256": teacher.weights_sha256,
              "sample_rate": cfg["sample_rate"], "window_seconds": cfg["window_seconds"],
              "note": "Clean ESC-50 clips only. Test fold not used; see `check`."}
    write_json(out, result)
    np.savez_compressed(out.with_name("calibration_probs.npz"), probs=p, labels=y)
    for c, lo, hi in zip(cfg["classes"], neg_th, pos_th):
        print(f"{c:16s} 0 if p < {lo:.4f}   1 if p >= {hi:.3f}   otherwise -1")
    print("Saved", out)


def cmd_check(a, cfg):
    cal = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    rows, y = esc_clips(a.esc50, cfg["test_folds"], cfg["classes"])
    teacher = Teacher(a.weights, a.batch, a.device)
    print(f"Teacher on {len(rows)} clean ESC-50 test-fold clips {cfg['test_folds']}", flush=True)
    p = score_esc(teacher, a.esc50, rows, cfg)
    labels = apply_thresholds(p, cal["pos_th"], cal["neg_th"])
    report = {"folds": cfg["test_folds"], "clips": len(rows), "per_class": label_report(y, labels, cfg["classes"])}
    out = Path(a.calibration).with_name("check_test_fold.json")
    write_json(out, report)
    for c, r in report["per_class"].items():
        print(f"{c:16s} positives: 1={r['positives_labeled_1']:.2f} -1={r['positives_labeled_-1']:.2f} "
              f"0={r['positives_labeled_0']:.2f} | negatives: 0={r['negatives_labeled_0']:.3f} "
              f"-1={r['negatives_labeled_-1']:.3f} 1={r['negatives_labeled_1']:.3f}")
    print("Saved", out)


def find_audio(root, classes):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Audio folder not found: {root}")
    allowed = set(classes) | {NO_TARGET}
    found, problems = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        if len(parts) != 3 or parts[0] not in SPLITS or parts[1] not in allowed:
            problems.append(str(path.relative_to(root)))
            continue
        found.append((parts[0], parts[1], path.resolve()))
    if problems:
        raise SystemExit("Files must be at <split>/<class>/<file>, split in "
                         f"{SPLITS}, class in {sorted(allowed)}. Wrong place:\n  " + "\n  ".join(problems[:20]))
    if not found:
        raise SystemExit(f"No audio files ({', '.join(sorted(AUDIO_SUFFIXES))}) under {root}")
    return found


def cmd_label(a, cfg):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output folder not empty (it may hold hand-edited labels): {out}")
    cal_path = Path(a.calibration)
    if not cal_path.is_file():
        raise SystemExit(f"Calibration missing: {cal_path}\nRun once: python pseudo_label.py calibrate")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    classes = cfg["classes"]
    if cal["classes"] != classes or cal["window_seconds"] != cfg["window_seconds"]:
        raise SystemExit("Calibration was made for different classes/window; run calibrate again")
    files = find_audio(a.audio_root, classes)
    teacher = Teacher(a.weights, a.batch, a.device)
    if teacher.weights_sha256 != cal["teacher_weights_sha256"]:
        raise SystemExit("Calibration belongs to different teacher weights")
    sr, seconds = cfg["sample_rate"], cfg["window_seconds"]
    hop = a.hop_seconds
    window_rows, file_rows = [], []
    t0 = time.time()
    for n, (split, file_class, path) in enumerate(files, 1):
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        spans = list(window_intervals(duration, seconds, hop))
        probs, levels, tops = [], [], []
        for i in range(0, len(spans), a.batch):
            waves = [load_audio(path, sr, seconds, s) for s, _ in spans[i:i + a.batch]]
            p527 = teacher(waves)
            probs.append(to_classes(p527, classes))
            levels += [rms(w) for w in waves]
            for row in p527:
                top = np.argsort(row)[::-1][:3]
                tops.append("; ".join(f"{teacher.names[k]} {row[k]:.2f}" for k in top))
        p = np.concatenate(probs) if probs else np.zeros((0, len(classes)))
        labels = apply_thresholds(p, cal["pos_th"], cal["neg_th"])
        for (s, e), pp, ll, lv, top in zip(spans, p, labels, levels, tops):
            window_rows.append({"split": split, "file_class": file_class, "path": str(path),
                                "start_seconds": round(s, 3), "end_seconds": round(e, 3),
                                "window_rms": f"{lv:.6g}",
                                **{f"p_{c}": f"{v:.4f}" for c, v in zip(classes, pp)},
                                **{f"label_{c}": int(v) for c, v in zip(classes, ll)},
                                "uncertain": int((ll == -1).any()), "teacher_top3": top})
        flags = []
        if file_class in classes and not (labels[:, classes.index(file_class)] == 1).any():
            flags.append(f"no window labeled 1 for {file_class}")
        if file_class == NO_TARGET and (labels == 1).any():
            flags.append("teacher found a target in a 'none' recording")
        others = [c for i, c in enumerate(classes) if c != file_class and (labels[:, i] == 1).any()]
        if file_class in classes and others:
            flags.append("other targets labeled 1: " + ", ".join(others))
        file_rows.append({"split": split, "file_class": file_class, "path": str(path),
                          "duration_seconds": round(duration, 2), "windows": len(spans),
                          **{f"windows_1_{c}": int((labels[:, i] == 1).sum()) for i, c in enumerate(classes)},
                          "uncertain_windows": int((labels == -1).any(1).sum()),
                          "needs_review": int(bool(flags)), "review_reason": " | ".join(flags)})
        print(f"[{n}/{len(files)}] {split}/{file_class}/{path.name}: {len(spans)} windows, "
              f"{file_rows[-1]['uncertain_windows']} uncertain{'  REVIEW: ' + file_rows[-1]['review_reason'] if flags else ''}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("windows.csv", window_rows), ("files.csv", file_rows)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write_json(out / "meta.json", {"audio_root": str(Path(a.audio_root).resolve()), "classes": classes,
               "window_seconds": seconds, "hop_seconds": hop, "sample_rate": sr,
               "calibration": str(cal_path.resolve()), "calibration_sha256": sha256(cal_path),
               "pos_th": cal["pos_th"], "neg_th": cal["neg_th"],
               "teacher_weights_sha256": teacher.weights_sha256,
               "files": len(file_rows), "windows": len(window_rows),
               "uncertain_windows": sum(r["uncertain"] for r in window_rows),
               "files_needing_review": sum(r["needs_review"] for r in file_rows),
               "label_values": "1 present, 0 absent, -1 uncertain (ignored in training/metrics)"})
    print(f"Saved {out}: {len(file_rows)} files, {len(window_rows)} windows, "
          f"{sum(r['needs_review'] for r in file_rows)} files need review")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "check", "label"):
        s = sub.add_parser(name)
        s.add_argument("--config", default=str(ROOT / "config.json"))
        s.add_argument("--weights", default=str(ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"))
        s.add_argument("--batch", type=int, default=16)
        s.add_argument("--device", default="auto", help="auto = cuda if available, else cpu")
        if name in ("calibrate", "check"):
            s.add_argument("--esc50", default=str(ROOT / "data" / "ESC-50-master"))
    c = sub.choices["calibrate"]
    c.add_argument("--out", default=str(DEFAULT_CALIBRATION))
    c.add_argument("--precision", type=float, default=0.95, help="Required precision of label 1")
    c.add_argument("--miss", type=float, default=0.05, help="Max share of true positives labeled 0")
    sub.choices["check"].add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    lab = sub.choices["label"]
    lab.add_argument("--audio-root", default=str(ROOT / "data" / "long_audio"))
    lab.add_argument("--out", default=str(ROOT / "data" / "long_audio_labels"))
    lab.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    lab.add_argument("--hop-seconds", type=float, default=2.5,
                     help="Window step for labels/training (inference may use a smaller hop)")
    a = p.parse_args()
    cfg = config(a.config)
    {"calibrate": cmd_calibrate, "check": cmd_check, "label": cmd_label}[a.command](a, cfg)


if __name__ == "__main__":
    main()
