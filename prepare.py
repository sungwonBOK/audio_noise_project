"""Split originals FIRST, then save deterministic mixture recipes (no WAV duplication)."""
import argparse
import csv
import random
from pathlib import Path

from common import config, read_json, sha256, write_json, ROOT


def build_manifest(meta_rows, audio_dir, cfg):
    groups = {}
    all_rows = []
    expected = set(cfg["classes"] + cfg["seen_noise"] + cfg["unseen_noise"])
    available = {r["category"] for r in meta_rows}
    if expected - available:
        raise ValueError(f"Unknown ESC-50 categories: {sorted(expected - available)}")
    for split in ("train", "val", "test"):
        rows = []
        for r in meta_rows:
            if int(r["fold"]) not in cfg[f"{split}_folds"] or r["category"] not in expected:
                continue
            if split == "train" and r["category"] in cfg["unseen_noise"]:
                continue
            source = r["src_file"]
            if source in groups and groups[source] != split:
                raise ValueError(f"Source leakage between splits: {source}")
            groups[source] = split
            p = (audio_dir / r["filename"]).resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            rows.append({**r, "path": str(p)})
        targets = [r for r in rows if r["category"] in cfg["classes"]]
        pools = {name: [r for r in rows if r["category"] in cfg[f"{name}_noise"]]
                 for name in ("seen", "unseen")}
        required = ("seen",) if split == "train" else ("seen", "unseen")
        if any(not pools[k] for k in required):
            raise ValueError(f"Missing noise pool in {split}")
        if any(not any(r["category"] == c for r in targets) for c in cfg["classes"]):
            raise ValueError(f"Missing target class in {split}")
        rng = random.Random(cfg["seed"] + {"train": 0, "val": 1, "test": 2}[split])
        recipes = []
        for t in targets:
            label = [int(c == t["category"]) for c in cfg["classes"]]
            gain = rng.uniform(*cfg["foreground_rms_range"])
            base = dict(split=split, labels=label, target_path=t["path"], target_source=t["src_file"],
                        target_class=t["category"], target_rms=gain, noise_path="", noise_source="",
                        noise_class="", snr_db=None, noise_group="none")
            recipes.append({**base, "kind": "clean"})
            for group in required:
                for snr in cfg["snrs_db"]:
                    noise = rng.choice(pools[group])
                    recipes.append({**base, "kind": "mixed", "snr_db": snr,
                                    "noise_group": group, "noise_path": noise["path"],
                                    "noise_source": noise["src_file"], "noise_class": noise["category"]})
        for group in required:
            for n in pools[group]:
                recipes.append(dict(split=split, labels=[0]*5, target_path="", target_source="",
                                    target_class="", target_rms=0, noise_path=n["path"],
                                    noise_source=n["src_file"], noise_class=n["category"],
                                    snr_db=None, noise_group=group, kind="negative"))
        for _ in range(cfg["silence_examples"]):
            recipes.append(dict(split=split, labels=[0]*5, target_path="", noise_path="",
                                target_source="", noise_source="", target_class="", noise_class="",
                                target_rms=0, snr_db=None, noise_group="none", kind="silence"))
        for i, row in enumerate(recipes):
            row["id"] = f"{split}-{i:06d}"
        all_rows.extend(recipes)
    return {"version": 1, "config": cfg, "rows": all_rows,
            "notes": "ESC-50 fold split before recipe generation; SNR uses full-window RMS."}


def add_long_rows(manifest, labels_dir, cfg):
    """Append windows of long CLEAN recordings labeled by pseudo_label.py (labels 1/0/-1).

    Per window: one clean row + one mixed row per noise group of that split (train: seen;
    val/test: seen and unseen), noise drawn from the same ESC-50 split pools, SNR from snrs_db.
    One gain per file. Windows whose five labels are all -1 carry no information and are dropped.
    """
    labels_dir = Path(labels_dir)
    meta = read_json(labels_dir / "meta.json")
    for key in ("classes", "window_seconds", "sample_rate"):
        if meta[key] != cfg[key]:
            raise ValueError(f"Long-audio labels/config mismatch: {key}. Run pseudo_label.py label again.")
    with (labels_dir / "windows.csv").open(encoding="utf-8-sig", newline="") as f:
        windows = list(csv.DictReader(f))
    if not windows:
        raise ValueError("windows.csv has no rows")
    classes = cfg["classes"]
    by_file = {}
    for w in windows:
        by_file.setdefault(w["path"], []).append(w)
    names = {}
    for path, ws in by_file.items():
        if len({w["split"] for w in ws}) != 1:
            raise ValueError(f"One file must stay in one split: {path}")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        names.setdefault(Path(path).name, set()).add(ws[0]["split"])
    leaked = sorted(n for n, s in names.items() if len(s) > 1)
    if leaked:
        raise ValueError(f"Same file name in several splits (possible leakage): {leaked[:10]}")
    pools = {}
    for r in manifest["rows"]:
        if r["kind"] == "negative":
            pools.setdefault((r["split"], r["noise_group"]), []).append(r)
    rng = random.Random(cfg["seed"] + 100)
    rows, dropped, silent_files = [], 0, []
    for path, ws in by_file.items():
        split = ws[0]["split"]
        ref = max(float(w["window_rms"]) for w in ws)
        if ref < 1e-6:
            silent_files.append(path)
            continue
        gain = rng.uniform(*cfg["foreground_rms_range"])
        groups = ("seen",) if split == "train" else ("seen", "unseen")
        for w in ws:
            labels = [int(w[f"label_{c}"]) for c in classes]
            if any(v not in (-1, 0, 1) for v in labels):
                raise ValueError(f"Labels must be 1, 0 or -1: {path} @ {w['start_seconds']}s")
            if all(v == -1 for v in labels):
                dropped += 1
                continue
            base = dict(split=split, labels=labels, target_path=path, target_source=f"long:{Path(path).name}",
                        target_class=w["file_class"], target_rms=gain, ref_rms=ref,
                        target_start=float(w["start_seconds"]), noise_path="", noise_source="",
                        noise_class="", snr_db=None, noise_group="none", source="long")
            rows.append({**base, "kind": "clean"})
            for group in groups:
                noise = rng.choice(pools[(split, group)])
                rows.append({**base, "kind": "mixed", "snr_db": rng.choice(cfg["snrs_db"]),
                             "noise_group": group, "noise_path": noise["noise_path"],
                             "noise_source": noise["noise_source"], "noise_class": noise["noise_class"]})
    for i, row in enumerate(rows):
        row["id"] = f"{row['split']}-long-{i:06d}"
    manifest["rows"].extend(rows)
    manifest["long_labels"] = {"dir": str(labels_dir.resolve()), "windows_sha256": sha256(labels_dir / "windows.csv"),
                               "files": len(by_file) - len(silent_files), "rows": len(rows),
                               "dropped_all_uncertain_windows": dropped, "skipped_silent_files": silent_files}
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esc50", required=True, help="Root containing audio/ and meta/esc50.csv")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--out", default="data/manifest.json")
    p.add_argument("--long-labels", default=None,
                   help="Folder made by `pseudo_label.py label` (windows.csv, meta.json); adds long-recording windows")
    a = p.parse_args()
    root = Path(a.esc50)
    with (root / "meta" / "esc50.csv").open(encoding="utf-8-sig", newline="") as f:
        meta = list(csv.DictReader(f))
    cfg = config(a.config)
    m = build_manifest(meta, root / "audio", cfg)
    if a.long_labels:
        m = add_long_rows(m, a.long_labels, cfg)
    write_json(a.out, m)
    for split in ("train", "val", "test"):
        rows = [r for r in m["rows"] if r["split"] == split]
        print(split, len(rows), {k: sum(r["kind"] == k for r in rows) for k in ("clean", "mixed", "negative", "silence")},
              "long-recording windows:", sum(r.get("source") == "long" for r in rows))
    print("Saved", a.out)


if __name__ == "__main__":
    main()
