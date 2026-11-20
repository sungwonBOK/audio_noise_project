import csv
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def config(path):
    c = read_json(path)
    classes = c["classes"]
    if len(classes) != 5 or len(set(classes)) != 5:
        raise ValueError("Exactly five unique target classes are required")
    if len(c["class_names_ko"]) != 5:
        raise ValueError("Five display names are required")
    if c["sample_rate"] != 32000 or c["window_seconds"] < 1:
        raise ValueError("This implementation uses standard 32 kHz Cnn14; window must be >= 1s")
    if not 0 < c["hop_seconds"] <= c["window_seconds"]:
        raise ValueError("Require 0 < hop_seconds <= window_seconds")
    folds = [set(c[f"{s}_folds"]) for s in ("train", "val", "test")]
    if any(not x for x in folds) or any(folds[i] & folds[j] for i in range(3) for j in range(i)):
        raise ValueError("Train/val/test folds must be nonempty and disjoint")
    seen, unseen = set(c["seen_noise"]), set(c["unseen_noise"])
    if not seen or not unseen or seen & unseen or set(classes) & (seen | unseen):
        raise ValueError("Target, seen noise and unseen noise classes must be separate")
    if c["batch_size"] < 1 or c["head_epochs"] < 1 or c["finetune_epochs"] < 0:
        raise ValueError("Invalid batch size/epochs")
    lo, hi = c["foreground_rms_range"]
    if not 0 < lo <= hi or not c["snrs_db"]:
        raise ValueError("Invalid foreground amplitude or SNR list")
    return c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install CUDA-enabled PyTorch and check NVIDIA driver. CPU requires explicit --device cpu.")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Freesound file names: FSD50K "64760.wav", Freesound download "64760__user__title.wav".
_FREESOUND_ID = re.compile(r"^(\d+)(?:__|$)")


def freesound_id(path):
    """Freesound sound ID from a file name, or None if the name does not start with one."""
    m = _FREESOUND_ID.match(Path(path).stem)
    return str(int(m.group(1))) if m else None


def esc50_source_ids(esc_root):
    """Freesound IDs of ALL 2,000 ESC-50 clips (column src_file), every fold."""
    meta = Path(esc_root) / "meta" / "esc50.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"{meta} (needed to find recordings shared with ESC-50; "
                                "run download_assets.py --esc50)")
    with meta.open(encoding="utf-8-sig", newline="") as f:
        return {str(int(r["src_file"])) for r in csv.DictReader(f)}


def load_checkpoint(path):
    # Project checkpoints contain tensors and primitive data only.
    return torch.load(path, map_location="cpu", weights_only=True)


def read_manifest(path, cfg):
    m = read_json(path)
    for key in ("classes", "sample_rate", "window_seconds", "seed", "train_folds", "val_folds",
                "test_folds", "seen_noise", "unseen_noise", "snrs_db", "foreground_rms_range", "silence_examples"):
        if m["config"][key] != cfg[key]:
            raise ValueError(f"Manifest/config mismatch: {key}. Run prepare.py again.")
    return m
