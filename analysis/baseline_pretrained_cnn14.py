"""Off-the-shelf PANNs Cnn14 (AudioSet 527-class head, no training) on OUR val/test windows.

Same windows as runs/noise_expansion_20261115 (stored manifest, paths remapped to local ESC-50),
same threshold rule (validation F1 grid, metrics.tune_thresholds), same test metrics (metrics.scores).
AudioSet labels -> our 5 classes via a mapping fixed BEFORE looking at results:
  strict: one label per class; broad: max over a few related labels.
Also recomputes AP and per-condition recall for the stored predictions of our 8 trained models.
Analysis only: does not modify any code or existing result.
"""
import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_utils import render_recipe  # noqa: E402
from metrics import scores, tune_thresholds  # noqa: E402
from model import SoundDetector  # noqa: E402

EXP = ROOT / "runs" / "noise_expansion_20261115"
CLASSES = ["car_horn", "siren", "door_wood_knock", "crying_baby", "glass_breaking"]
MAPPINGS = {
    "strict": [[308], [396], [359], [23], [443]],
    "broad": [[308, 309, 318], [396, 323, 324, 325, 397], [359, 354, 360, 437], [23, 22], [443, 441, 470]],
}


def load_rows(esc_audio):
    m = json.loads((EXP / "noise_30" / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in m["rows"] if r["split"] in ("val", "test")]
    for r in rows:
        for k in ("target_path", "noise_path"):
            if r[k]:
                r[k] = str(esc_audio / Path(r[k].replace("\\", "/")).name)
    return rows, m["config"]


def read_predictions(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = (ast.literal_eval(r["labels"]), ast.literal_eval(r["scores"]))
    return out


def condition(r):
    if r["kind"] == "mixed":
        return f"mixed_{int(r['snr_db'])}dB"
    return r["kind"]  # clean / noise / silence


def ap_per_class(y, p):
    from sklearn.metrics import average_precision_score
    return [float(average_precision_score(y[:, i], p[:, i])) for i in range(5)]


def summarize(name, rows_by_split, y, p, thresholds):
    test = rows_by_split["test"]
    res = scores(y["test"], p["test"], thresholds)
    pos_recall = {}
    pred = p["test"] >= np.asarray(thresholds)
    for cond in ["clean", "mixed_10dB", "mixed_5dB", "mixed_0dB"]:
        idx = [i for i, r in enumerate(test) if condition(r) == cond]
        yy = y["test"][idx].astype(bool)
        pos_recall[cond] = float((pred[idx] & yy).sum() / yy.sum())
    ap = ap_per_class(y["test"], p["test"])
    return {"model": name, "thresholds": [round(t, 2) for t in thresholds],
            "test_macro_f1": round(res["macro_f1"], 4),
            "test_f1": [round(v, 4) for v in res["f1"]],
            "test_macro_precision": round(float(np.mean(res["precision"])), 4),
            "test_macro_recall": round(float(np.mean(res["recall"])), 4),
            "negative_fp": f"{res['negative_false_positive_windows']}/{res['negative_windows']}",
            "test_macro_ap": round(float(np.mean(ap)), 4), "test_ap": [round(v, 4) for v in ap],
            **{f"recall_{k}": round(v, 4) for k, v in pos_recall.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esc50", default=str(ROOT / "data" / "ESC-50-master"))
    p.add_argument("--weights", default=str(ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"))
    p.add_argument("--out", default=str(ROOT / "runs" / "baseline_pretrained_01"))
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output folder not empty, choose a new --out: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows, cfg = load_rows(Path(a.esc50) / "audio")
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("val", "test")}
    stored_test = read_predictions(EXP / "noise_30" / "mixed" / "test" / "predictions.csv")
    stored_val = read_predictions(EXP / "noise_30" / "mixed" / "validation" / "predictions.csv")
    for s, stored in (("val", stored_val), ("test", stored_test)):
        assert [r["id"] for r in by_split[s]] == list(stored), f"{s} window order differs from stored predictions"
        assert all(stored[r["id"]][0] == r["labels"] for r in by_split[s]), f"{s} labels differ"
    print("windows:", {s: len(v) for s, v in by_split.items()}, "- ids/labels match stored predictions", flush=True)

    model = SoundDetector()
    model.load_pretrained(a.weights)
    bb = model.backbone.eval()
    probs = {}
    t0 = time.time()
    with torch.inference_mode():
        for s in ("val", "test"):
            chunks = []
            for i in range(0, len(by_split[s]), a.batch):
                wave = np.stack([render_recipe(r, cfg, "mixed") for r in by_split[s][i:i + a.batch]])
                chunks.append(bb(torch.from_numpy(wave))["clipwise_output"].numpy())
                if (i // a.batch) % 20 == 0:
                    print(f"  {s} {i + len(wave)}/{len(by_split[s])}  {time.time() - t0:.0f}s", flush=True)
            probs[s] = np.concatenate(chunks)
    np.savez_compressed(out / "audioset_527_probs.npz", **probs)

    y = {s: np.array([r["labels"] for r in by_split[s]]) for s in ("val", "test")}
    results = []
    for name, mapping in MAPPINGS.items():
        p5 = {s: np.stack([probs[s][:, idx].max(axis=1) for idx in mapping], axis=1) for s in probs}
        th = tune_thresholds(y["val"], p5["val"])
        results.append(summarize(f"pretrained_cnn14_{name} (no training)", by_split, y, p5, th))

    for stage in ["noise_06", "noise_18", "noise_30", "noise_43"]:
        for mode in ["clean", "mixed"]:
            run = EXP / stage / mode
            pt = read_predictions(run / "test" / "predictions.csv")
            th = json.loads((run / "validation" / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
            p5 = {"test": np.array([pt[r["id"]][1] for r in by_split["test"]])}
            results.append(summarize(f"ours_{stage}_{mode}", by_split, y, p5, th))

    keys = list(results[0])
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    (out / "comparison.json").write_text(json.dumps({"mappings": MAPPINGS, "results": results}, indent=2), encoding="utf-8")
    for r in results:
        print(json.dumps(r))
    print("done:", out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
