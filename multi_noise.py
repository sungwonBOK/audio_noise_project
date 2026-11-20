"""Layer several non-target noises in one window (1-stage of the 'real-life mixture' plan).

Two models, both trained in mixed mode on the 43-noise-type stage of noise_expansion.py:
  single : training data identical to that stage (one noise per mixed/negative window)
  multi  : the same training rows, but mixed and noise-only windows get 1-3 noises layered

Evaluation (thresholds always tuned on the matching validation split, never on test):
  val  -> test         one noise per window; same fingerprint as the main experiment
  val_multi -> test_multi  2-3 noises per window, built from the same val/test rows

The SNR of a mixed row is target vs the WHOLE noise bed (all noises together), 5-second RMS.
Extra noises come from the same split and noise group as the primary noise, are of different
classes, and never share a source recording with the target or each other.
"""
import argparse
import copy
import csv
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from common import ROOT, config, read_json, sha256, write_json
from noise_expansion import fingerprint, prepare_plan

DEFAULTS = {"train_noise_counts": [1, 2, 3], "eval_noise_counts": [2, 3], "extra_gain_db": [-6.0, 0.0],
            "stage": "noise_43"}
CONDITIONS = ("single", "multi")


def settings(cfg):
    s = {**DEFAULTS, **cfg.get("multi_noise", {})}
    for key in ("train_noise_counts", "eval_noise_counts"):
        if not s[key] or any(int(k) != k or k < 1 for k in s[key]):
            raise ValueError(f"multi_noise.{key} must be a nonempty list of integers >= 1")
    lo, hi = s["extra_gain_db"]
    if lo > hi:
        raise ValueError("multi_noise.extra_gain_db must be [low, high]")
    return s


def noise_pools(rows):
    """(split, noise_group) -> noise-only rows of that split; extras are drawn only from here."""
    pools = {}
    for r in rows:
        if r["kind"] == "negative":
            pools.setdefault((r["split"], r["noise_group"]), []).append(
                {"path": r["noise_path"], "source": r["noise_source"], "class": r["noise_class"]})
    return pools


def add_extra_noises(rows, pools, counts, gain_db, rng):
    """Return copies of rows; mixed/negative rows get `extra_noises` and `noise_count`.

    The number of noises is drawn from `counts`, but stops early when the pool has no more
    distinct classes (the held-out pool has only two), so `noise_count` is the real number.
    """
    out = []
    for r in rows:
        r = copy.deepcopy(r)
        if r["kind"] in ("mixed", "negative") and r.get("source") != "long":
            pool = pools[(r["split"], r["noise_group"])]
            classes, sources = {r["noise_class"]}, {r["noise_source"], r["target_source"]} - {""}
            extras = []
            for _ in range(rng.choice(counts) - 1):
                cands = [n for n in pool if n["class"] not in classes and n["source"] not in sources]
                if not cands:
                    break
                n = rng.choice(cands)
                extras.append({**n, "gain_db": round(rng.uniform(*gain_db), 3)})
                classes.add(n["class"])
                sources.add(n["source"])
            r["extra_noises"] = extras
            r["noise_count"] = 1 + len(extras)
        out.append(r)
    return out


def build(meta, audio, cfg):
    s = settings(cfg)
    plan = prepare_plan(meta, audio, cfg)
    stage = next((x for x in plan["stages"] if x["name"] == s["stage"]), None)
    if stage is None:
        raise ValueError(f"Unknown stage {s['stage']}; choose from {[x['name'] for x in plan['stages']]}")
    rows = stage["manifest"]["rows"]
    stage_cfg = {**stage["config"], "multi_noise": s}
    pools = noise_pools(rows)
    train = [r for r in rows if r["split"] == "train"]
    fixed = [r for r in rows if r["split"] != "train"]
    multi_eval = []
    for split, offset in (("val", 300), ("test", 301)):
        part = add_extra_noises([r for r in fixed if r["split"] == split], pools, s["eval_noise_counts"],
                                s["extra_gain_db"], random.Random(cfg["seed"] + offset))
        for r in part:
            r["split"] = f"{split}_multi"
            r["id"] = f"{split}_multi-" + r["id"].split("-", 1)[1]
        multi_eval.extend(part)
    trains = {"single": train,
              "multi": add_extra_noises(train, pools, s["train_noise_counts"], s["extra_gain_db"],
                                        random.Random(cfg["seed"] + 200))}
    manifests = {}
    for name in CONDITIONS:
        m = copy.deepcopy(stage["manifest"])
        m["config"] = stage_cfg
        m["rows"] = copy.deepcopy(trains[name]) + copy.deepcopy(fixed) + copy.deepcopy(multi_eval)
        m["notes"] = (f"multi_noise.py condition '{name}' on stage {s['stage']}. val/test are identical to "
                      "noise_expansion.py; val_multi/test_multi layer 2-3 noises. SNR = target vs whole noise bed.")
        manifests[name] = m
    train_src = {x for r in trains["multi"] for x in [r["target_source"], r["noise_source"]]
                 + [e["source"] for e in r.get("extra_noises", [])] if x}
    eval_src = {x for r in fixed + multi_eval for x in [r["target_source"], r["noise_source"]]
                + [e["source"] for e in r.get("extra_noises", [])] if x}
    if train_src & eval_src:
        raise AssertionError("Source recording shared between training and evaluation")
    info = {"stage": s["stage"], "settings": s, "val_fingerprint": plan["val_fingerprint"],
            "test_fingerprint": plan["test_fingerprint"],
            "val_multi_fingerprint": fingerprint([r for r in multi_eval if r["split"] == "val_multi"]),
            "test_multi_fingerprint": fingerprint([r for r in multi_eval if r["split"] == "test_multi"]),
            "cross_split_source_exclusions": plan["cross_split_source_exclusions"],
            "noise_count_distribution": {
                split: count_noises(rows_) for split, rows_ in
                (("train_multi", trains["multi"]), ("val_multi", [r for r in multi_eval if r["split"] == "val_multi"]),
                 ("test_multi", [r for r in multi_eval if r["split"] == "test_multi"]))}}
    return stage_cfg, manifests, info


def portable_fingerprint(rows):
    """fingerprint() with every audio path reduced to its file name (manifests hold absolute paths)."""
    def strip(v, key):
        if isinstance(v, dict):
            return {k: strip(x, k) for k, x in v.items()}
        if isinstance(v, list):
            return [strip(x, key) for x in v]
        return Path(str(v).replace("\\", "/")).name if key.endswith("path") and v else v
    return fingerprint([strip(r, "") for r in rows])


def count_noises(rows):
    out = {}
    for r in rows:
        if "noise_count" in r:
            key = f"{r['kind']}/{r['noise_count']}"
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def run(out, device):
    state = {"status": "running", "completed_steps": [], "started_unix": time.time()}
    save = lambda: write_json(out / "execution_status.json", state)

    def step(name, args):
        state["current_step"] = name
        save()
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        with (out / f"{name}.log").open("w", encoding="utf-8") as log:
            p = subprocess.Popen([sys.executable, "-u", *map(str, args)], cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if p.wait():
                raise RuntimeError(f"{name} failed; see {name}.log")
        state["completed_steps"].append({"name": name, "seconds": time.perf_counter() - started})
        save()

    dev = ["--device", device] if device else []
    try:
        # All training and validation tuning first; test splits are read only at the end.
        for c in CONDITIONS:
            run_dir = out / c
            step(f"{c}_train", ["train.py", "--config", out / "config.json", "--manifest", run_dir / "manifest.json",
                                "--mode", "mixed", "--record-metrics", "--out", run_dir / "mixed", *dev])
            for split, folder in (("val", "validation"), ("val_multi", "validation_multi")):
                step(f"{c}_{split}", ["evaluate.py", "--checkpoint", run_dir / "mixed/best.pt", "--manifest",
                                      run_dir / "manifest.json", "--split", split, "--tune",
                                      "--out", run_dir / "mixed" / folder, *dev])
        for c in CONDITIONS:
            run_dir = out / c
            for split, val_folder, folder in (("test", "validation", "test"), ("test_multi", "validation_multi", "test_multi")):
                step(f"{c}_{split}", ["evaluate.py", "--checkpoint", run_dir / "mixed/best.pt", "--manifest",
                                      run_dir / "manifest.json", "--split", split, "--thresholds",
                                      run_dir / "mixed" / val_folder / "thresholds.json",
                                      "--out", run_dir / "mixed" / folder, *dev])
        state.update(status="complete", finished_unix=time.time())
        state.pop("current_step", None)
        save()
    except BaseException as e:
        state.update(status="failed", error=str(e), finished_unix=time.time())
        save()
        raise
    summarize(out)


def summarize(out):
    """summary.csv / summary.json from report.json files (read-only)."""
    rows = []
    for c in CONDITIONS:
        for folder in ("validation", "validation_multi", "test", "test_multi"):
            path = out / c / "mixed" / folder / "report.json"
            if not path.is_file():
                continue
            rep = read_json(path)
            m = rep["ungated"]["metrics"]
            row = {"train": c, "eval": folder, "macro_f1": m["macro_f1"],
                   "negative_fp_windows": m["negative_false_positive_windows"],
                   "negative_windows": m["negative_windows"],
                   **{f"f1_{k}": v for k, v in zip(rep["classes"], m["f1"])},
                   "thresholds": rep["thresholds"]}
            for key, v in rep["ungated"]["conditions"].items():
                if key.startswith("noise_count/"):
                    row[f"macro_f1_{key.replace('/', '_')}"] = v.get("macro_f1")
            rows.append(row)
    write_json(out / "summary.json", {"rows": rows, "note": "One seed (config seed). Thresholds tuned on the "
                                      "matching validation split. Windows reuse target recordings; not independent events."})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['train']:>6} / {r['eval']:<17} macro-F1 {r['macro_f1']:.4f}  "
              f"no-target false alarms {r['negative_fp_windows']}/{r['negative_windows']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--esc50", default=str(ROOT / "data/ESC-50-master"))
    p.add_argument("--out", required=True, help="A NEW folder, e.g. runs/multi_noise_01")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--run-prepared", action="store_true")
    p.add_argument("--summarize", action="store_true", help="Only rebuild summary.csv/json from finished reports")
    p.add_argument("--device", default=None, help="Passed to train.py/evaluate.py (config default: cuda)")
    a = p.parse_args()
    out = Path(a.out).resolve()
    if a.summarize:
        return summarize(out)
    if a.run_prepared:
        read_json(out / "plan.json")
        if (out / "execution_status.json").exists():
            raise RuntimeError("This plan already started; use a new output folder")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Output folder must be new or empty")
        esc = Path(a.esc50)
        with (esc / "meta/esc50.csv").open(encoding="utf-8-sig", newline="") as f:
            meta = list(csv.DictReader(f))
        stage_cfg, manifests, info = build(meta, esc / "audio", config(a.config))
        write_json(out / "config.json", stage_cfg)
        for name, m in manifests.items():
            write_json(out / name / "manifest.json", m)
            info[f"{name}_manifest_sha256"] = sha256(out / name / "manifest.json")
        main = ROOT / "runs/noise_expansion_20261115" / info["stage"] / "manifest.json"
        if main.is_file():
            ref, new = read_json(main)["rows"], manifests["single"]["rows"]
            # Paths differ between PCs; compare contents with file names only.
            info["same_val_test_as_main_experiment"] = all(
                portable_fingerprint([r for r in ref if r["split"] == s]) ==
                portable_fingerprint([r for r in new if r["split"] == s]) for s in ("train", "val", "test"))
        write_json(out / "plan.json", info)
        for split, dist in info["noise_count_distribution"].items():
            print(split, dist)
        if "same_val_test_as_main_experiment" in info:
            print("Same train/val/test (single) as runs/noise_expansion_20261115:", info["same_val_test_as_main_experiment"])
    if not a.prepare_only:
        run(out, a.device)


if __name__ == "__main__":
    main()
