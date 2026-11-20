"""Nested non-target expansion with identical validation/test waveforms."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from common import ROOT, config, load_checkpoint, read_json, sha256, write_json
from prepare import build_manifest


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def prepare_plan(meta, audio, cfg, counts=(6, 18, 30, 43)):
    # Two source IDs in the full ESC-50 metadata occur across val/test folds
    # under different classes. Do not silently trust fold names over provenance.
    split_for_fold = {f:s for s in ("train","val","test") for f in cfg[f"{s}_folds"]}
    priority = {"train":0,"val":1,"test":2}
    owner = {}
    for r in meta:
        s = split_for_fold.get(int(r["fold"]))
        if s is not None and (r["src_file"] not in owner or priority[s] > priority[owner[r["src_file"]]]):
            owner[r["src_file"]] = s
    kept, excluded = [], []
    for r in meta:
        s = split_for_fold.get(int(r["fold"]))
        if s is not None and owner[r["src_file"]] != s:
            if r["category"] in cfg["classes"]:
                raise ValueError("Source conflict affects a target clip; resolve manually")
            excluded.append({"filename":r["filename"],"source":r["src_file"],"category":r["category"],
                             "removed_split":s,"kept_split":owner[r["src_file"]]})
        else:
            kept.append(r)
    meta = kept
    categories = sorted({r["category"] for r in meta})
    candidates = sorted(set(categories) - set(cfg["classes"]) - set(cfg["unseen_noise"]))
    initial = list(cfg["seen_noise"])
    additions = sorted(set(candidates) - set(initial))
    random.Random(cfg["seed"]).shuffle(additions)
    ordered = initial + additions
    if counts[0] != len(initial) or counts[-1] != len(ordered):
        raise ValueError("Stage counts must start at the original pool and end at all eligible non-target classes")
    full_cfg = copy.deepcopy(cfg)
    full_cfg["seen_noise"] = ordered
    full = build_manifest(meta, audio, full_cfg)
    fixed_eval = [copy.deepcopy(r) for r in full["rows"] if r["split"] != "train"]
    for row in fixed_eval:
        if row["noise_group"] == "seen":
            row["noise_group"] = "candidate_pool"
        elif row["noise_group"] == "unseen":
            row["noise_group"] = "heldout_types"
    target_rms = {r["target_path"]: r["target_rms"] for r in full["rows"]
                  if r["split"] == "train" and r["kind"] == "clean"}
    stages = []
    previous_paths = set()
    for count in counts:
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg["seen_noise"] = ordered[:count]
        m = build_manifest(meta, audio, stage_cfg)
        train = [r for r in m["rows"] if r["split"] == "train"]
        for row in train:
            if row["target_path"]:
                row["target_rms"] = target_rms[row["target_path"]]
        m["rows"] = train + copy.deepcopy(fixed_eval)
        m["notes"] = ("Nested noise-class expansion. Fixed validation/test waveforms include all 45 non-target classes. "
                      "candidate_pool is not necessarily trained at smaller stages; heldout_types are never fine-tuned. "
                      "Target amplitudes are fixed across stages. At most one target per window.")
        paths = {r["noise_path"] for r in train if r["noise_path"]}
        assert previous_paths <= paths
        previous_paths = paths
        train_sources = {r[k] for r in train for k in ("target_source", "noise_source") if r[k]}
        eval_sources = {r[k] for r in fixed_eval for k in ("target_source", "noise_source") if r[k]}
        assert not train_sources & eval_sources
        assert all(sum(r["labels"]) <= 1 for r in m["rows"])
        stages.append({"name": f"noise_{count:02d}", "noise_classes":count,
                       "noise_train_files":len(paths), "target_train_files":len(target_rms),
                       "train_windows":len(train), "config":stage_cfg, "manifest":m})
    val = [r for r in fixed_eval if r["split"] == "val"]
    test = [r for r in fixed_eval if r["split"] == "test"]
    return {"seed":cfg["seed"], "modes":["clean","mixed"], "stages":stages,
            "val_fingerprint":fingerprint(val), "test_fingerprint":fingerprint(test),
            "val_windows":len(val), "test_windows":len(test),
            "excluded_train_types":cfg["unseen_noise"],
            "cross_split_source_exclusions":excluded,
            "design":"Independent pretrained restart per condition; 10 head + 5 fine-tune epochs. Same epochs, unequal total updates as data grows. Noise diversity and negative/positive ratio both change; not a pure sample-count causal experiment.",
            "metrics":"Epoch curves use fixed threshold 0.5; tuned validation metrics separately labeled. Final test uses per-class thresholds selected on validation only."}


def run_experiment(out, plan):
    state = {"status":"running", "completed_steps":[], "started_unix":time.time()}
    def save():
        write_json(out / "execution_status.json", state)
    def step(name, args):
        state["current_step"] = name
        save()
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        with (out / f"{name}.log").open("w", encoding="utf-8") as log:
            p = subprocess.Popen([sys.executable,"-u",*map(str,args)],cwd=ROOT,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env=env)
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(line,end="",flush=True)
            code = p.wait()
        if code:
            raise RuntimeError(f"{name} failed ({code}); see log")
        state["completed_steps"].append({"name":name,"seconds":time.perf_counter()-started})
        save()
    try:
        # Complete all planned training and validation before inspecting test results.
        for stage in plan["stages"]:
            base = out / stage["name"]
            for mode in plan["modes"]:
                run = base / mode
                step(f"{stage['name']}_{mode}_train",["train.py","--config",base/"config.json",
                     "--manifest",base/"manifest.json","--mode",mode,"--record-metrics","--out",run])
                step(f"{stage['name']}_{mode}_val",["evaluate.py","--checkpoint",run/"best.pt",
                     "--manifest",base/"manifest.json","--split","val","--tune","--out",run/"validation"])
        for stage in plan["stages"]:
            base = out / stage["name"]
            for mode in plan["modes"]:
                run = base / mode
                step(f"{stage['name']}_{mode}_test",["evaluate.py","--checkpoint",run/"best.pt",
                     "--manifest",base/"manifest.json","--split","test",
                     "--thresholds",run/"validation/thresholds.json","--out",run/"test"])
        state.update(status="complete",finished_unix=time.time())
        state.pop("current_step",None)
        save()
    except BaseException as e:
        state.update(status="failed",error=str(e),finished_unix=time.time())
        save()
        raise
    post_process(out, state, step, save)


def post_process(out, state, step, save):
    """Figures, sigmoid analysis and presentation assets. Training results are already
    saved, so a failure here is recorded but does not mark the experiment failed."""
    baseline = ROOT / "runs/noise_expansion_20261115"
    jobs = [("figures", ["plot_noise_expansion.py", "--experiment", out]),
            ("sigmoid_analysis", ["analyze_sigmoid.py", "--experiment", out]),
            ("presentation", ["presentation_assets.py", "--experiment", out]
             + (["--baseline", baseline] if baseline.is_dir() and baseline.resolve() != out.resolve() else []))]
    state["post_processing"] = {}
    for name, args in jobs:
        try:
            step(f"post_{name}", args)
            state["post_processing"][name] = "ok"
        except Exception as e:
            state["post_processing"][name] = f"failed: {e}"
            print(f"WARNING post-processing {name} failed; rerun it manually. {e}", flush=True)
        state.pop("current_step", None)
        save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT/"config.json"))
    p.add_argument("--esc50", default=str(ROOT/"data/ESC-50-master"))
    p.add_argument("--out", required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--run-prepared", action="store_true")
    a=p.parse_args()
    out=Path(a.out).resolve()
    if a.run_prepared:
        plan=read_json(out/"plan.json")
        if (out/"execution_status.json").exists():
            raise RuntimeError("This plan already started; use a new output directory for a fresh experiment")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Output directory must be new or empty")
        out.mkdir(parents=True,exist_ok=True)
        esc=Path(a.esc50)
        with (esc/"meta/esc50.csv").open(encoding="utf-8-sig",newline="") as f:
            meta=list(csv.DictReader(f))
        prepared=prepare_plan(meta,esc/"audio",config(a.config))
        plan=copy.deepcopy(prepared)
        for stage in plan["stages"]:
            base=out/stage["name"]
            write_json(base/"config.json",stage.pop("config"))
            manifest=stage.pop("manifest")
            assert fingerprint([r for r in manifest["rows"] if r["split"]=="val"])==plan["val_fingerprint"]
            assert fingerprint([r for r in manifest["rows"] if r["split"]=="test"])==plan["test_fingerprint"]
            write_json(base/"manifest.json",manifest)
            stage["manifest_sha256"]=sha256(base/"manifest.json")
        write_json(out/"plan.json",plan)
        for s in plan["stages"]:
            print(s["name"],"target originals:",s["target_train_files"],"noise originals:",s["noise_train_files"],"train windows:",s["train_windows"])
        print("Fixed validation/test windows:",plan["val_windows"],plan["test_windows"])
    if not a.prepare_only:
        run_experiment(out,plan)


if __name__=="__main__":
    main()
