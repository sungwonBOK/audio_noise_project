"""Train a new head, then optionally fine-tune block6/fc1 from the best head."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from audio_utils import render_recipe
from augment import AugmentedRecipeDataset, augment_settings
from common import ROOT, config, device_for, read_manifest, seed_all, sha256, write_json, load_checkpoint
from model import SoundDetector
from metrics import scores, tune_thresholds


class RecipeDataset(Dataset):
    def __init__(self, rows, cfg, mode="mixed"):
        self.rows, self.cfg, self.mode = rows, cfg, mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return torch.from_numpy(render_recipe(row, self.cfg, self.mode)), torch.tensor(row["labels"], dtype=torch.float32)


def masked_bce(logits, y):
    """Mean BCE over known labels. -1 (uncertain pseudo label) contributes nothing.
    Without -1 labels this equals nn.BCEWithLogitsLoss()."""
    loss = F.binary_cross_entropy_with_logits(logits.float(), y.clamp(min=0), reduction="none")
    mask = (y >= 0).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def epoch(model, loader, device, loss_fn, optimizer=None, scaler=None, amp=False, collect=False):
    model.train(optimizer is not None)
    # Average over KNOWN label cells (not rows), so windows with -1 cells do not skew the
    # validation BCE used for checkpoint selection. Without -1 this is the usual mean BCE.
    total, count, rows = 0.0, 0, 0
    targets, probabilities = [], []
    for wave, y in loader:
        wave = wave.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(wave)
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check audio, learning rate and AMP settings")
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(optimizer)
                scaler.update()
        known = int((y >= 0).sum())
        total += float(loss.detach()) * known
        count += known
        rows += len(y)
        if collect:
            targets.append(y.detach().cpu().numpy())
            probabilities.append(logits.detach().float().sigmoid().cpu().numpy())
    if rows == 0:
        raise ValueError("Empty dataloader")
    if count == 0:
        raise ValueError("Every label in this dataloader is uncertain (-1)")
    if collect:
        return total / count, np.concatenate(targets), np.concatenate(probabilities)
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--pretrained", default="checkpoints/Cnn14_mAP=0.431.pth")
    p.add_argument("--mode", choices=["clean", "mixed"], required=True)
    p.add_argument("--out", required=True, help="A NEW run directory")
    p.add_argument("--device", default=None)
    p.add_argument("--record-metrics", action="store_true", help="Record per-epoch FP32 validation metrics at threshold 0.5 and validation-tuned thresholds")
    a = p.parse_args()
    cfg = config(a.config)
    device = device_for(a.device or cfg["device"])
    seed_all(cfg["seed"])
    manifest = read_manifest(a.manifest, cfg)
    # ESC-50 recipes have one target; windows of long recordings may hold several (teacher labels).
    if any(r.get("source") != "long" and sum(r["labels"]) > 1 for r in manifest["rows"]):
        raise ValueError("Overlapping TARGET classes are outside this project's training/evaluation scope")
    augment = augment_settings(cfg)
    if augment["enabled"] and any(r.get("extra_noises") for r in manifest["rows"]):
        raise ValueError("augment.py re-mixes a single noise and would drop extra_noises; disable augment")
    loaders = {}
    for split in ("train", "val"):
        rows = [r for r in manifest["rows"] if r["split"] == split]
        # Validation is identical for clean/mixed runs, including noisy examples.
        # Only the train split is re-mixed on the fly; validation stays a fixed recipe.
        if split == "train" and augment["enabled"]:
            ds = AugmentedRecipeDataset(rows, cfg, a.mode)
        else:
            ds = RecipeDataset(rows, cfg, a.mode if split == "train" else "mixed")
        loaders[split] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=split == "train",
                                    num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                                    generator=torch.Generator().manual_seed(cfg["seed"]))
    model = SoundDetector(spec_augment=augment["spec_augment"])
    model.load_pretrained(a.pretrained)
    model.to(device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    provenance = {"manifest_sha256": sha256(a.manifest), "pretrained_sha256": sha256(a.pretrained),
                  "torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
                  "device": str(device), "mode": a.mode,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "augment": augment, "train_items_per_epoch": len(loaders["train"].dataset),
                  "train_manifest_rows": sum(r["split"] == "train" for r in manifest["rows"]),
                  "long_labels": manifest.get("long_labels"),
                  "long_rows": {s: sum(r["split"] == s and r.get("source") == "long" for r in manifest["rows"])
                                for s in ("train", "val", "test")}}
    write_json(out / "provenance.json", provenance)
    amp = cfg["amp"] and device.type == "cuda"
    loss_fn = masked_bce
    global_best, history = float("inf"), []
    updates, presentations = 0, 0
    start = time.perf_counter()
    for stage, epochs in (("head", cfg["head_epochs"]), ("finetune", cfg["finetune_epochs"])):
        if epochs == 0:
            continue
        if stage == "finetune":
            # Preserve the best trained head; do NOT initialize it again.
            model.load_state_dict(load_checkpoint(out / "head_best.pt")["model"])
        model.configure_stage(stage)
        groups = [{"params": model.head.parameters(), "lr": cfg["head_lr"] if stage == "head" else cfg["finetune_head_lr"]}]
        if stage == "finetune":
            groups.append({"params": [p for p in model.backbone.parameters() if p.requires_grad],
                           "lr": cfg["backbone_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        stage_best = float("inf")
        for index in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            if hasattr(loaders["train"].dataset, "set_epoch"):
                # New noise/SNR/gain/onset every epoch, reproducible from (seed, epoch, item).
                loaders["train"].dataset.set_epoch(len(history))
            train_loss = epoch(model, loaders["train"], device, loss_fn, optimizer, scaler, amp)
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            if a.record_metrics:
                val_loss, val_y, val_prob = epoch(model, loaders["val"], device, loss_fn, amp=False, collect=True)
            else:
                val_loss = epoch(model, loaders["val"], device, loss_fn, amp=amp)
            validation_seconds = time.perf_counter() - val_start
            record = dict(stage=stage, epoch=index, train_bce=train_loss, val_bce=val_loss)
            updates += len(loaders["train"])
            presentations += len(loaders["train"].dataset)
            if a.record_metrics:
                fixed = scores(val_y, val_prob, [0.5]*5)
                thresholds = tune_thresholds(val_y, val_prob)
                tuned = scores(val_y, val_prob, thresholds)
                record.update(total_epoch=len(history)+1, train_windows=len(loaders["train"].dataset),
                              cumulative_optimizer_updates=updates, cumulative_train_presentations=presentations,
                              train_epoch_seconds=train_seconds, val_epoch_seconds=validation_seconds,
                              val_macro_precision=float(np.mean(fixed["precision"])),
                              val_macro_recall=float(np.mean(fixed["recall"])), val_macro_f1=fixed["macro_f1"],
                              val_negative_fpr=fixed["negative_window_false_positive_rate"],
                              val_tuned_macro_f1=tuned["macro_f1"],
                              val_tuned_negative_fpr=tuned["negative_window_false_positive_rate"])
                for i, name in enumerate(cfg["classes"]):
                    for metric in ("precision", "recall", "f1"):
                        record[f"val_{name}_{metric}"] = fixed[metric][i]
                with (out / "epoch_metrics.jsonl").open("a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"total_epoch":len(history)+1, "stage":stage, "epoch":index,
                                       "fixed_thresholds":[0.5]*5, "fixed_metrics":fixed,
                                       "validation_tuned_thresholds":thresholds, "tuned_metrics":tuned}, allow_nan=False)+"\n")
            history.append(record)
            print(record, flush=True)
            checkpoint = {"model": model.state_dict(), "config": cfg, "mode": a.mode,
                          "stage": stage, "epoch": index, "val_bce": val_loss,
                          "provenance": provenance}
            if val_loss < stage_best:
                stage_best = val_loss
                torch.save(checkpoint, out / f"{stage}_best.pt")
            if val_loss < global_best:
                global_best = val_loss
                torch.save(checkpoint, out / "best.pt")
            with (out / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(record))
                writer.writeheader()
                writer.writerows(history)
    write_json(out / "training_summary.json", {"best_val_bce": global_best,
               "training_wall_seconds": time.perf_counter() - start,
               "selection": "Minimum validation BCE across both stages; no test data used."})
    print("Selected checkpoint:", out / "best.pt")


if __name__ == "__main__":
    main()
