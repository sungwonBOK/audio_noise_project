"""Layer-by-layer intuition for the pretrained Cnn14 backbone (analysis only).

Uses only the AudioSet-pretrained Cnn14 weights (no trained 5-class checkpoint).
Does not modify any training/evaluation code or past results.

Outputs (in --out):
  receptive_field.csv          theoretical receptive field per block
  layer_summary.csv            per-layer channel purity, 1-NN accuracy, silhouette
  top_channels.json            example channels and the clips that excite them most
  fig_activation_maps.png      channel-mean activation per block for a few clips
  fig_top_clips_<layer>.png    log-mel of top-activating clips for example channels
  fig_tsne.png                 t-SNE of pooled features per layer (5 target classes)
  fig_layer_trend.png          purity / 1-NN accuracy vs layer
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_utils import load_audio  # noqa: E402
from model import SoundDetector  # noqa: E402

SR, SECONDS = 32000, 5.0
TARGETS = ["car_horn", "siren", "door_wood_knock", "crying_baby", "glass_breaking"]
MAJOR = ["animals", "natural", "human", "domestic", "urban"]
LAYERS = [f"block{i}" for i in range(1, 7)] + ["fc1"]
TOPK = 10


def receptive_field():
    """Each ConvBlock = two 3x3 convs (stride 1) + 2x2 avg pool (block6: no pool)."""
    rows, rf, jump = [], 1, 1
    for i in range(1, 7):
        rf += 2 * jump
        rf += 2 * jump
        if i < 6:
            rf += 1 * jump
            jump *= 2
        rows.append({"layer": f"block{i}", "rf_frames": rf, "rf_seconds": round(rf * 0.01, 3),
                     "rf_mel_bins": min(rf, 64), "covers_all_64_mel_bins": rf >= 64})
    return rows


def select_clips(meta_path, per_class, seed):
    rng = np.random.default_rng(seed)
    with open(meta_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_class = {}
    for r in rows:
        by_class.setdefault(r["category"], []).append(r)
    chosen = []
    for cat in sorted(by_class):
        items = sorted(by_class[cat], key=lambda r: r["filename"])
        # balanced: same count for every class, so channel purity is not biased to any class
        idx = rng.choice(len(items), size=min(per_class, len(items)), replace=False)
        chosen += [items[i] for i in sorted(idx)]
    return chosen


@torch.inference_mode()
def forward_layers(model, wave):
    """Same path as model.SoundDetector.forward, but returns every block output."""
    bb = model.backbone
    x = bb.spectrogram_extractor(wave.float())
    x = bb.logmel_extractor(x)
    logmel = x[:, 0]                       # [B, T, F]
    x = x.transpose(1, 3)
    x = bb.bn0(x)
    x = x.transpose(1, 3)
    outs = {}
    for i in range(1, 7):
        pool = (2, 2) if i < 6 else (1, 1)
        x = getattr(bb, f"conv_block{i}")(x, pool_size=pool, pool_type="avg")
        outs[f"block{i}"] = x              # [B, C, T, F]
    z = x.mean(dim=3)
    z = z.max(dim=2).values + z.mean(dim=2)
    outs["fc1"] = torch.relu(bb.fc1(z))     # [B, 2048]
    return logmel, outs


def pooled(feat):
    """Clip-level vector per channel, using Cnn14's own final pooling (freq mean, time max+mean)."""
    if feat.dim() == 2:
        return feat
    z = feat.mean(dim=3)
    return z.max(dim=2).values + z.mean(dim=2)


def peak(feat):
    if feat.dim() == 2:
        return feat
    return feat.amax(dim=(2, 3))


def channel_purity(peaks, labels, k=TOPK):
    """For each channel: share of its top-k clips that belong to the most common class."""
    order = np.argsort(-peaks, axis=0)[:k]          # [k, C]
    top_labels = labels[order]                        # [k, C]
    pur = np.array([Counter(top_labels[:, c]).most_common(1)[0][1] / k
                    for c in range(peaks.shape[1])])
    alive = peaks.max(axis=0) > 1e-6
    return pur, alive


def one_nn_accuracy(x, labels):
    x = x - x.mean(axis=0)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    return float((labels[sim.argmax(axis=1)] == labels).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esc50", default=str(ROOT / "data" / "ESC-50-master"))
    p.add_argument("--weights", default=str(ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"))
    p.add_argument("--out", default=str(ROOT / "runs" / "layer_intuition_01"))
    p.add_argument("--per-class", type=int, default=20, help="clips per class (all 50 classes, balanced; max 40)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="debug: only first N clips")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output folder not empty, choose a new --out: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    esc = Path(a.esc50)
    clips = select_clips(esc / "meta" / "esc50.csv", a.per_class, a.seed)
    if a.limit:
        clips = clips[:a.limit]
    cats = [c["category"] for c in clips]
    class_names = sorted(set(cats))
    labels = np.array([class_names.index(c) for c in cats])
    majors = np.array([MAJOR[int(c["target"]) // 10] for c in clips])
    print(f"clips: {len(clips)} ({len(class_names)} classes), device: cpu", flush=True)

    model = SoundDetector()
    model.load_pretrained(a.weights)
    model.eval()

    peaks = {l: [] for l in LAYERS}
    pools = {l: [] for l in LAYERS}
    t0 = time.time()
    for s in range(0, len(clips), a.batch):
        batch = clips[s:s + a.batch]
        wave = torch.from_numpy(np.stack([load_audio(esc / "audio" / c["filename"], SR, SECONDS) for c in batch]))
        _, outs = forward_layers(model, wave)
        for l in LAYERS:
            peaks[l].append(peak(outs[l]).numpy())
            pools[l].append(pooled(outs[l]).numpy())
        done = s + len(batch)
        el = time.time() - t0
        print(f"  {done}/{len(clips)} clips, {el:.0f}s elapsed, ~{el / done * (len(clips) - done):.0f}s left", flush=True)
    peaks = {l: np.concatenate(v) for l, v in peaks.items()}
    pools = {l: np.concatenate(v) for l, v in pools.items()}
    forward_seconds = time.time() - t0

    # ---------- per-layer summary ----------
    rng = np.random.default_rng(a.seed)
    summary, example_channels = [], {}
    tgt_mask = np.isin(np.array(cats), TARGETS)
    for l in LAYERS:
        pur, alive = channel_purity(peaks[l], labels)
        mpur, _ = channel_purity(peaks[l], majors)
        shuffled, _ = channel_purity(peaks[l], rng.permutation(labels))
        row = {
            "layer": l,
            "channels": int(peaks[l].shape[1]),
            "active_channels": int(alive.sum()),
            "class_purity_mean": round(float(pur[alive].mean()), 4),
            "class_purity_shuffled_baseline": round(float(shuffled[alive].mean()), 4),
            "share_channels_purity_ge_0.5": round(float((pur[alive] >= 0.5).mean()), 4),
            "major_category_purity_mean": round(float(mpur[alive].mean()), 4),
            "one_nn_acc_all_classes": round(one_nn_accuracy(pools[l], labels), 4),
            "one_nn_acc_5_targets": round(one_nn_accuracy(pools[l][tgt_mask], labels[tgt_mask]), 4),
            "silhouette_5_targets": round(float(silhouette_score(pools[l][tgt_mask], labels[tgt_mask], metric="cosine")), 4),
        }
        summary.append(row)
        print(row, flush=True)
        # example channels: most class-specific, and a median (generic) one
        alive_idx = np.where(alive)[0]
        ranked = alive_idx[np.argsort(-pur[alive_idx], kind="stable")]
        picks = {"most_specific": [int(c) for c in ranked[:2]],
                 "median": [int(ranked[len(ranked) // 2])]}
        example_channels[l] = {}
        for kind, chans in picks.items():
            for c in chans:
                top = np.argsort(-peaks[l][:, c])[:TOPK]
                example_channels[l][f"{kind}_ch{c}"] = {
                    "channel": c, "purity": float(pur[c]),
                    "top_clips": [{"file": clips[i]["filename"], "class": cats[i],
                                   "peak": round(float(peaks[l][i, c]), 4)} for i in top]}

    rf = receptive_field()
    with open(out / "receptive_field.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rf[0]))
        w.writeheader(); w.writerows(rf)
    with open(out / "layer_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    (out / "top_channels.json").write_text(json.dumps(example_channels, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- figure: trend ----------
    xs = np.arange(len(LAYERS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, lab in [("class_purity_mean", "channel top-10 class purity"),
                     ("class_purity_shuffled_baseline", "purity, shuffled labels (chance)"),
                     ("one_nn_acc_all_classes", f"1-NN accuracy ({len(class_names)} classes)"),
                     ("one_nn_acc_5_targets", "1-NN accuracy (5 targets)")]:
        ax.plot(xs, [r[key] for r in summary], marker="o", label=lab)
    ax.set_xticks(xs, LAYERS); ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.grid(alpha=.3)
    ax.set_title("Pretrained Cnn14: how class-specific each layer is"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_layer_trend.png", dpi=130); plt.close(fig)

    # ---------- figure: t-SNE (5 targets) ----------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    tl = labels[tgt_mask]
    for ax, l in zip(axes.flat, LAYERS):
        z = pools[l][tgt_mask]
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        emb = TSNE(n_components=2, perplexity=min(30, len(z) // 4), init="pca",
                   random_state=a.seed).fit_transform(z)
        for name in TARGETS:
            if name in class_names:
                m = tl == class_names.index(name)
                ax.scatter(emb[m, 0], emb[m, 1], s=14, label=name)
        ax.set_title(l); ax.set_xticks([]); ax.set_yticks([])
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"t-SNE of pooled features, 5 target classes ({a.per_class} clips each)")
    fig.tight_layout(); fig.savefig(out / "fig_tsne.png", dpi=110); plt.close(fig)

    # ---------- figure: activation maps for a few clips ----------
    demo = []
    for name in ["siren", "crying_baby", "door_wood_knock", "glass_breaking"]:
        idx = [i for i, c in enumerate(cats) if c == name]
        if idx:
            demo.append(idx[0])
    wave = torch.from_numpy(np.stack([load_audio(esc / "audio" / clips[i]["filename"], SR, SECONDS) for i in demo]))
    logmel, outs = forward_layers(model, wave)
    fig, axes = plt.subplots(len(demo), 7, figsize=(21, 2.6 * len(demo)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(demo):
        axes[r, 0].imshow(logmel[r].T.numpy(), origin="lower", aspect="auto", cmap="magma")
        axes[r, 0].set_title(f"{cats[i]}: log-mel" if r == 0 else cats[i], fontsize=9)
        axes[r, 0].set_ylabel(cats[i], fontsize=9)
        for k in range(1, 7):
            m = outs[f"block{k}"][r].mean(dim=0).T.numpy()   # [F, T]
            axes[r, k].imshow(m, origin="lower", aspect="auto", cmap="viridis")
            if r == 0:
                axes[r, k].set_title(f"block{k}  ({m.shape[1]}x{m.shape[0]})", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Channel-mean activation per block (x = time, y = mel frequency)")
    fig.tight_layout(); fig.savefig(out / "fig_activation_maps.png", dpi=100); plt.close(fig)

    # ---------- figure: top-activating clips for example channels ----------
    name_to_idx = {c["filename"]: i for i, c in enumerate(clips)}
    for l in ["block1", "block3", "block6"]:
        chans = list(example_channels[l].items())[:3]
        files = [[t["file"] for t in info["top_clips"][:4]] for _, info in chans]
        flat = [f for row in files for f in row]
        wave = torch.from_numpy(np.stack([load_audio(esc / "audio" / f, SR, SECONDS) for f in flat]))
        lm, _ = forward_layers(model, wave)
        fig, axes = plt.subplots(len(chans), 4, figsize=(14, 2.6 * len(chans)))
        axes = np.atleast_2d(axes)
        for r, (key, info) in enumerate(chans):
            for c in range(4):
                ax = axes[r, c]
                ax.imshow(lm[r * 4 + c].T.numpy(), origin="lower", aspect="auto", cmap="magma")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{cats[name_to_idx[files[r][c]]]}", fontsize=9)
            axes[r, 0].set_ylabel(f"ch{info['channel']}\npurity {info['purity']:.1f}", fontsize=9)
        fig.suptitle(f"{l}: top-4 clips that most excite example channels (rows)")
        fig.tight_layout(); fig.savefig(out / f"fig_top_clips_{l}.png", dpi=100); plt.close(fig)

    (out / "run_info.json").write_text(json.dumps({
        "clips": len(clips), "classes": len(class_names), "per_class": a.per_class,
        "seed": a.seed, "device": "cpu", "forward_seconds": round(forward_seconds, 1),
        "weights": str(a.weights), "note": "AudioSet-pretrained Cnn14 only; no project-trained head used."},
        indent=2), encoding="utf-8")
    print("done:", out)


if __name__ == "__main__":
    main()
