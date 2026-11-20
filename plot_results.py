"""Create simple presentation-ready comparison figures from evaluation reports."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True, help="Clean-training test report.json")
    p.add_argument("--mixed", required=True, help="Mixed-training test report.json")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    reports = [read_json(a.clean), read_json(a.mixed)]
    if reports[0]["classes"] != reports[1]["classes"] or reports[0]["manifest_sha256"] != reports[1]["manifest_sha256"]:
        raise ValueError("Cannot compare different class orders/evaluation datasets")
    if any(r["split"] != "test" for r in reports):
        raise ValueError("Use test reports for final figures")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for shift, r, label in zip((-0.18, 0.18), reports, ("Clean training", "Mixture training")):
        ax.bar(x + shift, r["ungated"]["metrics"]["f1"], width=0.36, label=label)
    ax.set(xticks=x, xticklabels=reports[0]["classes"], ylim=(0, 1), ylabel="F1", title="Per-class detection performance")
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend()
    fig.savefig(out / "class_f1.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, group in zip(axes, ("seen", "unseen")):
        for r, label in zip(reports, ("Clean training", "Mixture training")):
            conditions = r["ungated"]["conditions"]
            items = sorted([(float(k.rsplit("=", 1)[1]), v["macro_f1"]) for k, v in conditions.items()
                            if k.startswith(f"mixed/{group}/snr=")])
            if items:
                ax.plot(*zip(*items), marker="o", label=label)
        ax.set(xlabel="SNR (dB)", ylabel="Macro-F1", ylim=(0, 1), title=f"{group.capitalize()} noise classes")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.savefig(out / "snr_f1.png", dpi=180)
    plt.close(fig)
    print("Saved figures:", out)


if __name__ == "__main__":
    main()
