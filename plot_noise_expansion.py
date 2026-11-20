"""Summarize completed expansion runs; export epoch and data-size curves."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_checkpoint, read_json, write_json, sha256
from noise_expansion import fingerprint


def summarize(root):
    plan=read_json(root/"plan.json")
    if read_json(root/"execution_status.json")["status"]!="complete":
        raise RuntimeError("Experiments are not complete")
    records=[]
    histories={}
    for stage in plan["stages"]:
        base=root/stage["name"]
        m=read_json(base/"manifest.json")
        for split in ("val","test"):
            assert fingerprint([r for r in m["rows"] if r["split"]==split])==plan[f"{split}_fingerprint"]
        for mode in plan["modes"]:
            run=base/mode
            with (run/"history.csv").open(encoding="utf-8-sig",newline="") as f:
                history=list(csv.DictReader(f))
            assert len(history)==15
            histories[(stage["name"],mode)]=history
            ckpt=load_checkpoint(run/"best.pt")
            selected=next(r for r in history if r["stage"]==ckpt["stage"] and int(r["epoch"])==ckpt["epoch"])
            val=read_json(run/"validation/report.json")
            test=read_json(run/"test/report.json")
            assert val["checkpoint_sha256"]==test["checkpoint_sha256"]==sha256(run/"best.pt")
            training=read_json(run/"training_summary.json")
            row={"stage":stage["name"],"noise_classes":stage["noise_classes"],
                 "target_train_files":stage["target_train_files"],"noise_train_files":stage["noise_train_files"],
                 "train_windows":stage["train_windows"],"mode":mode,"epochs":len(history),
                 "total_optimizer_updates":int(history[-1]["cumulative_optimizer_updates"]),
                 "total_train_presentations":int(history[-1]["cumulative_train_presentations"]),
                 "selected_stage":ckpt["stage"],"selected_stage_epoch":ckpt["epoch"],
                 "best_val_bce":ckpt["val_bce"],"training_wall_seconds":training["training_wall_seconds"],
                 "val_fixed_macro_precision":float(selected["val_macro_precision"]),
                 "val_fixed_macro_recall":float(selected["val_macro_recall"]),
                 "val_fixed_macro_f1":float(selected["val_macro_f1"]),
                 "val_fixed_negative_fpr":float(selected["val_negative_fpr"])}
            for split,report in (("val_tuned",val),("test",test)):
                metric=report["ungated"]["metrics"]
                row.update({f"{split}_macro_precision":float(np.mean(metric["precision"])),
                            f"{split}_macro_recall":float(np.mean(metric["recall"])),
                            f"{split}_macro_f1":metric["macro_f1"],
                            f"{split}_negative_fpr":metric["negative_window_false_positive_rate"],
                            f"{split}_negative_fp_windows":metric["negative_false_positive_windows"],
                            f"{split}_negative_windows":metric["negative_windows"]})
            row["test_class_f1"]=test["ungated"]["metrics"]["f1"]
            row["test_class_recall"]=test["ungated"]["metrics"]["recall"]
            row["test_conditions"]=test["ungated"]["conditions"]
            records.append(row)
    summary={"plan":plan,"records":records,"interpretation":"One fixed seed and split. Noise diversity, number of negative windows, class balance and total updates vary together. Validation curves at fixed 0.5 are distinct from validation-tuned test metrics. No RMS gate used."}
    write_json(root/"summary.json",summary)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in records]
    with (root/"data_size_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    all_epochs=[]
    for stage in plan["stages"]:
        for mode in plan["modes"]:
            for h in histories[(stage["name"],mode)]:
                all_epochs.append({"noise_classes":stage["noise_classes"],"noise_train_files":stage["noise_train_files"],"mode":mode,**h})
    with (root/"all_epoch_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(all_epochs[0])); w.writeheader(); w.writerows(all_epochs)
    return summary,histories


def plot(root,summary,histories):
    plan=summary["plan"]
    out=root/"figures"
    out.mkdir(exist_ok=True)
    labels={"clean":"Clean training","mixed":"Mixture training"}
    colors={"clean":"#2878b5","mixed":"#e87523"}
    def epoch_plot(key,title,filename,loss=False):
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout="constrained",sharex=True,sharey=True)
        for ax,stage in zip(axes.flat,plan["stages"]):
            for mode in plan["modes"]:
                hs=histories[(stage["name"],mode)]
                x=[int(h["total_epoch"]) for h in hs]
                if loss:
                    ax.plot(x,[float(h["train_bce"]) for h in hs],color=colors[mode],linestyle="--",label=f"{labels[mode]}: train")
                    ax.plot(x,[float(h["val_bce"]) for h in hs],color=colors[mode],label=f"{labels[mode]}: val")
                else:
                    ax.plot(x,[float(h[key]) for h in hs],color=colors[mode],marker=".",label=labels[mode])
                ax.axvline(10.5,color="gray",linestyle=":",linewidth=0.7) if mode=="clean" else None
            ax.set(title=f"{stage['noise_classes']} noise classes / {stage['noise_train_files']} noise clips",xlabel="Epoch (head 1-10; fine-tune 11-15)",ylabel="BCE" if loss else title)
            if not loss: ax.set_ylim(0,1.02)
            ax.set_xticks([1,5,10,11,15]); ax.grid(alpha=0.2); ax.legend(fontsize=7)
        fig.suptitle(title+("" if loss else " | validation, threshold 0.5"),fontsize=13)
        fig.savefig(out/filename,dpi=170); plt.close(fig)
    epoch_plot(None,"Training and validation loss","epoch_loss.png",loss=True)
    metrics=(("val_macro_precision","Macro precision","precision"),("val_macro_recall","Macro recall","recall"),
             ("val_macro_f1","Macro-F1","f1"),("val_negative_fpr","Negative-window false positive rate","negative_fpr"))
    for key,title,name in metrics:
        epoch_plot(key,title,f"epoch_{name}.png")
    cfg=read_json(root/plan["stages"][0]["name"]/"config.json")
    for name in cfg["classes"]:
        for metric in ("precision","recall","f1"):
            epoch_plot(f"val_{name}_{metric}",f"{name}: {metric}",f"epoch_{name}_{metric}.png")
    records=summary["records"]
    def size_plot(key,title,filename,bounded=True):
        fig,ax=plt.subplots(figsize=(8,4.7),layout="constrained")
        for mode in plan["modes"]:
            rs=[r for r in records if r["mode"]==mode]
            ax.plot([r["noise_train_files"] for r in rs],[r[key] for r in rs],marker="o",label=labels[mode],color=colors[mode])
        ax.set(xlabel="Non-target training clips (120 target clips fixed)",ylabel=title,title=title)
        if bounded: ax.set_ylim(0,1.02)
        ax.set_xticks([s["noise_train_files"] for s in plan["stages"]]); ax.grid(alpha=0.2); ax.legend()
        fig.savefig(out/filename,dpi=170); plt.close(fig)
    for metric,title in (("macro_precision","Macro precision"),("macro_recall","Macro recall"),
                         ("macro_f1","Macro-F1"),("negative_fpr","Negative-window false positive rate")):
        size_plot(f"val_fixed_{metric}",f"Validation {title} (threshold 0.5)",f"data_size_val_{metric}.png")
        size_plot(f"test_{metric}",f"Test {title} (validation-tuned thresholds)",f"data_size_test_{metric}.png")
    size_plot("training_wall_seconds","Training wall time (seconds)","data_size_training_time.png",bounded=False)
    print("Saved",len(list(out.glob('*.png'))),"figures to",out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",required=True)
    a=p.parse_args()
    root=Path(a.experiment).resolve()
    summary,histories=summarize(root)
    plot(root,summary,histories)
    for r in summary["records"]:
        print(r["noise_classes"],r["mode"],"test Macro-F1",round(r["test_macro_f1"],6),"FPR",round(r["test_negative_fpr"],6))


if __name__=="__main__":
    main()
