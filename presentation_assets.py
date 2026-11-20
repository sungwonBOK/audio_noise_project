"""발표용 결과 자료 생성 (Presentation assets from a finished experiment).

학습이 끝난 실험 폴더(runs/<실험>)의 저장 결과만 읽어서 <실험>/presentation/ 에
아래 자료를 만듭니다. 체크포인트(.pt)나 GPU가 필요 없으므로, 결과 폴더만 받은
PC에서도 ESC-50 원본 음원만 있으면 다시 만들 수 있습니다.

- audio/      소리별·SNR별(깨끗함, 10, 5, 0 dB) 예시 WAV, 방해음만 들은 WAV,
              놓침·오검출 사례 WAV, 학습 증강 예시 WAV
- figures/    판정 카드(스펙트로그램 + 5개 소리 점수 + 기준값), SNR별 재현율,
              혼동표, 오검출을 일으킨 방해음, 학습 곡선, 기존 실험과의 비교,
              학습 증강 설명 그림
- index.html  위 자료를 한 화면에서 듣고 볼 수 있는 페이지
- summary.json, examples.csv  수치 요약과 예시 목록

모델 선택은 검증 Macro-F1 기준입니다(평가 점수로 고르지 않음).
"""
import argparse
import ast
import copy
import csv
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_utils import load_audio, rms, limit_peak, render_recipe
from common import ROOT, read_json, write_json
from metrics import scores
from plot_fonts import korean_font_family

SNR_ORDER = ["clean", 10, 5, 0]
SNR_LABEL = {"clean": "깨끗한 소리", 10: "SNR 10 dB", 5: "SNR 5 dB", 0: "SNR 0 dB"}
GROUP_LABEL = {"candidate_pool": "학습 후보 방해음", "heldout_types": "학습하지 않은 방해음"}


# ----------------------------------------------------------------------------- loading
def setup_fonts():
    plt.rcParams.update({"font.family": korean_font_family(), "font.size": 9, "axes.unicode_minus": False,
                         "axes.titlesize": 10, "figure.dpi": 100})


def read_predictions(path):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["id"]] = (np.asarray(ast.literal_eval(r["labels"]), int),
                             np.asarray(ast.literal_eval(r["scores"]), float))
    return rows


def read_history(path):
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{k: (float(v) if k.endswith("bce") else v) for k, v in r.items()} for r in csv.DictReader(f)]


def find_models(experiment):
    models = {}
    for stage in sorted(p for p in experiment.glob("noise_*") if p.is_dir()):
        for mode in ("clean", "mixed"):
            run = stage / mode
            if (run / "test/predictions.csv").is_file() and (run / "validation/thresholds.json").is_file():
                models[f"{stage.name}/{mode}"] = run
    if not models:
        raise FileNotFoundError(f"No finished models (test/predictions.csv) under {experiment}")
    return models


class Run:
    def __init__(self, experiment, name, run):
        self.name, self.dir = name, run
        self.stage_dir = run.parent
        self.cfg = read_json(self.stage_dir / "config.json")
        self.classes, self.names = self.cfg["classes"], self.cfg["class_names_ko"]
        self.thresholds = np.asarray(read_json(run / "validation/thresholds.json")["thresholds"], float)
        self.val_macro_f1 = read_json(run / "validation/report.json")["ungated"]["metrics"]["macro_f1"]
        self.pred = read_predictions(run / "test/predictions.csv")
        self.history = read_history(run / "history.csv")
        manifest = self.stage_dir / "manifest.json"
        self.rows = read_json(manifest)["rows"] if manifest.is_file() else []
        self.test_rows = [r for r in self.rows if r["split"] == "test"]
        self.augmented = bool((self.cfg.get("augment") or {}).get("enabled"))

    def arrays(self, ids=None):
        ids = ids if ids is not None else list(self.pred)
        y = np.stack([self.pred[i][0] for i in ids])
        p = np.stack([self.pred[i][1] for i in ids])
        return y, p


def load_runs(experiment):
    return {name: Run(experiment, name, run) for name, run in find_models(experiment).items()}


def make_resolver(audio_root):
    root = Path(audio_root) if audio_root else None

    def resolve(path):
        if not path or Path(path).is_file():
            return path
        if root is not None:
            candidate = root / path.replace("\\", "/").split("/")[-1]
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"Audio not found: {path}. Pass --audio-root <ESC-50 audio folder>.")
    return resolve


# ----------------------------------------------------------------------------- metrics
def judge(labels, score, thresholds, names):
    pred = score >= thresholds
    truth = labels.astype(bool)
    detected = ", ".join(names[i] for i in np.flatnonzero(pred)) or "목표음 없음"
    if (pred == truth).all():
        verdict = "정답"
    elif truth.any() and not pred[truth.argmax()]:
        verdict = "놓침" if not pred.any() else "다른 소리로 오인"
    else:
        verdict = "오검출"
    return verdict, detected


def macro_ap(y, p):
    from sklearn.metrics import average_precision_score
    return float(np.mean([average_precision_score(y[:, i], p[:, i]) for i in range(y.shape[1])]))


def condition_of(row):
    return "clean" if row["kind"] == "clean" else row["snr_db"]


def recall_table(run, group=None):
    """recall[class][condition] over positive test windows (optionally one noise group)."""
    table = {}
    for c in range(5):
        table[c] = {}
        for cond in SNR_ORDER:
            ids = [r["id"] for r in run.test_rows if r["labels"][c] == 1 and condition_of(r) == cond
                   and (cond == "clean" or group is None or r["noise_group"] == group)]
            if ids:
                _, p = run.arrays(ids)
                table[c][cond] = float(np.mean(p[:, c] >= run.thresholds[c]))
    return table


def cluster_bootstrap_f1(runs, rows, n=1000, seed=0):
    """Macro-F1 CI resampling source recordings (windows of one recording move together)."""
    keys = [r["target_path"] or r["noise_path"] or r["id"] for r in rows]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    clusters = list(groups.values())
    data = []
    for run in runs:
        y, p = run.arrays([r["id"] for r in rows])
        data.append((y.astype(bool), p >= run.thresholds))
    rng = np.random.default_rng(seed)
    out = [[] for _ in runs]
    for _ in range(n):
        pick = np.concatenate([clusters[j] for j in rng.integers(len(clusters), size=len(clusters))])
        for k, (y, pr) in enumerate(data):
            yy, pp = y[pick], pr[pick]
            tp, fp, fn = (yy & pp).sum(0), (~yy & pp).sum(0), (yy & ~pp).sum(0)
            out[k].append(np.mean(np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(5), where=(2 * tp + fp + fn) > 0)))
    return [np.asarray(o) for o in out]


def model_metrics(run):
    y, p = run.arrays([r["id"] for r in run.test_rows] or None)
    m = scores(y, p, run.thresholds)
    return {"macro_f1": m["macro_f1"], "macro_ap": macro_ap(y, p),
            "negative_fpr": m["negative_window_false_positive_rate"], "f1": m["f1"],
            "recall": m["recall"], "precision": m["precision"]}


# ----------------------------------------------------------------------------- audio
def components(row, cfg, resolve):
    """(mixture, target_part, noise_part) exactly as the model heard it (render_recipe)."""
    row = dict(row, target_path=resolve(row["target_path"]), noise_path=resolve(row["noise_path"]))
    sr, seconds = cfg["sample_rate"], cfg["window_seconds"]
    mixture = render_recipe(row, cfg)
    if row["kind"] != "mixed":
        return mixture, None, None
    target = load_audio(row["target_path"], sr, seconds)
    target = target * (row["target_rms"] / rms(target))
    noise = load_audio(row["noise_path"], sr, seconds)
    noise = noise * (rms(target) / (rms(noise) * 10 ** (row["snr_db"] / 20)))
    peak = max(1.0, float(np.max(np.abs(target + noise))) / 0.95)
    assert np.allclose(mixture, (target + noise) / peak, atol=1e-5)
    return mixture, (target / peak).astype(np.float32), (noise / peak).astype(np.float32)


def write_wav(path, x, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1, 1), sr, subtype="PCM_16")


def mel_db(x, sr):
    import librosa
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr, n_fft=1024, hop_length=320,
                                       n_mels=64, fmin=50, fmax=14000)
    return librosa.power_to_db(m, ref=1.0, amin=1e-10)


def mel_range(waves, sr, span=80.0):
    """One color scale for panels that are compared side by side."""
    top = max(float(mel_db(w, sr).max()) for w in waves)
    return top - span, top


def show_mel(ax, x, sr, title, seconds, vrange=(-90, -10)):
    ax.imshow(mel_db(x, sr), origin="lower", aspect="auto", cmap="magma", vmin=vrange[0], vmax=vrange[1],
              extent=[0, seconds, 0, 64])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("시간 (초)")
    ax.set_yticks([])


# ----------------------------------------------------------------------------- examples
def ladder_groups(run):
    """Per target recording: clean + seen-noise 10/5/0 dB rows (same recording, rising noise)."""
    groups = {}
    for r in run.test_rows:
        if not r["target_path"]:
            continue
        if r["kind"] == "clean" or r["noise_group"] == "candidate_pool":
            groups.setdefault(r["target_path"], {})[condition_of(r)] = r
    return {k: v for k, v in groups.items() if all(c in v for c in SNR_ORDER)}


def pick_examples(run, per_class):
    picks = []
    groups = ladder_groups(run)
    for c in range(5):
        cands = []
        for path, g in sorted(groups.items()):
            if g["clean"]["labels"][c] != 1:
                continue
            correct = [judge(np.asarray(g[k]["labels"]), run.pred[g[k]["id"]][1], run.thresholds, run.names)[0] == "정답"
                       for k in SNR_ORDER]
            margin = run.pred[g[0]["id"]][1][c] - run.thresholds[c]
            cands.append((sum(correct), margin, path, g))
        if not cands:
            continue
        success = sorted([x for x in cands if x[0] == 4], key=lambda x: -x[1])
        hard = sorted([x for x in cands if x[0] < 4], key=lambda x: (x[0], x[1]))
        chosen = []
        if success:
            chosen.append(("success", success[0]))
        if hard:
            chosen.append(("hard", hard[0]))
        for extra in sorted(cands, key=lambda x: -x[1]):
            if len(chosen) >= per_class:
                break
            if all(extra[2] != ch[1][2] for ch in chosen):
                chosen.append(("extra", extra))
        picks += [(c, kind, item[3]) for kind, item in chosen[:per_class]]
    return picks


def ladder_figure(run, c, kind, group, resolve, path):
    sr, seconds = run.cfg["sample_rate"], run.cfg["window_seconds"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), gridspec_kw={"height_ratios": [1.25, 1]},
                             layout="constrained")
    records = []
    waves = [components(group[cond], run.cfg, resolve)[0] for cond in SNR_ORDER]
    vrange = mel_range(waves, sr)
    for j, cond in enumerate(SNR_ORDER):
        row = group[cond]
        mixture = waves[j]
        score = run.pred[row["id"]][1]
        verdict, detected = judge(np.asarray(row["labels"]), score, run.thresholds, run.names)
        noise = f" · 방해음: {row['noise_class']}" if row["noise_class"] else ""
        show_mel(axes[0, j], mixture, sr, f"{SNR_LABEL[cond]}{noise}", seconds, vrange)
        ax = axes[1, j]
        colors = ["#2f7ed8" if i == c else "#b8c2cc" for i in range(5)]
        ax.barh(range(5), score, color=colors)
        for i, t in enumerate(run.thresholds):
            ax.plot([t, t], [i - 0.42, i + 0.42], color="black", lw=1.4)
        ax.set_xlim(0, 1)
        ax.set_yticks(range(5), run.names if j == 0 else [""] * 5)
        ax.invert_yaxis()
        ax.set_xlabel("점수 (검은 선 = 검출 기준)")
        good = verdict == "정답"
        ax.set_title(f"판정: {detected}  [{verdict}]", color="#1a7f37" if good else "#c62828",
                     fontsize=9, fontweight="bold")
        records.append((cond, row, score, verdict, detected))
    fig.suptitle(f"{run.names[c]} — 같은 녹음에 방해음을 점점 크게 섞었을 때의 판정", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return records


def export_examples(run, out, resolve, per_class):
    sr = run.cfg["sample_rate"]
    table = []
    counter = {}
    for c, kind, group in pick_examples(run, per_class):
        counter[c] = counter.get(c, 0) + 1
        slug = f"{run.classes[c]}_{counter[c]:02d}_{kind}"
        fig_path = out / "figures/decisions" / f"{slug}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        records = ladder_figure(run, c, kind, group, resolve, fig_path)
        for cond, row, score, verdict, detected in records:
            mixture, _, noise = components(row, run.cfg, resolve)
            name = "clean" if cond == "clean" else f"snr{cond:02d}dB"
            wav = out / "audio/by_snr" / slug / f"{name}.wav"
            write_wav(wav, mixture, sr)
            noise_wav = ""
            if noise is not None:
                noise_wav = out / "audio/by_snr" / slug / f"{name}_noise_only.wav"
                write_wav(noise_wav, noise, sr)
                noise_wav = noise_wav.relative_to(out).as_posix()
            table.append({"example": slug, "class": run.classes[c], "class_ko": run.names[c], "pick": kind,
                          "condition": SNR_LABEL[cond], "noise_class": row["noise_class"], "id": row["id"],
                          "true_score": float(score[c]), "threshold": float(run.thresholds[c]),
                          "detected": detected, "verdict": verdict, "audio": wav.relative_to(out).as_posix(),
                          "noise_audio": noise_wav, "figure": fig_path.relative_to(out).as_posix(),
                          "target_file": Path(row["target_path"].replace("\\", "/")).name})
    return table


def export_errors(run, out, resolve, limit):
    sr = run.cfg["sample_rate"]
    misses, alarms = [], []
    for r in run.test_rows:
        y, p = run.pred[r["id"]]
        verdict, detected = judge(y, p, run.thresholds, run.names)
        if verdict in ("놓침", "다른 소리로 오인"):
            c = int(np.argmax(y))
            misses.append((p[c] - run.thresholds[c], r, verdict, detected, c))
        elif verdict == "오검출" and not y.any():
            alarms.append((-(p - run.thresholds).max(), r, verdict, detected, None))
    table = []
    for label, items in (("miss", sorted(misses, key=lambda x: x[0])), ("false_alarm", sorted(alarms, key=lambda x: x[0]))):
        for k, (_, r, verdict, detected, c) in enumerate(items[:limit], 1):
            wave, _, _ = components(r, run.cfg, resolve)
            wav = out / "audio/errors" / f"{label}_{k:02d}.wav"
            write_wav(wav, wave, sr)
            cond = "방해음만" if r["kind"] == "negative" else ("무음" if r["kind"] == "silence" else SNR_LABEL[condition_of(r)])
            table.append({"type": "놓침" if label == "miss" else "오검출", "id": r["id"],
                          "true": run.names[c] if c is not None else "목표음 없음", "detected": detected,
                          "condition": cond, "noise_class": r["noise_class"] or "-",
                          "noise_group": GROUP_LABEL.get(r["noise_group"], r["noise_group"]),
                          "audio": wav.relative_to(out).as_posix()})
    return table, len(misses), len(alarms)


# ----------------------------------------------------------------------------- figures
def fig_recall_by_snr(run, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, group in zip(axes, ("candidate_pool", "heldout_types")):
        table = recall_table(run, group)
        for c in range(5):
            ax.plot(range(4), [table[c].get(k, np.nan) for k in SNR_ORDER], marker="o", label=run.names[c])
        macro = [np.nanmean([table[c].get(k, np.nan) for c in range(5)]) for k in SNR_ORDER]
        ax.plot(range(4), macro, color="black", lw=3, marker="s", label="평균")
        ax.set_xticks(range(4), [SNR_LABEL[k] for k in SNR_ORDER])
        ax.set_title(f"{GROUP_LABEL[group]}와 섞었을 때")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("재현율 (목표음을 찾아낸 비율)")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"방해음이 커질수록 얼마나 잘 찾는가 ({run.name})", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_confusion(run, path):
    labels = run.names + ["목표음 없음"]
    mat = np.zeros((6, 6), int)
    for r in run.test_rows:
        y, p = run.pred[r["id"]]
        pred = np.flatnonzero(p >= run.thresholds)
        t = int(np.argmax(y)) if y.any() else 5
        if not len(pred):
            j = 5
        elif t in pred:
            j = t
        else:
            j = int(pred[np.argmax(p[pred])])
        mat[t, j] += 1
    rate = mat / np.maximum(mat.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{mat[i, j]}\n({rate[i, j]:.0%})", ha="center", va="center", fontsize=8,
                    color="white" if rate[i, j] > 0.6 else "black")
    ax.set_xticks(range(6), labels, rotation=30, ha="right")
    ax.set_yticks(range(6), labels)
    ax.set_xlabel("모델 판정")
    ax.set_ylabel("실제 정답")
    ax.set_title(f"혼동표 — 평가 {len(run.test_rows)}구간 ({run.name})\n"
                 "여러 소리를 검출하면 정답 포함 시 정답, 아니면 가장 높은 점수로 셈", fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return mat.tolist()


def fig_false_alarms(run, path, top=12):
    counts = {}
    for r in run.test_rows:
        if r["kind"] != "negative":
            continue
        y, p = run.pred[r["id"]]
        for i in np.flatnonzero(p >= run.thresholds):
            counts.setdefault(r["noise_class"], np.zeros(5, int))[i] += 1
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    if not counts:
        ax.text(0.5, 0.5, "방해음만 있는 구간에서 오검출 없음", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        items = sorted(counts.items(), key=lambda kv: -kv[1].sum())[:top]
        left = np.zeros(len(items))
        for i in range(5):
            vals = np.array([v[i] for _, v in items])
            ax.barh(range(len(items)), vals, left=left, label=f"→ {run.names[i]}")
            left += vals
        ax.set_yticks(range(len(items)), [k for k, _ in items])
        ax.invert_yaxis()
        ax.set_xlabel("오검출 구간 수")
        ax.legend(fontsize=8)
    ax.set_title(f"어떤 방해음을 목표음으로 착각했나 (방해음만 있는 구간, {run.name})")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {k: v.tolist() for k, v in counts.items()}


def fig_training_curve(run, base, path):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for r, style, label in ((run, "-", "이번 실험"), (base, "--", "기존 실험")):
        if r is None or not r.history:
            continue
        x = range(1, len(r.history) + 1)
        ax.plot(x, [h["train_bce"] for h in r.history], style, color="#2f7ed8", label=f"{label} 학습 손실")
        ax.plot(x, [h["val_bce"] for h in r.history], style, color="#e8590c", label=f"{label} 검증 손실")
    head = sum(h["stage"] == "head" for h in run.history)
    if head:
        ax.axvline(head + 0.5, color="gray", lw=1)
        ax.text(head + 0.6, ax.get_ylim()[1] * 0.9, "미세조정 시작", color="gray", fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE 손실")
    ax.set_title(f"학습 곡선 — 학습·검증 손실 차이가 작을수록 과적합이 적음 ({run.name})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def same_test_set(a, b):
    if set(a.pred) != set(b.pred):
        return False
    return all(np.array_equal(a.pred[i][0], b.pred[i][0]) for i in a.pred)


def comparison(runs, base_runs, out):
    names = [n for n in runs if n in base_runs and same_test_set(runs[n], base_runs[n])]
    if not names:
        return None
    new = {n: model_metrics(runs[n]) for n in names}
    old = {n: model_metrics(base_runs[n]) for n in names}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    x = np.arange(len(names))
    for ax, key, title in zip(axes, ("macro_f1", "macro_ap", "negative_fpr"),
                              ("평가 Macro-F1 (높을수록 좋음)", "평가 Macro-AP (높을수록 좋음)",
                               "방해음·무음 구간 오검출률 (낮을수록 좋음)")):
        ax.bar(x - 0.2, [old[n][key] for n in names], 0.4, label="기존", color="#b8c2cc")
        ax.bar(x + 0.2, [new[n][key] for n in names], 0.4, label="이번 실험", color="#2f7ed8")
        ax.set_xticks(x, [n.replace("noise_", "").replace("/", "\n") for n in names], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if key != "negative_fpr":
            ax.set_ylim(max(0, min(min(old[n][key], new[n][key]) for n in names) - 0.1), 1)
    axes[0].legend()
    fig.suptitle("기존 실험과 같은 평가셋으로 비교 (가로축: 방해음 종류 수 / 학습 방식)", fontsize=12)
    fig.savefig(out / "figures/compare_models.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for n, color in zip(([m for m in ("noise_43/mixed", "noise_18/mixed") if m in names] or names[:2]), ("#2f7ed8", "#7048e8")):
        for r, style, label in ((runs[n], "-", "이번"), (base_runs[n], "--", "기존")):
            t = recall_table(r)
            ax.plot(range(4), [np.mean([t[c][k] for c in range(5)]) for k in SNR_ORDER], style, marker="o",
                    color=color, label=f"{n} {label}")
    ax.set_xticks(range(4), [SNR_LABEL[k] for k in SNR_ORDER])
    ax.set_ylabel("평균 재현율")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("SNR별 평균 재현율: 기존 vs 이번 실험")
    fig.savefig(out / "figures/compare_recall_by_snr.png", dpi=170)
    plt.close(fig)
    result = {}
    for n in names:
        rows = runs[n].test_rows
        a, b = cluster_bootstrap_f1([runs[n], base_runs[n]], rows)
        diff = a - b
        result[n] = {"new": new[n], "baseline": old[n],
                     "macro_f1_ci95_new": np.percentile(a, [2.5, 97.5]).tolist(),
                     "macro_f1_ci95_baseline": np.percentile(b, [2.5, 97.5]).tolist(),
                     "macro_f1_diff_ci95": np.percentile(diff, [2.5, 97.5]).tolist(),
                     "prob_new_better": float(np.mean(diff > 0))}
    return result


def augmentation_examples(run, out, resolve):
    """What the model sees in training: fixed recipe vs. on-the-fly augmentation."""
    from augment import augment_settings, render_augmented
    cfg = copy.deepcopy(run.cfg)
    if not run.augmented:
        extra = read_json(ROOT / "config_augmented.json").get("augment")
        if not extra:
            return None
        cfg["augment"] = extra
    s = augment_settings(cfg)
    train = [r for r in run.rows if r["split"] == "train"]
    pool = sorted({resolve(r["noise_path"]) for r in train if r["noise_path"]})
    noise_class = {resolve(r["noise_path"]): r["noise_class"] for r in train if r["noise_path"]}
    mixed = [r for r in train if r["kind"] == "mixed" and r["target_class"] == "glass_breaking"] or \
            [r for r in train if r["kind"] == "mixed"]
    if not mixed or not pool:
        return None
    row = dict(sorted(mixed, key=lambda r: r["id"])[0])
    row["target_path"], row["noise_path"] = resolve(row["target_path"]), resolve(row["noise_path"])
    sr, n = cfg["sample_rate"], round(cfg["sample_rate"] * cfg["window_seconds"])

    def search(settings, accept):
        for seed in range(500):
            wave, info = render_augmented(row, (0.0, 0.0), cfg, settings, "mixed", np.random.default_rng(seed), pool)
            if accept(info):
                return wave, info
        return wave, info

    full = dict(s, partial_overlap_prob=0.0)
    part = dict(s, partial_overlap_prob=1.0)
    panels = [("기존 방식: 항상 0초에서 시작, SNR 고정", render_recipe(row, cfg), f"SNR {row['snr_db']} dB")]
    w, i = search(full, lambda i: i.get("onset_sample", 0) > 1.5 * sr and i["snr_db"] > 3)
    panels.append(("시작 위치 변환", w, f"시작 {i['onset_sample'] / sr:.1f}초 · SNR {i['snr_db']:.1f} dB"))
    w, i = search(part, lambda i: (i["noise_span"][1] - i["noise_span"][0]) < 0.6 * n)
    a, b = i["noise_span"]
    panels.append(("시간 일부 겹치기", w, f"방해음 {a / sr:.1f}~{b / sr:.1f}초만 · SNR {i['snr_db']:.1f} dB"))
    w, i = search(full, lambda i: i["snr_db"] < -2)
    panels.append(("연속 SNR (어려운 조건)", w, f"SNR {i['snr_db']:.1f} dB · 방해음 {noise_class.get(i['noise_path'], '?')}"))
    fig, axes = plt.subplots(2, 4, figsize=(14, 5), layout="constrained", gridspec_kw={"height_ratios": [1, 1.3]})
    files = []
    t = np.arange(n) / sr
    vrange = mel_range([w for _, w, _ in panels], sr)
    for j, (title, wave, note) in enumerate(panels):
        axes[0, j].plot(t, wave, lw=0.4, color="#2f7ed8")
        axes[0, j].set_ylim(-1, 1)
        axes[0, j].set_title(f"{title}\n{note}", fontsize=9)
        axes[0, j].set_yticks([])
        show_mel(axes[1, j], wave, sr, "", cfg["window_seconds"], vrange)
        path = out / "audio/augmentation" / f"{j + 1}_{['fixed', 'onset_shift', 'partial_overlap', 'low_snr'][j]}.wav"
        write_wav(path, wave, sr)
        files.append({"title": title, "note": note, "audio": path.relative_to(out).as_posix()})
    fig.suptitle(f"학습 증강 예시 — 같은 목표음({row['target_class']})을 매 epoch 다르게 섞음", fontsize=12)
    fig.savefig(out / "figures/augmentation_examples.png", dpi=170)
    plt.close(fig)
    spec = s["spec_augment"]
    m = mel_db(panels[1][1], sr)
    masked = m.copy()
    lo, hi = float(m.max()) - 80, float(m.max())
    rng = np.random.default_rng(3)
    for _ in range(spec["time_stripes_num"]):
        w_ = int(rng.integers(max(2, spec["time_drop_width"] // 2), spec["time_drop_width"]))
        s0 = int(rng.integers(0, m.shape[1] - w_))
        masked[:, s0:s0 + w_] = m.min()
    for _ in range(spec["freq_stripes_num"]):
        w_ = int(rng.integers(max(2, spec["freq_drop_width"] // 2), spec["freq_drop_width"]))
        s0 = int(rng.integers(0, m.shape[0] - w_))
        masked[s0:s0 + w_, :] = m.min()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for ax, img, title in zip(axes, (m, masked), ("원래 스펙트로그램", "SpecAugment: 시간·주파수 일부를 가림")):
        ax.imshow(img, origin="lower", aspect="auto", cmap="magma", vmin=lo, vmax=hi,
                  extent=[0, cfg["window_seconds"], 0, 64])
        ax.set_title(title)
        ax.set_xlabel("시간 (초)")
        ax.set_yticks([])
    fig.suptitle("SpecAugment 예시 (학습 중에만 적용, 그림은 이해를 돕기 위한 재현)", fontsize=11)
    fig.savefig(out / "figures/spec_augment_example.png", dpi=170)
    plt.close(fig)
    return files


# ----------------------------------------------------------------------------- html
def write_html(out, ctx):
    e = html.escape

    def audio(src):
        return f'<audio controls preload="none" src="{e(src)}"></audio>' if src else ""

    def img(src, caption):
        return (f'<figure><img src="{e(src)}" alt="{e(caption)}"><figcaption>{e(caption)}</figcaption></figure>'
                if (out / src).is_file() else "")

    m = ctx["selected_metrics"]
    parts = [f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>실험 결과 — {e(ctx['experiment'])}</title>
<style>
:root{{--bg:#fff;--fg:#1d232a;--muted:#5b6670;--line:#dde3e8;--ok:#1a7f37;--bad:#c62828;--card:#f6f8fa}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:"Malgun Gothic",system-ui,sans-serif;line-height:1.55}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:1.6rem;margin:.2em 0}} h2{{margin-top:2.2em;border-bottom:2px solid var(--line);padding-bottom:.3em}}
.muted{{color:var(--muted)}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.kpi b{{display:block;font-size:1.5rem}} figure{{margin:16px 0}} img{{max-width:100%;border:1px solid var(--line);border-radius:8px}}
figcaption{{color:var(--muted);font-size:.9rem}} table{{border-collapse:collapse;width:100%;font-size:.9rem}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:middle}}
.ok{{color:var(--ok);font-weight:bold}} .bad{{color:var(--bad);font-weight:bold}} audio{{height:32px;max-width:220px}}
.scroll{{overflow-x:auto}}
</style></head><body><main>
<h1>실험 결과: {e(ctx['experiment'])}</h1>
<p class="muted">대표 모델 <b>{e(ctx['selected'])}</b> (검증 Macro-F1로 선택) · 학습 증강 {'사용' if ctx['augmented'] else '미사용'} ·
평가 {ctx['test_windows']}구간(5초) · 생성 스크립트 presentation_assets.py</p>
<div class="kpis">
<div class="kpi"><span class="muted">평가 Macro-F1</span><b>{m['macro_f1']:.3f}</b></div>
<div class="kpi"><span class="muted">평가 Macro-AP</span><b>{m['macro_ap']:.3f}</b></div>
<div class="kpi"><span class="muted">방해음·무음 오검출률</span><b>{m['negative_fpr']:.1%}</b></div>
<div class="kpi"><span class="muted">0 dB 평균 재현율</span><b>{ctx['recall_0db']:.1%}</b></div>
</div>"""]
    if ctx.get("comparison"):
        parts.append("<h2>기존 실험과 비교</h2><div class='scroll'><table><tr><th>모델</th><th>기존 F1</th><th>이번 F1</th>"
                     "<th>차이 95% 신뢰구간</th><th>이번이 더 나을 확률</th><th>기존 오검출률</th><th>이번 오검출률</th></tr>")
        for n, c in ctx["comparison"].items():
            lo, hi = c["macro_f1_diff_ci95"]
            parts.append(f"<tr><td>{e(n)}</td><td>{c['baseline']['macro_f1']:.3f}</td><td>{c['new']['macro_f1']:.3f}</td>"
                         f"<td>{lo:+.3f} ~ {hi:+.3f}</td><td>{c['prob_new_better']:.0%}</td>"
                         f"<td>{c['baseline']['negative_fpr']:.1%}</td><td>{c['new']['negative_fpr']:.1%}</td></tr>")
        parts.append("</table></div><p class='muted'>신뢰구간은 같은 원본 녹음에서 나온 구간을 묶어 1000회 bootstrap한 값입니다. "
                     "구간이 0을 포함하면 차이를 확정할 수 없습니다.</p>")
        parts.append(img("figures/compare_models.png", "모델별 평가 지표 비교"))
        parts.append(img("figures/compare_recall_by_snr.png", "SNR별 평균 재현율 비교"))
    parts.append("<h2>SNR별 소리 듣기와 판정</h2><p class='muted'>같은 목표음 녹음에 방해음을 점점 크게 섞었습니다. "
                 "‘방해음만’은 그 구간에 섞인 방해음 성분만 따로 저장한 파일입니다.</p>")
    by_example = {}
    for r in ctx["examples"]:
        by_example.setdefault(r["example"], []).append(r)
    for slug, rows in by_example.items():
        first = rows[0]
        kind = {"success": "모두 맞힌 예시", "hard": "어려운 예시", "extra": "추가 예시"}[first["pick"]]
        parts.append(f"<h3>{e(first['class_ko'])} — {kind} <span class='muted'>({e(first['target_file'])})</span></h3>")
        parts.append(img(first["figure"], "위: 스펙트로그램, 아래: 5개 소리 점수와 검출 기준(검은 선)"))
        parts.append("<div class='scroll'><table><tr><th>조건</th><th>섞은 소리</th><th>방해음만</th><th>방해음 종류</th>"
                     "<th>정답 소리 점수 / 기준</th><th>판정</th></tr>")
        for r in rows:
            cls = "ok" if r["verdict"] == "정답" else "bad"
            parts.append(f"<tr><td>{e(r['condition'])}</td><td>{audio(r['audio'])}</td><td>{audio(r['noise_audio'])}</td>"
                         f"<td>{e(r['noise_class'] or '-')}</td><td>{r['true_score']:.2f} / {r['threshold']:.2f}</td>"
                         f"<td class='{cls}'>{e(r['detected'])} [{e(r['verdict'])}]</td></tr>")
        parts.append("</table></div>")
    parts.append("<h2>전체 판정 결과</h2>")
    parts.append(img("figures/recall_by_snr.png", "방해음 크기별 재현율"))
    parts.append(img("figures/confusion.png", "혼동표"))
    parts.append(img("figures/false_alarms_by_noise.png", "오검출을 일으킨 방해음"))
    parts.append(img("figures/training_curve.png", "학습 곡선"))
    if ctx.get("augmentation"):
        parts.append("<h2>학습 증강은 무엇을 바꾸나</h2>")
        parts.append(img("figures/augmentation_examples.png", "학습 때 모델이 듣는 소리 예시"))
        parts.append("<table><tr><th>방식</th><th>설명</th><th>듣기</th></tr>")
        for a in ctx["augmentation"]:
            parts.append(f"<tr><td>{e(a['title'])}</td><td>{e(a['note'])}</td><td>{audio(a['audio'])}</td></tr>")
        parts.append("</table>")
        parts.append(img("figures/spec_augment_example.png", "SpecAugment 예시"))
    parts.append(f"<h2>틀린 사례</h2><p class='muted'>놓침 {ctx['n_misses']}건, 방해음·무음 오검출 {ctx['n_alarms']}건 중 "
                 "기준에서 가장 멀리 벗어난 사례입니다.</p><div class='scroll'><table><tr><th>유형</th><th>실제</th><th>판정</th>"
                 "<th>조건</th><th>방해음</th><th>듣기</th></tr>")
    for r in ctx["errors"]:
        parts.append(f"<tr><td class='bad'>{e(r['type'])}</td><td>{e(r['true'])}</td><td>{e(r['detected'])}</td>"
                     f"<td>{e(r['condition'])}</td><td>{e(r['noise_class'])} ({e(r['noise_group'])})</td><td>{audio(r['audio'])}</td></tr>")
    parts.append("</table></div></main></body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def build(experiment, out=None, baseline=None, model=None, audio_root=None, per_class=2, errors=8):
    setup_fonts()
    experiment = Path(experiment).resolve()
    out = Path(out).resolve() if out else experiment / "presentation"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {out}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    resolve = make_resolver(audio_root)
    runs = load_runs(experiment)
    selected = model or max(runs, key=lambda n: (runs[n].val_macro_f1, n))
    run = runs[selected]
    if not run.test_rows:
        raise FileNotFoundError(f"Manifest missing for {selected}; needed to render audio")
    base_runs = load_runs(Path(baseline).resolve()) if baseline and Path(baseline).is_dir() else {}
    base = base_runs.get(selected)

    examples = export_examples(run, out, resolve, per_class)
    error_rows, n_miss, n_alarm = export_errors(run, out, resolve, errors)
    fig_recall_by_snr(run, out / "figures/recall_by_snr.png")
    confusion = fig_confusion(run, out / "figures/confusion.png")
    alarms = fig_false_alarms(run, out / "figures/false_alarms_by_noise.png")
    fig_training_curve(run, base, out / "figures/training_curve.png")
    compare = comparison(runs, base_runs, out) if base_runs else None
    augmentation = augmentation_examples(run, out, resolve)
    metrics = model_metrics(run)
    table = recall_table(run)
    recall_0db = float(np.mean([table[c][0] for c in range(5)]))
    all_models = {n: {**model_metrics(r), "val_macro_f1": r.val_macro_f1, "augmented": r.augmented}
                  for n, r in runs.items()}
    summary = {"experiment": str(experiment), "selected_model": selected,
               "selection": "highest validation macro-F1 (validation-tuned thresholds)",
               "augmented": run.augmented, "test_windows": len(run.test_rows), "selected_metrics": metrics,
               "recall_by_condition": {run.classes[c]: {str(k): v for k, v in table[c].items()} for c in range(5)},
               "confusion_rows_true_cols_pred": {"labels": run.classes + ["none"], "matrix": confusion},
               "false_alarms_by_noise_class": alarms, "all_models": all_models,
               "baseline": str(Path(baseline).resolve()) if base_runs else None, "comparison": compare}
    write_json(out / "summary.json", summary)
    for name, rows in (("examples.csv", examples), ("errors.csv", error_rows)):
        if rows:
            with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    write_html(out, {"experiment": experiment.name, "selected": selected, "augmented": run.augmented,
                     "test_windows": len(run.test_rows), "selected_metrics": metrics, "recall_0db": recall_0db,
                     "comparison": compare, "examples": examples, "errors": error_rows,
                     "n_misses": n_miss, "n_alarms": n_alarm, "augmentation": augmentation})
    return out, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="runs/<실험 폴더>")
    p.add_argument("--out", default=None, help="기본값: <실험>/presentation")
    p.add_argument("--baseline", default=str(ROOT / "runs/noise_expansion_20261115"),
                   help="비교할 기존 실험 폴더 (없으면 비교 생략)")
    p.add_argument("--model", default=None, help="예: noise_43/mixed (기본값: 검증 Macro-F1 최고 모델)")
    p.add_argument("--audio-root", default=str(ROOT / "data/ESC-50-master/audio"),
                   help="manifest의 음원 경로가 이 PC에 없을 때 파일명으로 찾을 폴더")
    p.add_argument("--per-class", type=int, default=2, help="소리별 SNR 예시 수")
    p.add_argument("--errors", type=int, default=8, help="놓침·오검출 사례를 각각 몇 개 저장할지")
    a = p.parse_args()
    baseline = a.baseline if Path(a.baseline).resolve() != Path(a.experiment).resolve() else None
    out, summary = build(a.experiment, a.out, baseline, a.model, a.audio_root, a.per_class, a.errors)
    m = summary["selected_metrics"]
    print(f"Selected {summary['selected_model']}: test macro-F1 {m['macro_f1']:.4f}, macro-AP {m['macro_ap']:.4f}")
    print("Saved:", out / "index.html")


if __name__ == "__main__":
    main()
