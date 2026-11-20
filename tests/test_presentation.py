"""Presentation assets and result packing work from saved results only (no checkpoint)."""
import csv
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf

from common import ROOT, config, write_json
from prepare import build_manifest
from presentation_assets import build
from tools.pack_results import pack

SR = 32000


def fake_experiment(root):
    cfg = config(ROOT / "config.json")
    audio = root / "audio"
    audio.mkdir()
    rng = np.random.default_rng(0)
    meta = []
    categories = cfg["classes"] + cfg["seen_noise"] + cfg["unseen_noise"]
    for fold in range(1, 6):
        for i, category in enumerate(categories):
            for k in range(2):
                name = f"{fold}-{i}{k}-A-{i}.wav"
                x = np.zeros(5 * SR, np.float32)
                x[: SR * (1 + i % 4)] = rng.normal(0, 0.05, SR * (1 + i % 4))
                sf.write(audio / name, x, SR, subtype="FLOAT")
                meta.append(dict(filename=name, fold=str(fold), src_file=f"{fold}-{i}{k}", category=category))
    manifest = build_manifest(meta, audio, cfg)
    for r in manifest["rows"]:
        if r["split"] != "train":
            r["noise_group"] = {"seen": "candidate_pool", "unseen": "heldout_types"}.get(r["noise_group"], r["noise_group"])
    exp = root / "exp"
    for stage in ("noise_06",):
        base = exp / stage
        write_json(base / "config.json", cfg)
        write_json(base / "manifest.json", manifest)
        for mode in ("clean", "mixed"):
            run = base / mode
            (run / "test").mkdir(parents=True)
            (run / "validation").mkdir(parents=True)
            write_json(run / "validation/thresholds.json", {"thresholds": [0.5] * 5})
            write_json(run / "validation/report.json", {"ungated": {"metrics": {"macro_f1": 0.8 if mode == "mixed" else 0.7}}})
            with (run / "test/predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["id", "kind", "noise_group", "noise_class", "snr_db", "labels", "scores", "gated_scores", "skipped"])
                for r in manifest["rows"]:
                    if r["split"] != "test":
                        continue
                    hard = (r["snr_db"] == 0) * 0.6
                    s = np.clip(np.asarray(r["labels"]) * (0.9 - hard) + rng.uniform(0, 0.3, 5), 0, 1)
                    w.writerow([r["id"], r["kind"], r["noise_group"], r["noise_class"], r["snr_db"],
                                r["labels"], s.tolist(), "", False])
            with (run / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["stage", "epoch", "train_bce", "val_bce"])
                for e in range(1, 4):
                    w.writerow(["head" if e < 3 else "finetune", e, 0.1 / e, 0.08])
            (run / "best.pt").write_bytes(b"x" * 10)
    return exp, audio


class PresentationTests(unittest.TestCase):
    def test_build_and_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp, audio = fake_experiment(Path(tmp))
            out, summary = build(exp, baseline=exp, audio_root=audio, per_class=2, errors=3)
            self.assertEqual(summary["selected_model"], "noise_06/mixed")   # chosen by validation
            for name in ("index.html", "summary.json", "examples.csv", "figures/recall_by_snr.png",
                         "figures/confusion.png", "figures/false_alarms_by_noise.png",
                         "figures/training_curve.png", "figures/compare_models.png",
                         "figures/augmentation_examples.png", "figures/spec_augment_example.png"):
                self.assertTrue((out / name).is_file(), name)
            wavs = list((out / "audio/by_snr").rglob("*.wav"))
            self.assertTrue(any(p.name == "snr00dB.wav" for p in wavs))
            self.assertTrue(any(p.name == "snr00dB_noise_only.wav" for p in wavs))
            info = sf.info(str(wavs[0]))
            self.assertEqual((info.samplerate, round(info.duration, 2)), (SR, 5.0))
            page = (out / "index.html").read_text(encoding="utf-8")
            self.assertIn("audio/by_snr/", page)
            self.assertNotIn("qkrwl", page)
            with self.assertRaises(FileExistsError):
                build(exp, audio_root=audio)
            zpath, n, skipped = pack(exp)
            with zipfile.ZipFile(zpath) as z:
                names = z.namelist()
            self.assertEqual(skipped, 2)
            self.assertFalse(any(n_.endswith(".pt") for n_ in names))
            self.assertTrue(any(n_.endswith("presentation/index.html") for n_ in names))
            zpath, _, skipped = pack(exp, Path(tmp) / "with_model.zip", "noise_06/mixed")
            self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
