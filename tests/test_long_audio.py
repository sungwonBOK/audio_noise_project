"""Long clean recordings: teacher thresholds, -1 (uncertain) labels, window rows, rendering."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch

from audio_utils import render_recipe, rms
from augment import AugmentedRecipeDataset
from common import ROOT, config, freesound_id
from metrics import scores, tune_thresholds
from prepare import add_long_rows
from pseudo_label import apply_thresholds, drop_esc50_overlap, find_audio, pick_thresholds
from train import masked_bce

SR = 32000
CFG = config(ROOT / "config.json")
CLASSES = CFG["classes"]


class TeacherThresholdTests(unittest.TestCase):
    def test_three_way_labels(self):
        p = np.array([[0.9, 0.001, 0.05, 0.2, 0.0]])
        labels = apply_thresholds(p, pos_th=[0.5] * 5, neg_th=[0.01] * 5)
        self.assertEqual(labels.tolist(), [[1, 0, -1, -1, 0]])

    def test_pick_thresholds_meets_precision_and_miss(self):
        rng = np.random.default_rng(0)
        y = np.zeros((400, 5), int)
        y[:100, 0] = 1
        y[100:200, 1] = 1
        y[200:300, 2] = 1
        y[300:350, 3] = 1
        y[350:400, 4] = 1
        p = np.where(y == 1, rng.uniform(0.3, 1.0, y.shape), rng.uniform(0.0, 0.4, y.shape))
        pos, neg, details = pick_thresholds(y, p, precision=0.95, miss=0.05)
        for i in range(5):
            yy = y[:, i] == 1
            self.assertGreaterEqual(yy[p[:, i] >= pos[i]].mean(), 0.95)
            self.assertLessEqual((p[yy, i] < neg[i]).mean(), 0.05)
            self.assertLessEqual(neg[i], pos[i])

    def test_folder_layout_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "train" / "siren" / "a.wav"
            good.parent.mkdir(parents=True)
            sf.write(good, np.zeros(SR, np.float32), SR)
            self.assertEqual([(s, c, p.name) for s, c, p in find_audio(root, CLASSES)], [("train", "siren", "a.wav")])
            bad = root / "train" / "dog" / "b.wav"
            bad.parent.mkdir(parents=True)
            sf.write(bad, np.zeros(SR, np.float32), SR)
            with self.assertRaises(SystemExit):
                find_audio(root, CLASSES)


class Esc50OverlapTests(unittest.TestCase):
    def test_freesound_id_from_file_name(self):
        self.assertEqual(freesound_id("data/long_audio/train/siren/64760.wav"), "64760")      # FSD50K
        self.assertEqual(freesound_id("64760__someuser__siren-passing.flac"), "64760")       # Freesound download
        self.assertEqual(freesound_id("0123.wav"), "123")
        self.assertIsNone(freesound_id("street_2026.wav"))
        self.assertIsNone(freesound_id("64760_take2.wav"))

    def test_label_skips_files_shared_with_esc50(self):
        files = [("train", "siren", Path("a/100032.wav")), ("test", "siren", Path("b/555.wav")),
                 ("val", "none", Path("c/my_room.wav"))]
        kept, dropped = drop_esc50_overlap(files, {"100032"})
        self.assertEqual([p.name for _, _, p in kept], ["555.wav", "my_room.wav"])
        self.assertEqual([p.name for _, _, p in dropped], ["100032.wav"])

    def test_real_esc50_ids_are_numeric(self):
        meta = ROOT / "data" / "ESC-50-master" / "meta" / "esc50.csv"
        if not meta.is_file():
            self.skipTest("ESC-50 not downloaded")
        from common import esc50_source_ids
        ids = esc50_source_ids(meta.parents[1])
        self.assertGreater(len(ids), 100)
        self.assertIn("100032", ids)                                        # 1-100032-A-0.wav


class UncertainLabelTests(unittest.TestCase):
    def test_masked_bce_equals_standard_without_uncertain(self):
        logits = torch.randn(6, 5)
        y = torch.randint(0, 2, (6, 5)).float()
        self.assertAlmostEqual(float(masked_bce(logits, y)),
                               float(torch.nn.BCEWithLogitsLoss()(logits, y)), places=6)

    def test_masked_bce_ignores_uncertain(self):
        logits = torch.randn(4, 5)
        y = torch.randint(0, 2, (4, 5)).float()
        y_unc = y.clone()
        y_unc[0, 2] = -1
        logits2 = logits.clone()
        logits2[0, 2] = 50.0                      # any value at an uncertain cell must not matter
        self.assertAlmostEqual(float(masked_bce(logits, y_unc)), float(masked_bce(logits2, y_unc)), places=6)

    def test_epoch_loss_is_mean_over_known_cells(self):
        from train import epoch

        class Fixed(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits

            def forward(self, wave):
                return self.logits[wave[:, 0].long()]

        logits = torch.randn(4, 5)
        y = torch.randint(0, 2, (4, 5)).float()
        y[2:, 1:] = -1                                 # second batch is mostly uncertain
        wave = torch.arange(4).float().unsqueeze(1).repeat(1, 3)
        loader = [(wave[:2], y[:2]), (wave[2:], y[2:])]
        got = epoch(Fixed(logits), loader, torch.device("cpu"), masked_bce)
        known = y >= 0
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[known], y[known])
        self.assertAlmostEqual(got, float(expected), places=5)
        y_full = torch.randint(0, 2, (4, 5)).float()
        loader = [(wave[:2], y_full[:2]), (wave[2:], y_full[2:])]
        self.assertAlmostEqual(epoch(Fixed(logits), loader, torch.device("cpu"), masked_bce),
                               float(torch.nn.BCEWithLogitsLoss()(logits, y_full)), places=5)

    def test_metrics_skip_uncertain(self):
        y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        p = np.where(y == 1, 0.9, 0.1)
        base = scores(y, p, [0.5] * 5)
        y2 = y.copy()
        y2[2, 0] = -1
        p2 = p.copy()
        p2[2, 0] = 0.99                           # would be a false positive if counted
        s = scores(y2, p2, [0.5] * 5)
        self.assertEqual(s["fp"], base["fp"])
        self.assertEqual(s["uncertain_labels_ignored"], 1)
        self.assertNotIn("uncertain_labels_ignored", base)
        self.assertEqual(tune_thresholds(y2, p2), tune_thresholds(y, p))


class LongWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(1)
        x = np.full(20 * SR, 0.001, np.float32)
        t = np.arange(SR) / SR
        x[12 * SR:13 * SR] = 0.2 * np.sin(2 * np.pi * 800 * t)      # event at 12-13 s (no limiter clipping)
        cls.long = root / "audio" / "train" / "siren" / "street.wav"
        cls.long.parent.mkdir(parents=True)
        sf.write(cls.long, x, SR, subtype="FLOAT")
        cls.noise = root / "noise.wav"
        sf.write(cls.noise, rng.normal(0, 0.05, 5 * SR).astype(np.float32), SR, subtype="FLOAT")
        cls.labels_dir = root / "labels"
        cls.labels_dir.mkdir()
        windows = []
        for start, label in ((0.0, 0), (5.0, -1), (10.0, 1), (15.0, 0)):
            w = {"split": "train", "file_class": "siren", "path": str(cls.long), "start_seconds": start,
                 "end_seconds": start + 5, "window_rms": rms(x[int(start * SR):int((start + 5) * SR)])}
            for c in CLASSES:
                w[f"label_{c}"] = label if c == "siren" else (0 if label != -1 else -1)
            windows.append(w)
        with (cls.labels_dir / "windows.csv").open("w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(windows[0]))
            wr.writeheader()
            wr.writerows(windows)
        (cls.labels_dir / "meta.json").write_text(json.dumps(
            {k: CFG[k] for k in ("classes", "window_seconds", "sample_rate")}), encoding="utf-8")
        negative = dict(split="train", kind="negative", noise_group="seen", noise_path=str(cls.noise),
                        noise_source="n", noise_class="rain", labels=[0] * 5, target_path="",
                        target_rms=0, snr_db=None, id="train-000000")
        cls.manifest = add_long_rows({"rows": [negative]}, cls.labels_dir, CFG)
        cls.rows = [r for r in cls.manifest["rows"] if r.get("source") == "long"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_per_window_and_all_uncertain_dropped(self):
        self.assertEqual(len(self.rows), 3 * 2)                        # 3 windows x (clean + seen mix)
        self.assertEqual(self.manifest["long_labels"]["dropped_all_uncertain_windows"], 1)
        self.assertEqual(sorted({r["target_start"] for r in self.rows}), [0.0, 10.0, 15.0])
        self.assertEqual({r["kind"] for r in self.rows}, {"clean", "mixed"})

    def test_prepare_refuses_long_file_that_is_an_esc50_recording(self):
        negative = dict(split="train", kind="negative", noise_group="seen", noise_path=str(self.noise),
                        noise_source="n", noise_class="rain", labels=[0] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shared = tmp / "100032.wav"
            sf.write(shared, np.full(5 * SR, 0.01, np.float32), SR)
            w = {"split": "train", "file_class": "siren", "path": str(shared), "start_seconds": 0.0,
                 "end_seconds": 5.0, "window_rms": 0.01, **{f"label_{c}": 0 for c in CLASSES}}
            with (tmp / "windows.csv").open("w", newline="", encoding="utf-8-sig") as f:
                wr = csv.DictWriter(f, fieldnames=list(w))
                wr.writeheader()
                wr.writerow(w)
            (tmp / "meta.json").write_text(json.dumps(
                {k: CFG[k] for k in ("classes", "window_seconds", "sample_rate")}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ESC-50 recordings"):
                add_long_rows({"rows": [dict(negative)]}, tmp, CFG, esc_sources={"100032"})
            m = add_long_rows({"rows": [dict(negative)]}, tmp, CFG, esc_sources={"999"})
            self.assertEqual(sum(r.get("source") == "long" for r in m["rows"]), 2)

    def test_render_reads_labeled_window_with_file_gain(self):
        clean = {r["target_start"]: r for r in self.rows if r["kind"] == "clean"}
        event = render_recipe(clean[10.0], CFG)
        quiet = render_recipe(clean[0.0], CFG)
        self.assertEqual(event.shape, (5 * SR,))
        self.assertAlmostEqual(rms(event), clean[10.0]["target_rms"], places=4)   # loudest window -> target_rms
        self.assertLess(rms(quiet), rms(event) / 20)                              # quiet window stays quiet
        self.assertGreater(np.abs(event[2 * SR:3 * SR]).max(), 0.1)               # 12-13 s is at 2-3 s

    def test_noise_level_is_set_against_file_target_level(self):
        for r in self.rows:
            if r["kind"] != "mixed":
                continue
            clean = render_recipe({**r, "kind": "clean"}, CFG)
            mixed = render_recipe(r, CFG)
            if np.abs(mixed).max() >= 0.95:
                continue
            expected = r["target_rms"] / 10 ** (r["snr_db"] / 20)
            self.assertAlmostEqual(rms(mixed - clean), expected, places=4)

    def test_augmented_dataset_keeps_window_position(self):
        cfg = {**CFG, "augment": {"enabled": True}}
        ds = AugmentedRecipeDataset(self.manifest["rows"], cfg, "mixed")
        self.assertEqual(len(ds), 1 + len(self.rows))
        for i, (row_index, lo, hi) in enumerate(ds.slots):
            r = ds.rows[row_index]
            if r.get("source") != "long":
                continue
            self.assertEqual((lo, hi), (r["target_start"], r["target_start"]))
            wave, info = ds.render(i)
            self.assertEqual(wave.shape, (5 * SR,))
            self.assertEqual(info["crop_start"], r["target_start"])
            labels = ds[i][1]
            self.assertEqual(labels.tolist(), [float(v) for v in r["labels"]])


if __name__ == "__main__":
    unittest.main()
