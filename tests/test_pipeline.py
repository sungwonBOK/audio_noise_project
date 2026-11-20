"""Meaningful safeguards: split leakage, SNR, target scope, gate and thresholds."""
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf

from audio_utils import load_audio, mix_at_snr, rms, max_frame_dbfs, window_intervals, render_recipe
from common import ROOT, config, device_for
from metrics import scores, tune_thresholds
from prepare import build_manifest


class PipelineTests(unittest.TestCase):
    def test_cuda_does_not_silently_fall_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                device_for("cuda")
        self.assertEqual(device_for("cpu").type, "cpu")

    def test_snr_and_peak(self):
        rng = np.random.default_rng(12)
        target = rng.normal(0, 0.02, 10000).astype(np.float32)
        noise = rng.normal(0, 0.02, 10000).astype(np.float32)
        mixed = mix_at_snr(target, noise, 5)
        self.assertAlmostEqual(20*np.log10(rms(target)/rms(mixed-target)), 5, places=5)
        self.assertLessEqual(abs(mix_at_snr(target*100, noise, -10)).max(), 0.950001)
        with self.assertRaises(ValueError):
            mix_at_snr(target, np.zeros_like(target), 0)

    def test_gate_preserves_short_loud_event(self):
        x = np.zeros(160000, np.float32)
        self.assertLess(max_frame_dbfs(x, 32000), -100)
        x[80000:80320] = 0.1
        self.assertGreater(max_frame_dbfs(x, 32000), -40)

    def test_window_coverage_and_short_files(self):
        self.assertEqual(list(window_intervals(7, 5, 1)), [(0, 5), (1, 6), (2, 7)])
        self.assertEqual(list(window_intervals(2, 5, 1)), [(0, 2)])
        self.assertEqual(list(window_intervals(0, 5, 1)), [])
        self.assertEqual(list(window_intervals(10.5, 5, 5)), [(0, 5), (5, 10), (10, 10.5)])
        with self.assertRaises(ValueError):
            list(window_intervals(10, 5, 6))

    def test_resampling_stereo_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            x = np.full((22050, 2), 0.1, np.float32)
            sf.write(path, x, 44100, subtype="FLOAT")
            y = load_audio(path, 32000, 1.0)
            self.assertEqual(y.shape, (32000,))
            self.assertAlmostEqual(float(y[8000]), 0.1, places=3)
            self.assertEqual(float(y[-1]), 0.0)

    def test_manifest_disjoint_and_single_target(self):
        cfg = config(ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = []
            categories = cfg["classes"] + cfg["seen_noise"] + cfg["unseen_noise"]
            for fold in range(1, 6):
                for i, category in enumerate(categories):
                    filename = f"{fold}-{i}.wav"
                    sf.write(root / filename, np.ones(16000, np.float32)*0.02, 32000)
                    meta.append(dict(filename=filename, fold=str(fold), src_file=f"{fold}-{i}", category=category))
            m = build_manifest(meta, root, cfg)
            self.assertTrue(all(sum(r["labels"]) <= 1 for r in m["rows"]))
            train = [r for r in m["rows"] if r["split"] == "train"]
            self.assertTrue(all(r["noise_group"] != "unseen" for r in train))
            self.assertTrue(any(r["noise_group"] == "unseen" for r in m["rows"]))
            row = next(r for r in train if r["kind"] == "mixed")
            self.assertFalse(np.array_equal(render_recipe(row, cfg, "clean"), render_recipe(row, cfg, "mixed")))
            self.assertTrue(np.array_equal(render_recipe(row, cfg), render_recipe(row, cfg)))
            broken = copy.deepcopy(meta)
            broken[-1]["src_file"] = broken[0]["src_file"]
            with self.assertRaisesRegex(ValueError, "leakage"):
                build_manifest(broken, root, cfg)

    def test_thresholds_and_negative_denominator(self):
        y = np.eye(5, dtype=int).tolist() + [[0]*5]*2
        probs = np.asarray(y)*0.8 + 0.1
        thresholds = tune_thresholds(y, probs)
        m = scores(y, probs, thresholds)
        self.assertEqual(m["macro_f1"], 1.0)
        probs[-1, 0] = 0.99
        m = scores(y, probs, thresholds)
        self.assertEqual(m["negative_window_false_positive_rate"], 0.5)
        self.assertEqual(m["negative_windows"], 2)


if __name__ == "__main__":
    unittest.main()
