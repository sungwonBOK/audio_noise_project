"""On-the-fly training augmentation: reproducible, label-safe, and off by default."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch

from audio_utils import rms, max_frame_dbfs, render_recipe
from augment import (AugmentedRecipeDataset, augment_settings, crop_slots, mix_noise,
                     shift_target, render_augmented)
from common import ROOT, config
from model import SoundDetector

SR = 32000


def cfg_with(**augment):
    cfg = config(ROOT / "config.json")
    cfg["augment"] = {"enabled": True, **augment}
    return cfg


def row(kind, target="", noise="", label=None):
    return dict(split="train", kind=kind, labels=label or [0] * 5, target_path=target,
                noise_path=noise, target_rms=0.05, snr_db=5 if kind == "mixed" else None)


class AugmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        burst = np.zeros(5 * SR, np.float32)          # ESC-50-like: event then digital zeros
        burst[:SR // 2] = rng.normal(0, 0.1, SR // 2)
        cls.burst = root / "burst.wav"
        sf.write(cls.burst, burst, SR, subtype="FLOAT")
        cls.noise = root / "noise.wav"
        sf.write(cls.noise, rng.normal(0, 0.05, 5 * SR).astype(np.float32), SR, subtype="FLOAT")
        long_target = np.full(20 * SR, 1e-4, np.float32)  # quiet bed, event at 12.0-12.5 s
        long_target[12 * SR:12 * SR + SR // 2] = rng.normal(0, 0.2, SR // 2)
        cls.long_target = root / "long_target.wav"
        sf.write(cls.long_target, long_target, SR, subtype="FLOAT")
        cls.rows = [row("clean", cls.burst, label=[1, 0, 0, 0, 0]),
                    row("mixed", cls.burst, cls.noise, [1, 0, 0, 0, 0]),
                    row("negative", noise=cls.noise),
                    row("silence")]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_missing_section_is_disabled_and_unknown_keys_rejected(self):
        cfg = config(ROOT / "config.json")
        self.assertFalse(augment_settings(cfg)["enabled"])
        self.assertFalse(augment_settings(cfg)["spec_augment"]["enabled"])
        cfg["augment"] = {"enabled": True, "typo_key": 1}
        with self.assertRaises(ValueError):
            augment_settings(cfg)

    def test_reproducible_within_epoch_and_fresh_across_epochs(self):
        ds = AugmentedRecipeDataset(self.rows, cfg_with(), "mixed")
        ds.set_epoch(0)
        a = [ds[i][0] for i in range(len(ds))]
        b = [ds[i][0] for i in range(len(ds))]
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        ds.set_epoch(1)
        c = [ds[i][0] for i in range(len(ds))]
        self.assertFalse(torch.equal(a[1], c[1]))       # the mixed row changed
        self.assertEqual(float(c[3].abs().max()), 0.0)  # silence stays silence
        self.assertEqual(ds[0][1].tolist(), [1, 0, 0, 0, 0])

    def test_onset_moves_and_target_energy_is_kept(self):
        s = augment_settings(cfg_with())
        n = 5 * SR
        x = np.zeros(n, np.float32)
        x[:SR // 2] = np.random.default_rng(1).normal(0, 0.1, SR // 2)
        total = float(np.sum(np.square(x, dtype=np.float64)))
        onsets = set()
        for seed in range(40):
            out, onset = shift_target(x, np.random.default_rng(seed), s, n)
            onsets.add(onset)
            self.assertEqual(len(out), n)
            self.assertGreaterEqual(float(np.sum(np.square(out, dtype=np.float64))),
                                    s["min_target_energy_keep"] * total - 1e-9)
        self.assertGreater(len(onsets), 30)
        self.assertTrue(any(o > SR for o in onsets))     # no longer pinned to t=0

    def test_continuous_snr_and_partial_overlap(self):
        s = augment_settings(cfg_with(partial_overlap_prob=0.0))
        rng = np.random.default_rng(3)
        t = rng.normal(0, 0.02, SR).astype(np.float32)
        bed = rng.normal(0, 0.02, SR).astype(np.float32)
        mixed, span = mix_noise(t, bed, 3.7, rng, s)
        self.assertEqual(span, (0, SR))
        self.assertAlmostEqual(20 * np.log10(rms(t) / rms(mixed - t)), 3.7, places=4)
        s = augment_settings(cfg_with(partial_overlap_prob=1.0))
        mixed, (a, b) = mix_noise(t, bed, 0.0, rng, s)
        self.assertLess(b - a, SR + 1)
        residual = mixed - t
        self.assertEqual(float(np.abs(residual[:a]).max(initial=0)), 0.0)
        self.assertEqual(float(np.abs(residual[b:]).max(initial=0)), 0.0)
        snrs = [render_augmented(self.rows[1], (0, 0), cfg_with(), augment_settings(cfg_with()),
                                 "mixed", np.random.default_rng(k), [str(self.noise)])[1]["snr_db"]
                for k in range(50)]
        self.assertTrue(all(-5 <= v <= 15 for v in snrs))
        self.assertGreater(len({round(v, 3) for v in snrs}), 45)

    def test_long_source_overlapping_crops_keep_the_event(self):
        cfg = cfg_with(time_shift=False)
        s = augment_settings(cfg)
        slots = crop_slots(str(self.long_target), 5.0, s)
        self.assertEqual(len(slots), 16)                 # 20 s file, 5 s window, 1 s hop
        self.assertTrue(all(b > a for a, b in slots))
        self.assertEqual(crop_slots(str(self.burst), 5.0, s), [(0.0, 0.0)])
        ds = AugmentedRecipeDataset([row("clean", self.long_target, label=[0, 1, 0, 0, 0])], cfg, "mixed")
        self.assertEqual(len(ds), 16)
        starts = set()
        for i in range(len(ds)):
            wave, info = ds.render(i)
            starts.add(round(info["crop_start"], 2))
            self.assertGreater(max_frame_dbfs(wave, SR), -30)  # event present in every positive crop
        self.assertGreater(len(starts), 1)

    def test_validation_recipe_is_unchanged(self):
        cfg = cfg_with()
        r = dict(self.rows[1], noise_path=str(self.noise), target_path=str(self.burst))
        self.assertTrue(np.array_equal(render_recipe(r, cfg), render_recipe(r, config(ROOT / "config.json"))))

    def test_spec_augment_only_in_training_and_backbone_stays_eval(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        plain = SoundDetector()
        model = SoundDetector(spec_augment=augment_settings(cfg_with(spec_augment={"enabled": True}))["spec_augment"])
        model.load_state_dict(plain.state_dict())          # no extra parameters
        wave = torch.randn(2, SR) * 0.02
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(wave), plain.eval()(wave))
        model.train()
        self.assertTrue(model.spec_augmenter.training)
        self.assertFalse(model.backbone.training)
        with torch.no_grad():
            self.assertFalse(torch.allclose(model(wave), model(wave)))


if __name__ == "__main__":
    unittest.main()
