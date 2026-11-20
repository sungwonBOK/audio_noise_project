import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_utils import render_recipe, rms
from common import ROOT, config
from multi_noise import add_extra_noises, build, noise_pools, portable_fingerprint
from noise_expansion import fingerprint


def tone(path, freq, amp, sr=32000, seconds=5.0):
    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.cfg = config(ROOT / "config.json")
        tone(d / "t.wav", 440, 0.2)
        tone(d / "n1.wav", 1000, 0.05)
        tone(d / "n2.wav", 1700, 0.3)
        tone(d / "n3.wav", 2500, 0.01)
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, kind, extras, snr=0):
        d = self.d
        return dict(kind=kind, target_path=str(d / "t.wav") if kind == "mixed" else "", target_rms=0.05,
                    noise_path=str(d / "n1.wav"), snr_db=snr if kind == "mixed" else None,
                    extra_noises=[{"path": str(d / p), "gain_db": g} for p, g in extras])

    def test_no_extras_is_unchanged(self):
        r = self.row("mixed", [])
        old = {k: v for k, v in r.items() if k != "extra_noises"}
        np.testing.assert_array_equal(render_recipe(r, self.cfg), render_recipe(old, self.cfg))

    def test_snr_is_against_whole_noise_bed(self):
        r = self.row("mixed", [("n2.wav", 0.0), ("n3.wav", -6.0)], snr=5)
        mix = render_recipe(r, self.cfg)
        target = render_recipe({**r, "kind": "clean"}, self.cfg)  # target alone, same gain
        bed = mix - target
        self.assertAlmostEqual(20 * np.log10(rms(target) / rms(bed)), 5.0, places=2)

    def test_extra_gain_is_relative_to_primary_noise(self):
        r = self.row("negative", [("n2.wav", -6.0)])
        bed = render_recipe(r, self.cfg)
        primary = render_recipe({**r, "extra_noises": []}, self.cfg)
        extra = bed - primary
        self.assertAlmostEqual(20 * np.log10(rms(extra) / rms(primary)), -6.0, places=2)

    def test_long_rows_refuse_extras(self):
        with self.assertRaises(ValueError):
            render_recipe({**self.row("mixed", [("n2.wav", 0.0)]), "source": "long"}, self.cfg)


class ExtraNoiseTests(unittest.TestCase):
    def rows(self):
        rows = []
        for split in ("train", "val"):
            for i in range(6):
                rows.append(dict(split=split, kind="negative", noise_group="seen", noise_path=f"{split}{i}.wav",
                                 noise_source=f"{split}-src{i}", noise_class=f"c{i % 3}",
                                 target_source="", labels=[0] * 5))
            rows.append(dict(split=split, kind="mixed", noise_group="seen", noise_path=f"{split}0.wav",
                             noise_source=f"{split}-src0", noise_class="c0", target_source=f"{split}-t",
                             labels=[1, 0, 0, 0, 0]))
            rows.append(dict(split=split, kind="clean", noise_group="none", noise_path="", noise_source="",
                             noise_class="", target_source=f"{split}-t", labels=[1, 0, 0, 0, 0]))
        return rows

    def test_extras_distinct_same_split_and_capped(self):
        rows = self.rows()
        out = add_extra_noises(rows, noise_pools(rows), [3, 5], [-6, 0], random.Random(0))
        self.assertEqual(rows[0].get("extra_noises"), None, "input rows must not be modified")
        for r in out:
            if r["kind"] == "clean":
                self.assertNotIn("extra_noises", r)
                continue
            classes = [r["noise_class"]] + [e["class"] for e in r["extra_noises"]]
            sources = [r["noise_source"]] + [e["source"] for e in r["extra_noises"]]
            self.assertEqual(len(set(classes)), len(classes))
            self.assertEqual(len(set(sources)), len(sources))
            self.assertTrue(all(s.startswith(r["split"]) for s in sources))
            self.assertEqual(r["noise_count"], 3)  # only 3 classes exist; 5 is capped
            self.assertTrue(all(-6 <= e["gain_db"] <= 0 for e in r["extra_noises"]))
            self.assertEqual(r["labels"], next(x for x in rows if x["kind"] == r["kind"]
                                               and x["split"] == r["split"])["labels"])

    def test_portable_fingerprint_ignores_folders(self):
        a = [{"target_path": "C:\\a\\x.wav", "extra_noises": [{"path": "C:\\a\\n.wav"}], "snr_db": 0}]
        b = [{"target_path": "/content/b/x.wav", "extra_noises": [{"path": "/content/b/n.wav"}], "snr_db": 0}]
        self.assertEqual(portable_fingerprint(a), portable_fingerprint(b))
        self.assertNotEqual(portable_fingerprint(a), portable_fingerprint([{**b[0], "snr_db": 5}]))


class BuildTests(unittest.TestCase):
    def test_single_matches_expansion_stage_and_no_leakage(self):
        cfg = config(ROOT / "config.json")
        categories = cfg["classes"] + cfg["seen_noise"] + cfg["unseen_noise"] + [f"extra_{i:02d}" for i in range(37)]
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp)
            meta = []
            for fold in range(1, 6):
                for cat in categories:
                    name = f"{fold}-{cat}.wav"
                    (audio / name).touch()
                    meta.append(dict(filename=name, fold=str(fold), category=cat, src_file=f"{fold}-{cat}"))
            stage_cfg, manifests, info = build(meta, audio, cfg)
        single, multi = manifests["single"]["rows"], manifests["multi"]["rows"]
        for split in ("val", "test", "val_multi", "test_multi"):
            self.assertEqual(fingerprint([r for r in single if r["split"] == split]),
                             fingerprint([r for r in multi if r["split"] == split]))
        self.assertEqual(fingerprint([r for r in single if r["split"] == "val"]), info["val_fingerprint"])
        self.assertTrue(all("extra_noises" not in r for r in single if r["split"] == "train"))
        tr_s = [r for r in single if r["split"] == "train"]
        tr_m = [r for r in multi if r["split"] == "train"]
        self.assertEqual(len(tr_s), len(tr_m))
        for a, b in zip(tr_s, tr_m):  # same rows; only extra noises added
            self.assertEqual({k: v for k, v in b.items() if k not in ("extra_noises", "noise_count")}, a)
        self.assertTrue(any(len(r.get("extra_noises", [])) > 0 for r in tr_m))
        for r in (r for r in single if r["split"].endswith("_multi") and "noise_count" in r):
            self.assertGreaterEqual(r["noise_count"], 2)
        self.assertEqual(stage_cfg["multi_noise"]["stage"], "noise_43")
        self.assertTrue(all(sum(r["labels"]) <= 1 for r in multi))


if __name__ == "__main__":
    unittest.main()
