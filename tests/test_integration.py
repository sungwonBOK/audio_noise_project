"""Exercise file/CLI contracts with a tiny model; real Cnn14 is tested separately."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
from torch import nn

import train
import evaluate
import infer
import plot_results
from common import ROOT, config, write_json, read_json
from prepare import build_manifest


class TinyDetector(nn.Module):
    def __init__(self, spec_augment=None):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.head = nn.Linear(8, 5)
        self.configure_stage("head")

    def load_pretrained(self, path):
        pass

    def configure_stage(self, stage):
        self.backbone.requires_grad_(stage == "finetune")

    def forward(self, wave):
        x = torch.stack([wave.mean(1), wave.square().mean(1), wave.abs().mean(1), wave.amax(1)], 1)
        return self.head(self.backbone(x))


class IntegrationTests(unittest.TestCase):
    def run_main(self, module, args):
        with patch.object(sys, "argv", [module.__name__] + [str(a) for a in args]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_train_tune_evaluate_infer_plot(self):
        torch.set_num_threads(2)
        cfg = config(ROOT / "config.json")
        cfg.update(head_epochs=1, finetune_epochs=1, silence_examples=1, batch_size=16,
                   amp=False, window_seconds=1.0, hop_seconds=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = []
            categories = cfg["classes"] + cfg["seen_noise"] + cfg["unseen_noise"]
            t = np.arange(32000) / 32000
            for fold in range(1, 6):
                for i, category in enumerate(categories):
                    name = f"{fold}-{i}.wav"
                    sf.write(root / name, (0.05*np.sin(2*np.pi*(100+i*50)*t)).astype(np.float32), 32000)
                    meta.append(dict(filename=name, fold=str(fold), src_file=f"{fold}-{i}", category=category))
            manifest_path = root / "manifest.json"
            write_json(manifest_path, build_manifest(meta, root, cfg))
            config_path = root / "config.json"
            write_json(config_path, cfg)
            pretrained = root / "dummy_pretrained"
            pretrained.write_bytes(b"synthetic-test-only")
            run = root / "run"
            with patch.object(train, "SoundDetector", TinyDetector):
                self.run_main(train, ["--config", config_path, "--manifest", manifest_path,
                    "--pretrained", pretrained, "--mode", "mixed", "--device", "cpu", "--record-metrics", "--out", run])
            self.assertTrue((run / "head_best.pt").exists())
            self.assertTrue((run / "finetune_best.pt").exists())
            import json
            records=[json.loads(s) for s in (run/"epoch_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(records),2)
            self.assertEqual(records[0]["fixed_thresholds"],[0.5]*5)
            self.assertEqual(records[1]["total_epoch"],2)
            common_args = ["--checkpoint", run / "best.pt", "--manifest", manifest_path, "--device", "cpu"]
            with patch.object(evaluate, "SoundDetector", TinyDetector):
                self.run_main(evaluate, common_args + ["--split", "val", "--tune", "--out", run / "val"])
                self.run_main(evaluate, common_args + ["--split", "test", "--thresholds", run / "val/thresholds.json",
                    "--rms-db", "-60", "--out", run / "test"])
                with self.assertRaises(SystemExit):
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.run_main(evaluate, common_args + ["--split", "test", "--tune", "--out", run / "illegal"])
            report = read_json(run / "test/report.json")
            self.assertGreater(report["ungated"]["metrics"]["n_windows"], 0)
            self.assertEqual(report["gated"]["timing"]["skipped_windows"], 1)
            self.assertEqual(report["gated"]["positive_windows_skipped"], 0)
            with patch.object(infer, "SoundDetector", TinyDetector):
                self.run_main(infer, ["--checkpoint", run / "best.pt", "--audio", root / "1-0.wav",
                    "--thresholds", run / "val/thresholds.json", "--device", "cpu", "--out", root / "infer"])
            self.assertEqual(len(read_json(root / "infer/result.json")["windows"]), 1)
            self.run_main(plot_results, ["--clean", run / "test/report.json", "--mixed", run / "test/report.json", "--out", root / "plots"])
            self.assertGreater((root / "plots/class_f1.png").stat().st_size, 1000)
            self.assertGreater((root / "plots/snr_f1.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
