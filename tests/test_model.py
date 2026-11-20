import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from model import SoundDetector


class ModelTests(unittest.TestCase):
    def test_real_cnn14_forward_and_staged_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        model = SoundDetector()
        model.train()
        before_bn = model.backbone.conv_block6.bn1.running_mean.clone()
        wave = torch.randn(1, 32000) * 0.02
        with torch.no_grad():
            expected = model.head(model.backbone(wave)["embedding"])
            actual = model(wave)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        logits = model(wave)
        self.assertEqual(tuple(logits.shape), (1, 5))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNone(model.backbone.fc1.weight.grad)
        self.assertFalse(model.backbone.training)
        saved_head = model.head.weight.detach().clone()
        model.zero_grad(set_to_none=True)
        model.configure_stage("finetune")
        model.train()
        self.assertTrue(torch.equal(saved_head, model.head.weight))
        model(wave).sum().backward()
        self.assertIsNotNone(model.backbone.fc1.weight.grad)
        self.assertIsNotNone(model.backbone.conv_block6.conv1.weight.grad)
        self.assertIsNone(model.backbone.conv_block5.conv1.weight.grad)
        self.assertTrue(torch.equal(before_bn, model.backbone.conv_block6.bn1.running_mean))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for AMP regression test")
    def test_cuda_amp_silence_is_finite(self):
        model = SoundDetector().cuda().train()
        wave = torch.zeros(1, 32000, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(wave)
        self.assertTrue(torch.isfinite(logits).all())
        torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), torch.zeros_like(logits).float()).backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
