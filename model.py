"""Use the original author's Cnn14; retain checkpoint-compatible frontend."""
import importlib
import torch
from torch import nn


class SoundDetector(nn.Module):
    def __init__(self, spec_augment=None):
        super().__init__()
        try:
            Cnn14 = importlib.import_module("third_party.panns.models").Cnn14
        except ModuleNotFoundError as e:
            raise RuntimeError("Run python download_assets.py first; then install requirements.txt") from e
        self.backbone = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                              mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.head = nn.Linear(2048, 5)
        # Optional training-only SpecAugment. It lives outside the backbone so the
        # backbone (BatchNorm statistics, Dropout) can stay in eval mode, and it has
        # no parameters, so checkpoints stay compatible with/without it.
        self.spec_augmenter = None
        if spec_augment and spec_augment.get("enabled"):
            from torchlibrosa.augmentation import SpecAugmentation
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=int(spec_augment["time_drop_width"]),
                time_stripes_num=int(spec_augment["time_stripes_num"]),
                freq_drop_width=int(spec_augment["freq_drop_width"]),
                freq_stripes_num=int(spec_augment["freq_stripes_num"]))
        self.stage = "head"
        self.configure_stage("head")

    def load_pretrained(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state["model"], strict=True)

    def configure_stage(self, stage):
        if stage not in ("head", "finetune"):
            raise ValueError(stage)
        self.stage = stage
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(True)
        if stage == "finetune":
            self.backbone.conv_block6.requires_grad_(True)
            self.backbone.fc1.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep original functional Dropout/SpecAugment OFF and all BN statistics fixed.
        # (self.spec_augmenter, if configured, follows `mode` via super().train.)
        # eval() does not disable autograd: block6/fc1 still learn in stage 2.
        self.backbone.eval()
        return self

    def forward(self, waveform):
        # The original log-mel frontend can underflow its 1e-10 floor under
        # float16 autocast, yielding -inf on quiet bins. Keep that frontend FP32.
        # The following is Cnn14's eval-mode embedding path: same modules,
        # pooling and weights, with Dropout/SpecAugment intentionally disabled.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            x = self.backbone.spectrogram_extractor(waveform.float())
            x = self.backbone.logmel_extractor(x)
            x = x.transpose(1, 3)
            x = self.backbone.bn0(x)
            x = x.transpose(1, 3)
            if self.spec_augmenter is not None and self.training:
                # Same place as the original Cnn14: after bn0, before conv blocks.
                x = self.spec_augmenter(x.contiguous())
        for index in range(1, 7):
            pool = (2, 2) if index < 6 else (1, 1)
            x = getattr(self.backbone, f"conv_block{index}")(x, pool_size=pool, pool_type="avg")
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        embedding = torch.relu(self.backbone.fc1(x))
        return self.head(embedding)
