"""Network definitions for Tasks 1-3."""
import torch
import torch.nn as nn


def conv_bn_act(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def up_bn_act(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvAutoencoder(nn.Module):
    """Convolutional denoising autoencoder with a spatial bottleneck.

    128x128x3 -> 64 -> 32 -> 16x16xlatent_ch -> 32 -> 64 -> 128x128x3.
    Channels grow base, 2*base, 4*base while resolution halves. The latent is a
    16x16xlatent_ch tensor (latent_ch in {4..32} -> 12x-48x compression of the
    49,152 input values). No skip connections by default, so all information must
    pass through the bottleneck; `skip=True` adds ONE skip at 64x64 for the ablation.
    """

    def __init__(self, base_ch=32, latent_ch=16, dropout=0.0, skip=False):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        drop = (lambda: nn.Dropout2d(dropout)) if dropout > 0 else (lambda: nn.Identity())
        self.skip = skip
        self.enc1 = nn.Sequential(conv_bn_act(3, c1), conv_bn_act(c1, c1, 2), conv_bn_act(c1, c1))   # 64
        self.enc2 = nn.Sequential(conv_bn_act(c1, c2, 2), conv_bn_act(c2, c2), drop())               # 32
        self.enc3 = nn.Sequential(conv_bn_act(c2, c3, 2), conv_bn_act(c3, c3), drop())               # 16
        self.to_latent = nn.Conv2d(c3, latent_ch, 1)
        self.from_latent = nn.Sequential(conv_bn_act(latent_ch, c3), drop())
        self.dec3 = nn.Sequential(up_bn_act(c3, c2), conv_bn_act(c2, c2))                             # 32
        self.dec2 = nn.Sequential(up_bn_act(c2, c1), conv_bn_act(c1, c1))                             # 64
        dec1_in = c1 * 2 if skip else c1
        self.dec1 = nn.Sequential(conv_bn_act(dec1_in, c1), up_bn_act(c1, c1), conv_bn_act(c1, c1))   # 128
        self.head = nn.Conv2d(c1, 3, 3, 1, 1)
        self.latent_ch = latent_ch

    def encode(self, x):
        e1 = self.enc1(x)
        return self.to_latent(self.enc3(self.enc2(e1))), e1

    def decode(self, z, e1=None):
        d = self.dec2(self.dec3(self.from_latent(z)))
        if self.skip:
            d = torch.cat([d, e1], dim=1)
        return torch.sigmoid(self.head(self.dec1(d)))

    def forward(self, x):
        z, e1 = self.encode(x)
        return self.decode(z, e1)

    def bottleneck_size(self, img_size=128):
        return self.latent_ch * (img_size // 8) ** 2


class CorruptionClassifier(nn.Module):
    """VGG-style CNN predicting clean / salt_pepper / blur / occlusion logits."""

    def __init__(self, channels=(32, 64, 128, 256), dropout=0.3, n_classes=4):
        super().__init__()
        layers, cin = [], 3
        for c in channels:
            layers += [conv_bn_act(cin, c), conv_bn_act(c, c), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_autoencoder(cfg):
    return ConvAutoencoder(cfg["base_ch"], cfg["latent_ch"], cfg.get("dropout", 0.0), cfg.get("skip", False))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
