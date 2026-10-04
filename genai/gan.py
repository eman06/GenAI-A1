"""Task 4 - style-conditioned pix2pix-style cGAN (U-Net generator + PatchGAN discriminator)."""
import torch
import torch.nn as nn


def down(cin, cout, norm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(cin, cout, dropout=0.0):
    layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class StyleUNetGenerator(nn.Module):
    """U-Net 128 -> 2 -> 128 with skip connections.

    The style id is mapped through a learned nn.Embedding and used twice:
      1. tiled to an emb_dim x 128 x 128 map and concatenated with the photo at the input;
      2. projected and added to the 2x2 bottleneck features (global conditioning).
    Output: 1-channel sketch in [-1, 1] (tanh).
    """

    def __init__(self, base_ch=64, n_styles=3, emb_dim=16, dropout=0.5):
        super().__init__()
        b = base_ch
        self.embed = nn.Embedding(n_styles, emb_dim)
        chs = [b, b * 2, b * 4, b * 8, b * 8, b * 8]                      # 64,32,16,8,4,2
        self.downs = nn.ModuleList()
        cin = 3 + emb_dim
        for i, c in enumerate(chs):
            self.downs.append(down(cin, c, norm=i != 0))
            cin = c
        self.style_to_bottleneck = nn.Linear(emb_dim, chs[-1])
        ups_out = [b * 8, b * 8, b * 4, b * 2, b]                          # 4,8,16,32,64
        self.ups = nn.ModuleList()
        cin = chs[-1]
        for i, c in enumerate(ups_out):
            skip = chs[-2 - i]
            self.ups.append(up(cin if i == 0 else cin + skip_prev, c, dropout if i < 3 else 0.0))
            cin, skip_prev = c, skip
        self.final = nn.Sequential(nn.ConvTranspose2d(cin + chs[0], 1, 4, 2, 1), nn.Tanh())

    def forward(self, photo, style):
        e = self.embed(style)                                             # (B, E)
        x = torch.cat([photo, e[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])], 1)
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = x + self.style_to_bottleneck(e)[:, :, None, None]
        skips = skips[:-1][::-1]                                          # 4,8,16,32,64
        for i, u in enumerate(self.ups):
            x = u(x)
            x = torch.cat([x, skips[i]], 1)
        return self.final(x)


class StylePatchDiscriminator(nn.Module):
    """70x70-style PatchGAN on (photo, sketch, tiled style embedding); 128x128 input -> 14x14 logits."""

    def __init__(self, base_ch=64, n_styles=3, emb_dim=16):
        super().__init__()
        b = base_ch
        self.embed = nn.Embedding(n_styles, emb_dim)
        self.net = nn.Sequential(
            down(3 + 1 + emb_dim, b, norm=False), down(b, b * 2), down(b * 2, b * 4),
            nn.Conv2d(b * 4, b * 8, 4, 1, 1, bias=False), nn.BatchNorm2d(b * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b * 8, 1, 4, 1, 1))

    def forward(self, photo, sketch, style):
        e = self.embed(style)[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, e], 1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class GeneratorExport(nn.Module):
    """ONNX wrapper: photo in [0,1] (B,3,128,128) + style int64 (B,) -> sketch in [0,1] (B,1,128,128)."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, photo, style):
        return (self.g(photo * 2 - 1, style) + 1) / 2
