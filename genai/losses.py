"""Reconstruction losses and image-quality metrics."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


def ssim_per_image(pred, target):
    return ssim(pred.float(), target.float(), data_range=1.0, size_average=False)


def psnr_per_image(pred, target, eps=1e-10):
    mse = F.mse_loss(pred.float(), target.float(), reduction="none").flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def l1_per_image(pred, target):
    return (pred.float() - target.float()).abs().flatten(1).mean(1)


class L1SSIMLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - SSIM)  -- the assignment's L_UDAE."""

    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        l1 = F.l1_loss(pred, target)
        s = ssim(pred, target, data_range=1.0, size_average=True)
        return self.alpha * l1 + (1 - self.alpha) * (1 - s), {"l1": l1.item(), "ssim": s.item()}


def selection_objective(l1, ssim_value):
    """Alpha-independent validation objective used for Optuna and checkpoint selection.

    Equal-weighted L1 + (1 - SSIM). It must NOT depend on alpha, otherwise trials with
    different alphas would be scored on different scales and the search would favour
    whichever alpha makes its own loss smallest.
    """
    return l1 + (1.0 - ssim_value)
