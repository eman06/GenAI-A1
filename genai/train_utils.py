"""Shared training / evaluation loops for the restoration autoencoders (Tasks 1-2)."""
import copy
import time

import numpy as np
import torch
import torchvision.utils as vutils

from .losses import L1SSIMLoss, l1_per_image, psnr_per_image, selection_objective, ssim_per_image


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(device):
    enabled = device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):        # torch >= 2.3
        return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate_restorer(model, loader, device):
    """Mean L1 / SSIM / PSNR of model(noisy) vs clean over a deterministic loader."""
    model.eval()
    l1s, ssims, psnrs = [], [], []
    for noisy, clean, *_ in loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(noisy)
        out = out.float().clamp(0, 1)
        l1s.append(l1_per_image(out, clean).cpu())
        ssims.append(ssim_per_image(out, clean).cpu())
        psnrs.append(psnr_per_image(out, clean).cpu())
    l1, s, p = (torch.cat(v).mean().item() for v in (l1s, ssims, psnrs))
    return {"l1": l1, "ssim": s, "psnr": p, "objective": selection_objective(l1, s)}


def sample_grid(model, noisy, clean, device):
    """Rows of (noisy, restored, clean) as an HWC uint8 array for MLflow."""
    model.eval()
    with torch.no_grad():
        out = model(noisy.to(device)).float().clamp(0, 1).cpu()
    rows = torch.cat([noisy, out, clean], dim=0)
    grid = vutils.make_grid(rows, nrow=noisy.shape[0], padding=2)
    return (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def fit_restorer(model, train_loader, val_loader, *, alpha, lr, epochs, device, weight_decay=1e-5,
                 trial=None, mlflow=None, metric_prefix="", fixed_batch=None, log_every=5):
    """Train with L_UDAE, keep the weights with the best alpha-independent val objective.

    If `trial` is given, the val objective is reported every epoch so Optuna's pruner can
    stop unpromising trials early.
    """
    import optuna
    loss_fn = L1SSIMLoss(alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_loader),
                                                pct_start=0.15)
    scaler = make_scaler(device)
    best, best_state, history = None, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for noisy, clean, _ in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(noisy)
            loss, _ = loss_fn(out, clean)          # loss computed in fp32 (SSIM is unstable in fp16)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * noisy.shape[0]
            n += noisy.shape[0]
        val = evaluate_restorer(model, val_loader, device)
        rec = {"epoch": epoch, "train_loss": tot / n, **{f"val_{k}": v for k, v in val.items()},
               "lr": sched.get_last_lr()[0], "epoch_time": time.time() - t0}
        history.append(rec)
        print(f"  ep {epoch:3d} train {rec['train_loss']:.4f} | val L1 {val['l1']:.4f} "
              f"SSIM {val['ssim']:.4f} PSNR {val['psnr']:.2f} obj {val['objective']:.4f} ({rec['epoch_time']:.0f}s)")
        if mlflow is not None:
            mlflow.log_metrics({metric_prefix + k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
            if fixed_batch is not None and (epoch % log_every == 0 or epoch == epochs):
                mlflow.log_image(sample_grid(model, *fixed_batch, device),
                                 f"{metric_prefix}samples/epoch_{epoch:03d}.png")
        if best is None or val["objective"] < best["objective"]:
            best = {**val, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
        if trial is not None:
            trial.report(val["objective"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    return best, history


def fixed_val_batch(val_loader, n=8):
    """The same n validation images every time, one of each condition where possible."""
    ds = val_loader.dataset
    picks, seen = [], {}
    for i, e in enumerate(ds.entries):
        if seen.get(e["type"], 0) < max(1, n // 4):
            picks.append(i)
            seen[e["type"]] = seen.get(e["type"], 0) + 1
        if len(picks) == n:
            break
    items = [ds[i] for i in picks]
    return torch.stack([it[0] for it in items]), torch.stack([it[1] for it in items])
