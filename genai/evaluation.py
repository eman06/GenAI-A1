"""Test-set evaluation and report figures shared by Tasks 1-3."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .corruptions import SEVERITY_NAMES  # noqa: E402
from .losses import l1_per_image, psnr_per_image, ssim_per_image  # noqa: E402

PRETTY = {"clean": "Clean", "salt_pepper": "Salt & pepper", "blur": "Gaussian blur", "occlusion": "Occlusion"}


@torch.no_grad()
def evaluate_on_manifest(predict_fn, loader, device):
    """predict_fn(noisy, label) -> (restored, extras: dict[str, Tensor(B,...)]).

    Returns a per-sample DataFrame with input-vs-clean and output-vs-clean metrics plus any
    extras (e.g. routing weights), so every later table is a groupby on the same data.
    """
    rows = []
    entries = loader.dataset.entries
    for noisy, clean, label, level, idx in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out, extras = predict_fn(noisy, label.to(device))
        out = out.float().clamp(0, 1)
        m = {
            "in_l1": l1_per_image(noisy, clean), "in_ssim": ssim_per_image(noisy, clean),
            "in_psnr": psnr_per_image(noisy, clean),
            "l1": l1_per_image(out, clean), "ssim": ssim_per_image(out, clean), "psnr": psnr_per_image(out, clean),
        }
        m = {k: v.cpu().numpy() for k, v in m.items()}
        ex = {k: v.cpu().numpy() for k, v in (extras or {}).items()}
        for j in range(noisy.shape[0]):
            e = entries[int(idx[j])]
            r = {"entry": int(idx[j]), "image": e["index"], "type": e["type"], "level": int(level[j]),
                 "label": int(label[j]), **{k: float(v[j]) for k, v in m.items()}}
            for k, v in ex.items():
                if v.ndim == 1:
                    r[k] = float(v[j]) if v.dtype.kind == "f" else int(v[j])
                else:
                    for c in range(v.shape[1]):
                        r[f"{k}_{c}"] = float(v[j, c])
            rows.append(r)
    df = pd.DataFrame(rows)
    df["severity"] = df["level"].map(lambda l: "none" if l < 0 else SEVERITY_NAMES[l])
    return df


def summary_tables(df, out_dir, prefix):
    """By corruption type, and by type x severity; CSV + LaTeX."""
    cols = ["in_psnr", "psnr", "in_ssim", "ssim", "l1"]
    by_type = df.groupby("type", sort=False)[cols].mean()
    by_type.loc["overall"] = df[cols].mean()
    by_sev = df.groupby(["type", "severity"], sort=False)[cols].mean()
    by_type.to_csv(os.path.join(out_dir, f"{prefix}_by_type.csv"))
    by_sev.to_csv(os.path.join(out_dir, f"{prefix}_by_type_severity.csv"))
    for name, t in (("by_type", by_type), ("by_type_severity", by_sev)):
        with open(os.path.join(out_dir, f"{prefix}_{name}.tex"), "w") as f:
            f.write(t.to_latex(float_format="%.3f"))
    print(by_type.round(4).to_string())
    print(by_sev.round(4).to_string())
    return by_type, by_sev


def _img(t):
    return np.clip(t.permute(1, 2, 0).cpu().numpy(), 0, 1)


def restoration_grid(dataset, entry_ids, predict_fn, device, path, titles=None, extra_text=None):
    """Rows: clean | corrupted | restored | |error| map."""
    n = len(entry_ids)
    fig, axes = plt.subplots(n, 4, figsize=(8.4, 2.2 * n))
    axes = np.atleast_2d(axes)
    for r, eid in enumerate(entry_ids):
        noisy, clean, label, *_ = dataset[eid]
        with torch.no_grad():
            out, _ = predict_fn(noisy.unsqueeze(0).to(device), torch.tensor([label], device=device))
        out = out[0].float().clamp(0, 1).cpu()
        err = (out - clean).abs().mean(0).numpy()
        e = dataset.entries[eid]
        label = titles[r] if titles else f"{PRETTY[e['type']]}" + (f" ({SEVERITY_NAMES[e['level']]})" if e.get("level", -1) >= 0 else "")
        for c, (im, t) in enumerate([(_img(clean), "clean target"), (_img(noisy), label),
                                     (_img(out), "restored"), (err, "|error|")]):
            ax = axes[r, c]
            if c == 3:
                ax.imshow(im, cmap="inferno", vmin=0, vmax=0.5)
            else:
                ax.imshow(im)
            ax.set_title(t if r == 0 or c == 1 else "", fontsize=8)
            ax.axis("off")
        if extra_text:
            axes[r, 2].text(2, 124, extra_text[r], fontsize=6, color="yellow", backgroundcolor=(0, 0, 0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_examples(df):
    """12 representative rows: 3 clean + every corruption at every severity, near-median SSIM."""
    picks = []
    clean = df[df.type == "clean"]
    med = clean.ssim.median()
    picks += clean.iloc[(clean.ssim - med).abs().argsort()[:3]].entry.tolist()
    for t in ("salt_pepper", "blur", "occlusion"):
        for lvl in range(3):
            g = df[(df.type == t) & (df.level == lvl)]
            picks.append(int(g.iloc[(g.ssim - g.ssim.median()).abs().argsort()[:1]].entry.iloc[0]))
    return picks


def pick_failures(df, per_type=1):
    """Worst output SSIM for each input condition."""
    out = []
    for t in ("clean", "salt_pepper", "blur", "occlusion"):
        out += df[df.type == t].nsmallest(per_type, "ssim").entry.tolist()
    return out


def bar_by_severity(by_sev, path, metric="psnr", title=""):
    t = by_sev.reset_index()
    t = t[t.type != "clean"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    types = ["salt_pepper", "blur", "occlusion"]
    x = np.arange(len(types))
    for i, sev in enumerate(SEVERITY_NAMES):
        vals = [t[(t.type == ty) & (t.severity == sev)][metric].mean() for ty in types]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=sev)
    ax.set_xticks(x, [PRETTY[ty] for ty in types])
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(title="severity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path, title):
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(h.epoch, h.train_loss, label="train loss")
    axes[0].set_title("train loss (L_UDAE)")
    axes[1].plot(h.epoch, h.val_ssim, color="tab:green")
    axes[1].set_title("val SSIM")
    axes[2].plot(h.epoch, h.val_psnr, color="tab:red")
    axes[2].set_title("val PSNR (dB)")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
