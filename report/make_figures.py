"""Extra report figures built from the real result files and test samples.

    python report/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genai.corruptions import CLASSES, SEVERITY_NAMES, apply_corruption, make_level_spec  # noqa: E402

R = "drive_results/outputs/results"
T4 = "drive_results/t4/outputs/results/task4"
S = "backend/app/samples"
F = "report/figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated categorical palette (fixed order)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PRETTY = {"clean": "Clean", "salt_pepper": "Salt & pepper", "blur": "Gaussian blur", "occlusion": "Occlusion"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def style_axes(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# 1. dataset samples ----------------------------------------------------------------
pets = sorted(f for f in os.listdir(S) if f.startswith("pet"))
faces = sorted(f for f in os.listdir(S) if f.startswith("face"))
fig, axes = plt.subplots(1, len(pets), figsize=(1.35 * len(pets), 1.6))
for ax, f in zip(axes, pets):
    ax.imshow(Image.open(os.path.join(S, f)))
    ax.set_title(f.replace("pet_test_", "test #").replace(".png", ""), fontsize=7, color=MUTED)
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{F}/data_pet_samples.png", dpi=200)
plt.close(fig)

# 2. corruption gallery: every corruption x every fixed test severity -----------------
img = np.asarray(Image.open(os.path.join(S, "pet_test_0733.png")).convert("RGB"), dtype=np.float32) / 255.0
labels = {"salt_pepper": ["p = 0.03", "p = 0.08", "p = 0.15"],
          "blur": ["k=3, σ=0.7", "k=5, σ=1.5", "k=7, σ=2.5"],
          "occlusion": ["1 rect ≈10%", "2 rects ≈20%", "3 rects ≈35%"]}
fig, axes = plt.subplots(3, 4, figsize=(6.4, 5.0))
for r, ctype in enumerate(["salt_pepper", "blur", "occlusion"]):
    axes[r, 0].imshow(img)
    axes[r, 0].set_title("clean" if r == 0 else "", fontsize=8)
    axes[r, 0].set_ylabel(PRETTY[ctype], fontsize=8)
    for lvl in range(3):
        spec = make_level_spec(np.random.default_rng(2_000_003 + 733 * 100 + (r + 1) * 10 + lvl), ctype, lvl)
        out = apply_corruption(img, spec)
        title = f"{SEVERITY_NAMES[lvl]}: {labels[ctype][lvl]}"
        if ctype == "occlusion":
            title += f" ({spec['coverage'] * 100:.1f}%)"
        axes[r, lvl + 1].imshow(np.clip(out, 0, 1))
        axes[r, lvl + 1].set_title(title, fontsize=7)
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
fig.tight_layout()
fig.savefig(f"{F}/data_corruption_gallery.png", dpi=200)
plt.close(fig)

# 3. system comparison: PSNR and SSIM by input condition (grouped bars, labelled) ----
cmp = pd.read_csv(f"{R}/task3/comparison_tasks123.csv", index_col=0)
order = ["Universal DAE (T1)", "Hard routing, predicted (T2)", "Hard routing, oracle (T2)", "Soft MoE (T3)"]
short = ["Universal DAE", "Hard (predicted)", "Hard (oracle)", "Soft MoE"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, metric, unit in [(axes[0], "PSNR", "dB"), (axes[1], "SSIM", "")]:
    x = np.arange(4)
    w = 0.2
    for k, (sysname, lab) in enumerate(zip(order, short)):
        vals = [cmp.loc[sysname, f"{c} {metric}"] for c in CLASSES]
        bars = ax.bar(x + (k - 1.5) * w, vals, w - 0.02, color=SERIES[k], label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}" if metric == "PSNR" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=5.2, color=INK, rotation=90)
    ax.set_xticks(x, ["Clean", "S&P", "Blur", "Occlusion"])
    ax.set_ylabel(f"{metric} {('(' + unit + ')') if unit else ''}".strip())
    ax.set_ylim(0, 58 if metric == "PSNR" else 1.18)
    style_axes(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, ncol=4, loc="lower center")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f"{F}/cmp_systems_by_type.png", dpi=200)
plt.close(fig)

# 4. severity curves: PSNR vs severity per corruption, one panel per corruption -------
files = {"Universal DAE": f"{R}/task1/test_final_by_type_severity.csv",
         "Hard (predicted)": f"{R}/task2/hard_predicted_by_type_severity.csv",
         "Hard (oracle)": f"{R}/task2/hard_oracle_by_type_severity.csv",
         "Soft MoE": f"{R}/task3/moe_test_by_type_severity.csv"}
tabs = {k: pd.read_csv(v) for k, v in files.items()}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
for ax, ctype in zip(axes, ["salt_pepper", "blur", "occlusion"]):
    inp = tabs["Soft MoE"][tabs["Soft MoE"].type == ctype].set_index("severity").loc[SEVERITY_NAMES, "in_psnr"]
    ax.plot(range(3), inp.values, color=MUTED, lw=1.5, ls="--", marker="o", ms=4, label="Corrupted input")
    for k, (name, d) in enumerate(tabs.items()):
        v = d[d.type == ctype].set_index("severity").loc[SEVERITY_NAMES, "psnr"]
        ax.plot(range(3), v.values, color=SERIES[k], lw=2, marker="osD^"[k], ms=5, label=name, alpha=0.95)
    ax.set_xticks(range(3), SEVERITY_NAMES)
    ax.set_title(PRETTY[ctype], fontsize=9)
    style_axes(ax)
axes[0].set_ylabel("PSNR (dB)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, ncol=5, loc="lower center")
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(f"{F}/cmp_psnr_by_severity.png", dpi=200)
plt.close(fig)

# 5. three specialist curves in one figure ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
for k, e in enumerate(["salt_pepper", "blur", "occlusion"]):
    h = pd.DataFrame(json.load(open(f"{R}/task2/specialist_{e}_history.json")))
    axes[0].plot(h.epoch, h.train_loss, color=SERIES[k], lw=2, label=PRETTY[e])
    axes[1].plot(h.epoch, h.val_ssim, color=SERIES[k], lw=2, label=PRETTY[e])
axes[0].set_title("training loss (α·L1 + (1−α)(1−SSIM))", fontsize=8)
axes[1].set_title("validation SSIM (own corruption only)", fontsize=8)
for ax in axes:
    ax.set_xlabel("epoch")
    style_axes(ax)
axes[1].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(f"{F}/t2_specialists_curves.png", dpi=200)
plt.close(fig)

# 6. FS2K style distribution per split ------------------------------------------------
split = json.load(open("manifests/fs2k_split.json"))
cnt = split["style_counts"]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
x = np.arange(3)
for k, part in enumerate(["train", "val", "test"]):
    bars = ax.bar(x + (k - 1) * 0.27, cnt[part], 0.25, color=SERIES[k], label=part)
    for b, v in zip(bars, cnt[part]):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(x, ["Style 1", "Style 2", "Style 3"])
ax.set_ylabel("image pairs")
ax.legend(fontsize=7, frameon=False)
style_axes(ax)
fig.tight_layout()
fig.savefig(f"{F}/t4_style_distribution.png", dpi=200)
plt.close(fig)

# 7. face samples (model inputs used in the app) -----------------------------------
fig, axes = plt.subplots(1, len(faces), figsize=(1.35 * len(faces), 1.6))
for ax, f in zip(axes, faces):
    ax.imshow(Image.open(os.path.join(S, f)))
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{F}/data_face_samples.png", dpi=200)
plt.close(fig)
print("figures written")
