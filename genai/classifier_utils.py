"""Training / evaluation of the 4-class corruption classifier (Task 2, reused as the Task 3 gate)."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .corruptions import CLASSES


@torch.no_grad()
def predict_classifier(model, loader, device):
    model.eval()
    probs, labels, levels = [], [], []
    for noisy, _clean, label, level, _idx in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(noisy.to(device, non_blocking=True))
        probs.append(F.softmax(logits.float(), 1).cpu())
        labels.append(label)
        levels.append(level)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(levels).numpy()


def classification_report_dict(probs, labels):
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(labels, pred, labels=range(4), zero_division=0)
    cm = confusion_matrix(labels, pred, labels=range(4))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    return {
        "accuracy": float((pred == labels).mean()),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion": cm.tolist(), "confusion_normalized": cm_norm.tolist(),
    }


def fit_classifier(model, train_loader, val_loader, *, lr, weight_decay, epochs, device, trial=None,
                   mlflow=None, label_smoothing=0.0):
    """Cross-entropy on runtime-corrupted, class-balanced batches; select by val macro-F1."""
    import optuna
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.15)
    from .train_utils import make_scaler
    scaler = make_scaler(device)
    best, best_state, history = None, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for noisy, _clean, label in train_loader:
            noisy, label = noisy.to(device, non_blocking=True), label.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(noisy)
            loss = F.cross_entropy(logits.float(), label, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(label)
            correct += (logits.argmax(1) == label).sum().item()
            n += len(label)
        probs, labels, _ = predict_classifier(model, val_loader, device)
        val_loss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-8)), torch.from_numpy(labels)))
        rep = classification_report_dict(probs, labels)
        rec = {"epoch": epoch, "train_loss": tot / n, "train_acc": correct / n, "val_loss": val_loss,
               "val_acc": rep["accuracy"], "val_macro_f1": rep["macro_f1"], "epoch_time": time.time() - t0}
        history.append(rec)
        print(f"  ep {epoch:3d} train {rec['train_loss']:.4f}/{rec['train_acc']:.3f} | val loss {val_loss:.4f} "
              f"acc {rep['accuracy']:.4f} F1 {rep['macro_f1']:.4f} ({rec['epoch_time']:.0f}s)")
        if mlflow is not None:
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
        if best is None or rep["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"], "val_loss": val_loss, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
        if trial is not None:
            trial.report(rep["macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    return best, history


def plot_confusion(cm_norm, path, title="Normalized confusion matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm_norm)
    names = ["clean", "salt&pepper", "blur", "occlusion"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(4), names, rotation=30)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_classifier_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(h.epoch, h.train_loss, label="train")
    axes[0].plot(h.epoch, h.val_loss, label="val")
    axes[0].set_title("cross-entropy")
    axes[0].legend()
    axes[1].plot(h.epoch, h.train_acc, label="train acc")
    axes[1].plot(h.epoch, h.val_acc, label="val acc")
    axes[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    axes[1].set_title("accuracy / F1")
    axes[1].legend()
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
