"""Task 3 - jointly trained soft mixture-of-experts restoration.

Gate initialised from the Task 2 classifier, experts from the Task 2 specialists, plus an identity
branch. Stage 1 (warm-up): experts frozen, only the gate trains. Stage 2: everything is fine-tuned
jointly with a smaller learning rate on
    L = l1 * L1 + ls * (1 - SSIM) + lc * CE(gate logits, corruption label) + lb * L_balance,
    L_balance = sum_k (mean_batch(w_k) - 1/4)^2      (batches are exactly class-balanced).

    python scripts/task3_moe.py optuna --trials 8
    python scripts/task3_moe.py train
    python scripts/task3_moe.py eval
    python scripts/task3_moe.py export
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pytorch_msssim import ssim  # noqa: E402

from genai import paths  # noqa: E402
from genai.corruptions import CLASSES, SEVERITY_NAMES, apply_corruption  # noqa: E402
from genai.data import PetData, to_chw  # noqa: E402
from genai.evaluation import (PRETTY, evaluate_on_manifest, pick_failures, restoration_grid,  # noqa: E402
                              summary_tables)
from genai.losses import l1_per_image, psnr_per_image, selection_objective, ssim_per_image  # noqa: E402
from genai.routing import HardRoutedRestorer, SoftMoEExport, SoftMoERestorer  # noqa: E402
from genai.train_utils import get_device, make_scaler, set_seed  # noqa: E402
from task2_classifier import load_classifier  # noqa: E402
from task2_specialists import EXPERTS, load_specialists  # noqa: E402

STUDY = "task3_moe"
OUT = os.path.join(paths.RESULTS_DIR, "task3")
CKPT = os.path.join(paths.CKPT_DIR, "task3_moe.pt")
BRANCHES = ["identity", "salt_pepper", "blur", "occlusion"]
DEFAULT = {"lr_joint": 1e-4, "tau": 1.0, "lambda_c": 0.1, "lambda_b": 0.01, "lambda_1": 0.8,
           "lr_gate": 1e-3, "batch_size": 32}


def build_moe(device, tau):
    specs = load_specialists(device)
    return SoftMoERestorer(load_classifier(device), [specs[e] for e in EXPERTS], tau=tau).to(device)


def balance_loss(w):
    return ((w.mean(0) - 1.0 / w.shape[1]) ** 2).sum()


@torch.no_grad()
def evaluate_moe(moe, loader, device):
    moe.eval()
    l1s, ss, ps, ws, labels = [], [], [], [], []
    for noisy, clean, label, _lvl, _i in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out, w, _ = moe(noisy)
        out = out.clamp(0, 1)
        l1s.append(l1_per_image(out, clean).cpu())
        ss.append(ssim_per_image(out, clean).cpu())
        ps.append(psnr_per_image(out, clean).cpu())
        ws.append(w.cpu())
        labels.append(label)
    w, labels = torch.cat(ws), torch.cat(labels)
    l1, s, p = (torch.cat(v).mean().item() for v in (l1s, ss, ps))
    mean_w = w.mean(0)
    per_class = torch.stack([w[labels == k].mean(0) for k in range(4)])     # (true class, branch)
    return {"l1": l1, "ssim": s, "psnr": p, "objective": selection_objective(l1, s),
            "route_acc": (w.argmax(1) == labels).float().mean().item(),
            "mean_w": mean_w.tolist(), "min_branch_w": mean_w.min().item(), "w_by_class": per_class.tolist()}


def set_experts_trainable(moe, flag):
    for e in moe.experts:
        for p in e.parameters():
            p.requires_grad = flag


def run_stage(moe, train_loader, val_loader, device, cfg, epochs, lr, params, stage, trial=None,
              mlflow=None, step0=0, best=None):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(train_loader)))
    scaler = make_scaler(device)
    lam1, lams = cfg["lambda_1"], 1.0 - cfg["lambda_1"]
    history = []
    for ep in range(1, epochs + 1):
        moe.train()
        for e in moe.experts:      # keep the specialists' BatchNorm statistics fixed during fine-tuning
            e.eval()
        t0, sums, n = time.time(), np.zeros(5), 0
        for noisy, clean, label in train_loader:
            noisy, clean, label = noisy.to(device), clean.to(device), label.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out, w, logits = moe(noisy)
            out = out.float()
            l1 = F.l1_loss(out, clean)
            s = ssim(out.clamp(0, 1), clean, data_range=1.0, size_average=True)
            ce = F.cross_entropy(logits.float(), label)
            bal = balance_loss(w.float())
            loss = lam1 * l1 + lams * (1 - s) + cfg["lambda_c"] * ce + cfg["lambda_b"] * bal
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += np.array([loss.item(), l1.item(), s.item(), ce.item(), bal.item()]) * len(label)
            n += len(label)
        val = evaluate_moe(moe, val_loader, device)
        tr = dict(zip(["loss", "l1", "ssim", "ce", "balance"], sums / n))
        step = step0 + ep
        rec = {"stage": stage, "step": step, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in val.items() if not isinstance(v, list)},
               **{f"val_w_{b}": val["mean_w"][i] for i, b in enumerate(BRANCHES)}, "epoch_time": time.time() - t0}
        history.append(rec)
        print(f"  [{stage}] ep {ep} loss {tr['loss']:.4f} ce {tr['ce']:.3f} bal {tr['balance']:.4f} | val SSIM "
              f"{val['ssim']:.4f} PSNR {val['psnr']:.2f} route_acc {val['route_acc']:.3f} "
              f"w={np.round(val['mean_w'], 3).tolist()} ({rec['epoch_time']:.0f}s)")
        if mlflow is not None:
            mlflow.log_metrics({k: v for k, v in rec.items() if isinstance(v, float)}, step=step)
        if best is None or val["objective"] < best["val"]["objective"]:
            best = {"val": val, "state": copy.deepcopy(moe.state_dict()), "step": step}
        if trial is not None:
            trial.report(val["objective"], step)
            if val["min_branch_w"] < 0.02:            # routing collapse: a branch is (almost) never used
                raise optuna.TrialPruned(f"routing collapse, mean weights {val['mean_w']}")
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, history


def train_moe(cfg, data, device, warmup_epochs, joint_epochs, workers, trial=None, mlflow=None):
    set_seed(42)
    moe = build_moe(device, cfg["tau"])
    tl = data.train_loader(cfg["batch_size"], mode="balanced", num_workers=workers)
    vl = data.val_loader(num_workers=workers)
    init = evaluate_moe(moe, vl, device)
    print(f"  init (classifier gate + specialists, tau={cfg['tau']}): SSIM {init['ssim']:.4f} PSNR {init['psnr']:.2f}")
    set_experts_trainable(moe, False)
    best, h1 = run_stage(moe, tl, vl, device, cfg, warmup_epochs, cfg["lr_gate"], list(moe.gate.parameters()),
                         "warmup", trial, mlflow)
    set_experts_trainable(moe, True)
    best, h2 = run_stage(moe, tl, vl, device, cfg, joint_epochs, cfg["lr_joint"], list(moe.parameters()),
                         "joint", trial, mlflow, step0=warmup_epochs, best=best)
    moe.load_state_dict(best["state"])
    return moe, best, init, h1 + h2


def stage_optuna(args, data, device, mlflow):
    def objective(trial):
        cfg = dict(DEFAULT)
        cfg.update({
            "lr_joint": trial.suggest_float("lr_joint", 1e-5, 3e-4, log=True),
            "tau": trial.suggest_float("tau", 0.5, 2.0),
            "lambda_c": trial.suggest_float("lambda_c", 0.01, 0.5, log=True),
            "lambda_b": trial.suggest_float("lambda_b", 1e-3, 0.1, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 0.5, 0.95),
        })
        with mlflow.start_run(run_name=f"moe_trial_{trial.number:03d}", nested=True):
            mlflow.log_params({**cfg, "trial": trial.number})
            try:
                _, best, _, _ = train_moe(cfg, data, device, args.warmup, args.epochs, args.workers, trial, mlflow)
            except optuna.TrialPruned as e:
                mlflow.set_tag("state", f"pruned: {e}")
                raise
            mlflow.log_metrics({"best_val_objective": best["val"]["objective"], "best_val_ssim": best["val"]["ssim"]})
        trial.set_user_attr("val_ssim", best["val"]["ssim"])
        trial.set_user_attr("val_psnr", best["val"]["psnr"])
        trial.set_user_attr("mean_w", best["val"]["mean_w"])
        return best["val"]["objective"]

    study = optuna.create_study(study_name=STUDY, storage=paths.optuna_storage(STUDY), load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=args.warmup))
    if not study.trials:
        study.enqueue_trial({k: DEFAULT[k] for k in ("lr_joint", "tau", "lambda_c", "lambda_b", "lambda_1")})
    done = len([t for t in study.trials if t.state.is_finished()])
    with mlflow.start_run(run_name="task3_moe_optuna"):
        study.optimize(objective, n_trials=max(0, args.trials - done), gc_after_trial=True)
        study.trials_dataframe().to_csv(os.path.join(OUT, "moe_optuna_trials.csv"), index=False)
        best = {"trial": study.best_trial.number, "val_objective": study.best_value, **study.best_params,
                **study.best_trial.user_attrs,
                "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
                "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials)}
        json.dump(best, open(os.path.join(OUT, "moe_best_params.json"), "w"), indent=2)
        print("BEST:", best)
        from task1_optuna import save_optuna_plots
        tmp = os.path.join(OUT, "_plots")
        os.makedirs(tmp, exist_ok=True)
        save_optuna_plots(study, tmp)
        for f in os.listdir(tmp):
            os.replace(os.path.join(tmp, f), os.path.join(OUT, "moe_" + f))
        os.rmdir(tmp)
        mlflow.log_artifacts(OUT, "task3")


def load_cfg():
    cfg = dict(DEFAULT)
    p = os.path.join(OUT, "moe_best_params.json")
    if os.path.exists(p):
        cfg.update({k: v for k, v in json.load(open(p)).items() if k in DEFAULT and k != "batch_size"})
    return cfg


def stage_train(args, data, device, mlflow):
    cfg = load_cfg()
    print("config:", cfg)
    with mlflow.start_run(run_name="task3_moe_final"):
        mlflow.log_params({**cfg, "warmup_epochs": args.warmup, "joint_epochs": args.epochs})
        moe, best, init, hist = train_moe(cfg, data, device, args.warmup, args.epochs, args.workers, mlflow=mlflow)
        torch.save({"state_dict": moe.state_dict(), "cfg": cfg, "best_val": best["val"], "init_val": init}, CKPT)
        json.dump(hist, open(os.path.join(OUT, "moe_history.json"), "w"), indent=2)
        plot_moe_history(hist, os.path.join(OUT, "moe_curves.png"), args.warmup)
        mlflow.log_artifact(os.path.join(OUT, "moe_curves.png"))
        mlflow.log_artifact(CKPT, "checkpoints")
    print("init val:", {k: v for k, v in init.items() if k != "w_by_class"})
    print("best val:", {k: v for k, v in best["val"].items() if k != "w_by_class"})


def plot_moe_history(hist, path, warmup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(hist)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(h.step, h.train_loss, label="total")
    axes[0].plot(h.step, h.train_ce, label="CE")
    axes[0].plot(h.step, h.train_balance * 10, label="balance x10")
    axes[0].set_title("training losses")
    axes[0].legend(fontsize=7)
    axes[1].plot(h.step, h.val_ssim, color="tab:green")
    axes[1].set_title("val SSIM")
    for b in BRANCHES:
        axes[2].plot(h.step, h[f"val_w_{b}"], label=b)
    axes[2].set_title("mean val routing weight")
    axes[2].legend(fontsize=7)
    for a in axes:
        a.axvline(warmup + 0.5, color="grey", ls="--", lw=0.8)
        a.set_xlabel("epoch (dashed: end of gate warm-up)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_moe(device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    moe = build_moe(device, ck["cfg"]["tau"])
    moe.load_state_dict(ck["state_dict"])
    return moe.eval()


def routing_heatmap(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, labels = [], []
    for t in CLASSES:
        sevs = ["none"] if t == "clean" else SEVERITY_NAMES
        for s in sevs:
            g = df[(df.type == t) & (df.severity == s)]
            rows.append([g[f"w_{k}"].mean() for k in range(4)])
            labels.append(PRETTY[t] + ("" if s == "none" else f" / {s}"))
    m = np.array(rows)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="white" if m[i, j] < 0.6 else "black", fontsize=8)
    ax.set_xticks(range(4), ["identity", "s&p expert", "blur expert", "occl. expert"], rotation=20)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Mean soft-MoE routing weight (test)", fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(m, index=labels, columns=BRANCHES)


def weight_distribution_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8), sharey=True)
    for ax, t in zip(axes, CLASSES):
        g = df[df.type == t]
        ax.boxplot([g[f"w_{k}"] for k in range(4)], showfliers=False)
        ax.set_xticks(range(1, 5), ["id", "s&p", "blur", "occ"])
        ax.set_title(f"true: {PRETTY[t]}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("routing weight")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def mixed_corruption_experiment(moe, router, data, device, n=500):
    """Images with TWO corruptions (blur k5 s1.5 then salt&pepper p0.08) - outside the training
    distribution and exactly the case where hard routing must pick a single expert."""
    n = min(n, len(data.test))
    rows = []
    for i in range(n):
        clean = data.test[i].astype(np.float32) / 255.0
        noisy = apply_corruption(apply_corruption(clean, {"type": "blur", "kernel": 5, "sigma": 1.5}),
                                 {"type": "salt_pepper", "prob": 0.08, "seed": 777 + i})
        rows.append((to_chw(noisy), to_chw(clean)))
    x = torch.stack([r[0] for r in rows]).to(device)
    y = torch.stack([r[1] for r in rows]).to(device)
    res, ws = {}, []
    for name, fn in [("input", lambda a: a), ("hard_routed", lambda a: router(a)[0]),
                     ("soft_moe", lambda a: moe(a)[0])]:
        outs = []
        for b in range(0, n, 100):
            outs.append(fn(x[b:b + 100]).float().clamp(0, 1))
            if name == "soft_moe":
                ws.append(moe(x[b:b + 100])[1].cpu())
        o = torch.cat(outs)
        res[name] = {"psnr": psnr_per_image(o, y).mean().item(), "ssim": ssim_per_image(o, y).mean().item()}
    res["soft_moe_mean_w"] = torch.cat(ws).mean(0).tolist()
    preds = torch.cat([router(x[b:b + 100])[2].cpu() for b in range(0, n, 100)])
    res["hard_route_counts"] = {CLASSES[k]: int((preds == k).sum()) for k in range(4)}
    return res, x[:4], y[:4]


def stage_eval(args, data, device, mlflow):
    moe = load_moe(device)
    loader = data.test_loader(num_workers=args.workers)

    def predict(x, label=None):
        out, w, _ = moe(x)
        return out, {"w": w}

    df = evaluate_on_manifest(predict, loader, device)
    df.to_csv(os.path.join(OUT, "moe_test_per_sample.csv"), index=False)
    by_type, by_sev = summary_tables(df, OUT, "moe_test")

    # ---- routing analysis
    wcols = [f"w_{k}" for k in range(4)]
    w_by = df.groupby(["type", "severity"], sort=False)[wcols].mean()
    w_by.columns = BRANCHES
    w_by.to_csv(os.path.join(OUT, "moe_weights_by_type_severity.csv"))
    open(os.path.join(OUT, "moe_weights_by_type_severity.tex"), "w").write(w_by.to_latex(float_format="%.3f"))
    routing_heatmap(df, os.path.join(OUT, "moe_routing_heatmap.png"))
    weight_distribution_plot(df, os.path.join(OUT, "moe_weight_distribution.png"))
    W = df[wcols].values
    df["w_max"] = W.max(1)
    df["w_entropy"] = -(W * np.log(W + 1e-9)).sum(1)
    df["top_branch"] = W.argmax(1)
    usage = {
        "mean_weight": dict(zip(BRANCHES, W.mean(0).round(4).tolist())),
        "top1_share": {b: float((df.top_branch == k).mean()) for k, b in enumerate(BRANCHES)},
        "route_acc_top1": float((df.top_branch == df.label).mean()),
        "inactive_branches(<2% mean weight)": [b for k, b in enumerate(BRANCHES) if W[:, k].mean() < 0.02],
        "off_target_weight_by_true_class": {
            CLASSES[c]: float(1 - df[df.label == c][f"w_{c}"].mean()) for c in range(4)},
        "share_dominated(w_max>0.9)": float((df.w_max > 0.9).mean()),
        "share_distributed(w_max<0.6)": float((df.w_max < 0.6).mean()),
    }
    json.dump(usage, open(os.path.join(OUT, "moe_expert_usage.json"), "w"), indent=2)
    print(json.dumps(usage, indent=2))

    ds = loader.dataset

    def wtxt(ids):
        d = df.set_index("entry")
        return ["w=" + "/".join(f"{d.loc[e, c]:.2f}" for c in wcols) for e in ids]

    dominated = df.sort_values("w_max", ascending=False).groupby("type").head(1).entry.tolist()
    distributed = df.sort_values("w_entropy", ascending=False).head(4).entry.tolist()
    restoration_grid(ds, dominated, predict, device, os.path.join(OUT, "moe_examples_dominant.png"), extra_text=wtxt(dominated))
    restoration_grid(ds, distributed, predict, device, os.path.join(OUT, "moe_examples_distributed.png"), extra_text=wtxt(distributed))
    fails = pick_failures(df)
    restoration_grid(ds, fails, predict, device, os.path.join(OUT, "moe_failures.png"), extra_text=wtxt(fails))

    # ---- comparison with Tasks 1 and 2 (same test manifest)
    comp = {"Soft MoE (T3)": df}
    for name, rel in [("Universal DAE (T1)", "task1/test_per_sample_final.csv"),
                      ("Hard routing, predicted (T2)", "task2/hard_predicted_per_sample.csv"),
                      ("Hard routing, oracle (T2)", "task2/hard_oracle_per_sample.csv")]:
        p = os.path.join(paths.RESULTS_DIR, rel)
        if os.path.exists(p):
            comp[name] = pd.read_csv(p)
    rows = {}
    for name, d in comp.items():
        g = d.groupby("type")
        rows[name] = {**{f"{t} PSNR": g.psnr.mean().get(t, np.nan) for t in CLASSES},
                      **{f"{t} SSIM": g.ssim.mean().get(t, np.nan) for t in CLASSES},
                      "overall PSNR": d.psnr.mean(), "overall SSIM": d.ssim.mean()}
    comp_df = pd.DataFrame(rows).T
    comp_df.to_csv(os.path.join(OUT, "comparison_tasks123.csv"))
    open(os.path.join(OUT, "comparison_tasks123.tex"), "w").write(comp_df.to_latex(float_format="%.3f"))
    print(comp_df.round(3).to_string())

    # ---- two-corruption experiment
    specs = load_specialists(device)
    router = HardRoutedRestorer(load_classifier(device), [specs[e] for e in EXPERTS]).to(device).eval()
    mixed, _, _ = mixed_corruption_experiment(moe, router, data, device)
    json.dump(mixed, open(os.path.join(OUT, "mixed_corruption_experiment.json"), "w"), indent=2)
    print("mixed blur+s&p:", json.dumps(mixed, indent=1))

    with mlflow.start_run(run_name="task3_moe_test"):
        mlflow.log_metrics({"test_psnr": df.psnr.mean(), "test_ssim": df.ssim.mean(),
                            "test_route_acc": usage["route_acc_top1"]})
        mlflow.log_artifacts(OUT, "task3_results")


def stage_export(args, data, device, mlflow):
    from genai.onnx_utils import export_onnx, verify_onnx
    cpu = torch.device("cpu")
    moe = load_moe(cpu)
    wrapper = SoftMoEExport(moe).eval()
    path = os.path.join(paths.ONNX_DIR, "task3_soft_moe.onnx")
    export_onnx(wrapper, (torch.rand(1, 3, 128, 128),), path, ["input"], ["output", "weights"])
    ds = data.test_loader(num_workers=0).dataset
    x = torch.stack([ds[i][0] for i in range(40)])
    rep = verify_onnx(wrapper, path, x, atol=1e-4, report_path=os.path.join(OUT, "onnx_check.json"))
    assert rep["passed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["optuna", "train", "eval", "export"])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="joint fine-tuning epochs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    if args.batch_size:
        DEFAULT["batch_size"] = args.batch_size
    args.warmup = args.warmup if args.warmup is not None else {"optuna": 1}.get(args.stage, 2)
    args.epochs = args.epochs or {"optuna": 3}.get(args.stage, 12)
    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    mlflow = paths.setup_mlflow("task3_soft_moe")
    {"optuna": stage_optuna, "train": stage_train, "eval": stage_eval, "export": stage_export}[args.stage](
        args, data, device, mlflow)


if __name__ == "__main__":
    main()
