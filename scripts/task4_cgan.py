"""Task 4 - style-conditioned face-to-sketch cGAN on FS2K.

    python scripts/task4_cgan.py prepare            # FS2K (already downloaded/unzipped under GENAI_DATA)
    python scripts/task4_cgan.py optuna --trials 10 --epochs 15
    python scripts/task4_cgan.py train --epochs 100
    python scripts/task4_cgan.py eval
    python scripts/task4_cgan.py export
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
import torchvision.utils as vutils  # noqa: E402

from genai import paths  # noqa: E402
from genai.fs2k import STYLE_NAMES, FS2KData, prepare  # noqa: E402
from genai.gan import GeneratorExport, StylePatchDiscriminator, StyleUNetGenerator, init_weights  # noqa: E402
from genai.losses import l1_per_image, psnr_per_image, ssim_per_image  # noqa: E402
from genai.train_utils import get_device, set_seed  # noqa: E402

STUDY = "task4_cgan"
OUT = os.path.join(paths.RESULTS_DIR, "task4")
CACHE = os.path.join(paths.DATA_ROOT, "fs2k_128.npz")
SPLIT = os.path.join(paths.MANIFEST_DIR, "fs2k_split.json")
CKPT = os.path.join(paths.CKPT_DIR, "task4_generator.pt")
DEFAULT = {"lr_g": 2e-4, "lr_d": 2e-4, "batch_size": 8, "base_ch": 64, "dropout": 0.5, "emb_dim": 16, "lambda_l1": 100.0}


def to01(t):
    return (t.float().clamp(-1, 1) + 1) / 2


@torch.no_grad()
def evaluate_gen(G, loader, device):
    G.eval()
    l1s, ss, ps = [], [], []
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        fake = to01(G(photo, style))
        real = to01(sketch)
        l1s.append(l1_per_image(fake, real).cpu())
        ss.append(ssim_per_image(fake, real).cpu())
        ps.append(psnr_per_image(fake, real).cpu())
    l1, s, p = (torch.cat(v).mean().item() for v in (l1s, ss, ps))
    return {"l1": l1, "ssim": s, "psnr": p, "objective": l1 + (1 - s)}


def sample_grid(G, fixed, device):
    """Rows: photo | generated (true style) | ground-truth sketch, for the same fixed val photos."""
    photo, sketch, style = fixed
    G.eval()
    with torch.no_grad():
        fake = to01(G(photo.to(device), style.to(device))).cpu()
    rows = torch.cat([to01(photo), fake.repeat(1, 3, 1, 1), to01(sketch).repeat(1, 3, 1, 1)])
    grid = vutils.make_grid(rows, nrow=photo.shape[0], padding=2)
    return (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def train_gan(cfg, data, device, epochs, workers, trial=None, mlflow=None, log_every=5):
    set_seed(42)
    G = StyleUNetGenerator(cfg["base_ch"], 3, cfg["emb_dim"], cfg["dropout"]).to(device)
    D = StylePatchDiscriminator(cfg["base_ch"], 3, cfg["emb_dim"]).to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    # pix2pix schedule: constant LR for the first half, linear decay to 0 over the second half
    lam = lambda ep: 1.0 - max(0, ep - epochs // 2) / float(epochs - epochs // 2 + 1)  # noqa: E731
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lam)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lam)
    tl = data.loader("train", cfg["batch_size"], shuffle=True, augment=True, workers=workers)
    vl = data.loader("val", 32, workers=workers)
    vd = data.val
    idx = np.linspace(0, len(vd["style"]) - 1, 8).astype(int)
    fixed = (torch.from_numpy(vd["photo"][idx].transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1),
             torch.from_numpy(vd["sketch"][idx].transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1),
             torch.from_numpy(vd["style"][idx]))
    bce = torch.nn.BCEWithLogitsLoss()
    best, best_state, history = None, None, []
    for ep in range(1, epochs + 1):
        G.train()
        D.train()
        t0, sums, n = time.time(), np.zeros(4), 0
        for photo, sketch, style in tl:
            photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
            fake = G(photo, style)
            # ---- discriminator: real pairs -> 1, generated pairs -> 0
            pr = D(photo, sketch, style)
            pf = D(photo, fake.detach(), style)
            d_real = bce(pr, torch.ones_like(pr))
            d_fake = bce(pf, torch.zeros_like(pf))
            opt_d.zero_grad(set_to_none=True)
            (0.5 * (d_real + d_fake)).backward()
            opt_d.step()
            # ---- generator: fool D + stay close to the paired sketch
            pf = D(photo, fake, style)
            g_adv = bce(pf, torch.ones_like(pf))
            g_l1 = F.l1_loss(fake, sketch)
            opt_g.zero_grad(set_to_none=True)
            (g_adv + cfg["lambda_l1"] * g_l1).backward()
            opt_g.step()
            sums += np.array([d_real.item(), d_fake.item(), g_adv.item(), g_l1.item()]) * len(style)
            n += len(style)
        sch_g.step()
        sch_d.step()
        val = evaluate_gen(G, vl, device)
        tr = dict(zip(["d_real_loss", "d_fake_loss", "g_adv_loss", "g_l1_loss"], sums / n))
        rec = {"epoch": ep, **tr, **{f"val_{k}": v for k, v in val.items()}, "epoch_time": time.time() - t0}
        history.append(rec)
        print(f"  ep {ep:3d} D real {tr['d_real_loss']:.3f} fake {tr['d_fake_loss']:.3f} | G adv {tr['g_adv_loss']:.3f} "
              f"L1 {tr['g_l1_loss']:.4f} | val SSIM {val['ssim']:.4f} L1 {val['l1']:.4f} ({rec['epoch_time']:.0f}s)")
        if mlflow is not None:
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=ep)
            if ep % log_every == 0 or ep == epochs or ep == 1:
                mlflow.log_image(sample_grid(G, fixed, device), f"val_samples/epoch_{ep:03d}.png")
        if best is None or val["objective"] < best["objective"]:
            best = {**val, "epoch": ep}
            best_state = copy.deepcopy(G.state_dict())
        if trial is not None:
            trial.report(val["objective"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    final_state = copy.deepcopy(G.state_dict())
    return G, best, best_state, final_state, history


def stage_optuna(args, data, device, mlflow):
    def objective(trial):
        cfg = {
            "lr_g": trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
            "lr_d": trial.suggest_float("lr_d", 2e-5, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "base_ch": trial.suggest_categorical("base_ch", [32, 48, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "emb_dim": trial.suggest_categorical("emb_dim", [8, 16, 32]),
            "lambda_l1": trial.suggest_float("lambda_l1", 10.0, 200.0, log=True),
        }
        with mlflow.start_run(run_name=f"cgan_trial_{trial.number:03d}", nested=True):
            mlflow.log_params({**cfg, "trial": trial.number, "epochs": args.epochs})
            _, best, _, _, _ = train_gan(cfg, data, device, args.epochs, args.workers, trial, mlflow, log_every=args.epochs)
            mlflow.log_metrics({"best_val_objective": best["objective"], "best_val_ssim": best["ssim"]})
        trial.set_user_attr("val_ssim", best["ssim"])
        trial.set_user_attr("val_l1", best["l1"])
        return best["objective"]

    study = optuna.create_study(study_name=STUDY, storage=paths.optuna_storage(STUDY), load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
    if not study.trials:
        study.enqueue_trial(DEFAULT)
    done = len([t for t in study.trials if t.state.is_finished()])
    with mlflow.start_run(run_name="task4_cgan_optuna"):
        study.optimize(objective, n_trials=max(0, args.trials - done), gc_after_trial=True)
        study.trials_dataframe().to_csv(os.path.join(OUT, "cgan_optuna_trials.csv"), index=False)
        best = {"trial": study.best_trial.number, "val_objective": study.best_value, **study.best_params,
                **study.best_trial.user_attrs,
                "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
                "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials)}
        json.dump(best, open(os.path.join(OUT, "cgan_best_params.json"), "w"), indent=2)
        print("BEST:", best)
        from task1_optuna import save_optuna_plots
        tmp = os.path.join(OUT, "_plots")
        os.makedirs(tmp, exist_ok=True)
        save_optuna_plots(study, tmp)
        for f in os.listdir(tmp):
            os.replace(os.path.join(tmp, f), os.path.join(OUT, "cgan_" + f))
        os.rmdir(tmp)
        mlflow.log_artifacts(OUT, "task4")


def load_cfg():
    cfg = dict(DEFAULT)
    p = os.path.join(OUT, "cgan_best_params.json")
    if os.path.exists(p):
        cfg.update({k: v for k, v in json.load(open(p)).items() if k in DEFAULT})
    return cfg


def plot_gan_history(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(hist)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(h.epoch, h.d_real_loss, label="D real")
    axes[0].plot(h.epoch, h.d_fake_loss, label="D fake")
    axes[0].plot(h.epoch, h.g_adv_loss, label="G adversarial")
    axes[0].set_title("adversarial losses")
    axes[0].legend(fontsize=7)
    axes[1].plot(h.epoch, h.g_l1_loss, label="G L1 (train)")
    axes[1].plot(h.epoch, h.val_l1, label="val L1")
    axes[1].set_title("reconstruction (L1)")
    axes[1].legend(fontsize=7)
    axes[2].plot(h.epoch, h.val_ssim, color="tab:green")
    axes[2].set_title("val SSIM")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage_train(args, data, device, mlflow):
    cfg = load_cfg()
    print("config:", cfg)
    with mlflow.start_run(run_name="task4_cgan_final"):
        mlflow.log_params({**cfg, "epochs": args.epochs, "n_train": len(data.train["style"]),
                           "n_val": len(data.val["style"])})
        G, best, best_state, final_state, hist = train_gan(cfg, data, device, args.epochs, args.workers, mlflow=mlflow)
        # GAN validation L1/SSIM favour blurry outputs early on, so we deploy the FINAL generator
        # (end of the LR decay) and keep the best-val-SSIM weights only for reference.
        torch.save({"state_dict": final_state, "best_val_state_dict": best_state, "cfg": cfg,
                    "best_val": best, "final_val": hist[-1]}, CKPT)
        json.dump(hist, open(os.path.join(OUT, "cgan_history.json"), "w"), indent=2)
        plot_gan_history(hist, os.path.join(OUT, "cgan_curves.png"))
        mlflow.log_artifact(os.path.join(OUT, "cgan_curves.png"))
        mlflow.log_artifact(CKPT, "checkpoints")
    print("best val:", best, "| final:", hist[-1])


def load_generator(device, which="state_dict"):
    ck = torch.load(CKPT, map_location=device)
    cfg = ck["cfg"]
    G = StyleUNetGenerator(cfg["base_ch"], 3, cfg["emb_dim"], cfg["dropout"])
    G.load_state_dict(ck[which])
    return G.to(device).eval()


def stage_eval(args, data, device, mlflow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = load_generator(device)
    tl = data.loader("test", 32, workers=args.workers)
    rows = []
    with torch.no_grad():
        for photo, sketch, style in tl:
            photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
            fake, real = to01(G(photo, style)), to01(sketch)
            m = {"l1": l1_per_image(fake, real), "ssim": ssim_per_image(fake, real), "psnr": psnr_per_image(fake, real)}
            for j in range(len(style)):
                rows.append({"style": int(style[j]), **{k: float(v[j]) for k, v in m.items()}})
    df = pd.DataFrame(rows)
    df["name"] = data.test["name"]
    df.to_csv(os.path.join(OUT, "cgan_test_per_sample.csv"), index=False)
    by_style = df.groupby("style")[["l1", "ssim", "psnr"]].mean()
    by_style.index = [STYLE_NAMES[i] for i in by_style.index]
    by_style.loc["overall"] = df[["l1", "ssim", "psnr"]].mean()
    by_style.to_csv(os.path.join(OUT, "cgan_test_by_style.csv"))
    open(os.path.join(OUT, "cgan_test_by_style.tex"), "w").write(by_style.to_latex(float_format="%.3f"))
    print(by_style.round(4).to_string())

    td = data.test

    def gen(i, s):
        p = torch.from_numpy(td["photo"][i:i + 1].transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1).to(device)
        with torch.no_grad():
            return to01(G(p, torch.tensor([s], device=device)))[0, 0].cpu().numpy()

    # (a) photo | generated (true style) | ground truth, 4 per style
    picks = []
    for s in range(3):
        ids = np.where(td["style"] == s)[0]
        picks += ids[np.linspace(0, len(ids) - 1, 4).astype(int)].tolist()
    fig, axes = plt.subplots(len(picks), 3, figsize=(6, 2 * len(picks)))
    for r, i in enumerate(picks):
        for c, (im, t) in enumerate([(td["photo"][i], "photo"), (gen(i, int(td["style"][i])), f"generated ({STYLE_NAMES[td['style'][i]]})"),
                                     (td["sketch"][i][..., 0], "ground truth")]):
            axes[r, c].imshow(im, cmap="gray" if c else None, vmin=0, vmax=1 if c == 1 else 255)
            axes[r, c].set_title(t, fontsize=7)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cgan_test_examples.png"), dpi=130)
    plt.close(fig)

    # (b) style control: same photo rendered in all three styles
    ids = picks[::2][:6]
    fig, axes = plt.subplots(len(ids), 4, figsize=(8, 2 * len(ids)))
    for r, i in enumerate(ids):
        axes[r, 0].imshow(td["photo"][i])
        axes[r, 0].set_title("photo", fontsize=7)
        for s in range(3):
            axes[r, s + 1].imshow(gen(i, s), cmap="gray", vmin=0, vmax=1)
            axes[r, s + 1].set_title(STYLE_NAMES[s], fontsize=7)
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cgan_style_control.png"), dpi=130)
    plt.close(fig)

    # (c) failure cases: lowest SSIM
    worst = df.nsmallest(6, "ssim").index.tolist()
    fig, axes = plt.subplots(len(worst), 3, figsize=(6, 2 * len(worst)))
    for r, i in enumerate(worst):
        for c, (im, t) in enumerate([(td["photo"][i], "photo"), (gen(i, int(td["style"][i])), f"generated SSIM {df.ssim[i]:.3f}"),
                                     (td["sketch"][i][..., 0], "ground truth")]):
            axes[r, c].imshow(im, cmap="gray" if c else None, vmin=0, vmax=1 if c == 1 else 255)
            axes[r, c].set_title(t, fontsize=7)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cgan_failures.png"), dpi=130)
    plt.close(fig)

    with mlflow.start_run(run_name="task4_cgan_test"):
        mlflow.log_metrics({"test_ssim": df.ssim.mean(), "test_l1": df.l1.mean(), "test_psnr": df.psnr.mean()})
        mlflow.log_artifacts(OUT, "task4_results")


def stage_export(args, data, device, mlflow):
    from genai.onnx_utils import export_onnx, verify_onnx
    G = load_generator(torch.device("cpu"))
    wrapper = GeneratorExport(G).eval()
    path = os.path.join(paths.ONNX_DIR, "task4_generator.onnx")
    export_onnx(wrapper, (torch.rand(1, 3, 128, 128), torch.zeros(1, dtype=torch.long)), path,
                ["photo", "style"], ["sketch"])
    td = data.test
    n = min(24, len(td["style"]))
    photo = torch.from_numpy(td["photo"][:n].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
    style = torch.arange(n) % 3
    rep = verify_onnx(wrapper, path, (photo, style), atol=1e-4, report_path=os.path.join(OUT, "onnx_check.json"))
    assert rep["passed"]


def stage_prepare(args, data, device, mlflow):
    split = prepare(paths.DATA_ROOT, CACHE, SPLIT)
    print(json.dumps({k: v for k, v in split.items() if k not in ("train", "val")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "optuna", "train", "eval", "export"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    args.epochs = args.epochs or {"optuna": 15}.get(args.stage, 100)
    os.makedirs(OUT, exist_ok=True)
    paths.ensure_dirs()
    if args.stage == "prepare":
        return stage_prepare(args, None, None, None)
    device = get_device()
    data = FS2KData(CACHE, SPLIT)
    mlflow = paths.setup_mlflow("task4_face_to_sketch")
    {"optuna": stage_optuna, "train": stage_train, "eval": stage_eval, "export": stage_export}[args.stage](
        args, data, device, mlflow)


if __name__ == "__main__":
    main()
