"""Task 2 - three specialist denoising autoencoders (salt&pepper / blur / occlusion).

A single shared Optuna search picks one common architecture + optimiser setting: every trial
trains all three specialists (each only on its own corruption) and is scored by the mean of their
validation objectives. The final specialists are then trained independently with that config.

    python scripts/task2_specialists.py optuna --trials 10 --epochs 4
    python scripts/task2_specialists.py train --epochs 30
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

from genai import paths  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.evaluation import plot_history  # noqa: E402
from genai.models import ConvAutoencoder, build_autoencoder, count_params  # noqa: E402
from genai.train_utils import fit_restorer, fixed_val_batch, get_device, set_seed  # noqa: E402

STUDY = "task2_specialists"
EXPERTS = ["salt_pepper", "blur", "occlusion"]
DEFAULT = {"lr": 1e-3, "batch_size": 32, "latent_ch": 16, "base_ch": 32, "alpha": 0.8}
OUT = os.path.join(paths.RESULTS_DIR, "task2")


def ckpt_path(expert):
    return os.path.join(paths.CKPT_DIR, f"task2_specialist_{expert}.pt")


def stage_optuna(args, data, device, mlflow):
    val_loaders = {e: data.val_loader(types=[e], num_workers=args.workers) for e in EXPERTS}

    def objective(trial):
        p = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "latent_ch": trial.suggest_categorical("latent_ch", [4, 8, 16, 32]),
            "base_ch": trial.suggest_categorical("base_ch", [16, 32, 48, 64]),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
        }
        scores = {}
        with mlflow.start_run(run_name=f"spec_trial_{trial.number:03d}", nested=True):
            mlflow.log_params({**p, "trial": trial.number, "epochs": args.epochs})
            for k, expert in enumerate(EXPERTS):
                set_seed(42)
                model = ConvAutoencoder(p["base_ch"], p["latent_ch"]).to(device)
                tl = data.train_loader(p["batch_size"], types=[expert], num_workers=args.workers)
                best, _ = fit_restorer(model, tl, val_loaders[expert], alpha=p["alpha"], lr=p["lr"],
                                       epochs=args.epochs, device=device, mlflow=mlflow, metric_prefix=f"{expert}_")
                scores[expert] = best["objective"]
                trial.set_user_attr(f"{expert}_ssim", best["ssim"])
                trial.set_user_attr(f"{expert}_psnr", best["psnr"])
                # prune on the running mean after each specialist
                trial.report(float(np.mean(list(scores.values()))), k)
                if trial.should_prune():
                    mlflow.set_tag("state", "pruned")
                    raise optuna.TrialPruned()
            mean = float(np.mean(list(scores.values())))
            mlflow.log_metrics({"mean_val_objective": mean, **{f"{e}_val_objective": v for e, v in scores.items()}})
        return mean

    study = optuna.create_study(study_name=STUDY, storage=paths.optuna_storage(STUDY), load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0))
    if not study.trials:
        study.enqueue_trial(DEFAULT)
    done = len([t for t in study.trials if t.state.is_finished()])
    with mlflow.start_run(run_name="task2_specialists_optuna"):
        study.optimize(objective, n_trials=max(0, args.trials - done), gc_after_trial=True)
        study.trials_dataframe().to_csv(os.path.join(OUT, "specialists_optuna_trials.csv"), index=False)
        best = {"trial": study.best_trial.number, "mean_val_objective": study.best_value, **study.best_params,
                **study.best_trial.user_attrs,
                "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
                "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials)}
        json.dump(best, open(os.path.join(OUT, "specialists_best_params.json"), "w"), indent=2)
        print("BEST:", best)
        from task1_optuna import save_optuna_plots
        tmp = os.path.join(OUT, "_spec_plots")
        os.makedirs(tmp, exist_ok=True)
        save_optuna_plots(study, tmp)
        for f in os.listdir(tmp):
            os.replace(os.path.join(tmp, f), os.path.join(OUT, "specialists_" + f))
        os.rmdir(tmp)
        mlflow.log_artifacts(OUT, "task2")


def load_cfg():
    cfg = dict(DEFAULT)
    p = os.path.join(OUT, "specialists_best_params.json")
    if os.path.exists(p):
        cfg.update({k: v for k, v in json.load(open(p)).items() if k in DEFAULT})
    cfg["dropout"] = 0.0
    return cfg


def stage_train(args, data, device, mlflow):
    cfg = load_cfg()
    print("shared specialist config:", cfg)
    experts = args.only or EXPERTS
    for expert in experts:
        print(f"=== specialist: {expert}")
        set_seed(42)
        model = ConvAutoencoder(cfg["base_ch"], cfg["latent_ch"]).to(device)
        tl = data.train_loader(cfg["batch_size"], types=[expert], num_workers=args.workers)
        vl = data.val_loader(types=[expert], num_workers=args.workers)
        with mlflow.start_run(run_name=f"task2_specialist_{expert}"):
            mlflow.log_params({**cfg, "expert": expert, "epochs": args.epochs, "params": count_params(model)})
            best, hist = fit_restorer(model, tl, vl, alpha=cfg["alpha"], lr=cfg["lr"], epochs=args.epochs,
                                      device=device, mlflow=mlflow, fixed_batch=fixed_val_batch(vl))
            torch.save({"state_dict": model.state_dict(), "model_cfg": cfg, "expert": expert, "best_val": best},
                       ckpt_path(expert))
            plot_history(hist, os.path.join(OUT, f"specialist_{expert}_curves.png"), f"Task 2 specialist: {expert}")
            json.dump(hist, open(os.path.join(OUT, f"specialist_{expert}_history.json"), "w"), indent=2)
            mlflow.log_metrics({f"best_val_{k}": v for k, v in best.items()})
            mlflow.log_artifact(os.path.join(OUT, f"specialist_{expert}_curves.png"))
            mlflow.log_artifact(ckpt_path(expert), "checkpoints")
        print(expert, "best val:", best)


def load_specialists(device):
    out = {}
    for e in EXPERTS:
        ck = torch.load(ckpt_path(e), map_location=device, weights_only=False)
        m = build_autoencoder(ck["model_cfg"])
        m.load_state_dict(ck["state_dict"])
        out[e] = m.to(device).eval()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["optuna", "train"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only", nargs="*", choices=EXPERTS)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    args.epochs = args.epochs or {"optuna": 4, "train": 30}[args.stage]
    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    mlflow = paths.setup_mlflow("task2_hard_routing")
    {"optuna": stage_optuna, "train": stage_train}[args.stage](args, data, device, mlflow)


if __name__ == "__main__":
    main()
