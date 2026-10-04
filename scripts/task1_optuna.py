"""Task 1 - Optuna search for the universal denoising autoencoder.

Search space: lr, batch size, bottleneck (latent channels), encoder base channels,
dropout, loss weight alpha. Objective (minimise): val L1 + (1 - val SSIM).
Resumable: re-running continues the same SQLite study.

    python scripts/task1_optuna.py --trials 16 --epochs 6
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna  # noqa: E402

from genai import paths  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.models import ConvAutoencoder, count_params  # noqa: E402
from genai.train_utils import fit_restorer, get_device, set_seed  # noqa: E402

STUDY = "task1_udae"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    val_loader = data.val_loader(num_workers=args.workers)
    mlflow = paths.setup_mlflow("task1_universal_dae")

    def objective(trial):
        p = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "latent_ch": trial.suggest_categorical("latent_ch", [4, 8, 16, 32]),
            "base_ch": trial.suggest_categorical("base_ch", [16, 32, 48, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
        }
        set_seed(42)
        model = ConvAutoencoder(p["base_ch"], p["latent_ch"], p["dropout"]).to(device)
        train_loader = data.train_loader(p["batch_size"], mode="uniform", num_workers=args.workers)
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number:03d}", nested=True):
            mlflow.log_params({**p, "trial": trial.number, "epochs": args.epochs,
                               "bottleneck_dim": model.bottleneck_size(), "params": count_params(model)})
            try:
                best, _ = fit_restorer(model, train_loader, val_loader, alpha=p["alpha"], lr=p["lr"],
                                       epochs=args.epochs, device=device, trial=trial, mlflow=mlflow)
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                raise
            mlflow.set_tag("state", "complete")
            mlflow.log_metrics({"best_val_objective": best["objective"], "best_val_ssim": best["ssim"],
                                "best_val_psnr": best["psnr"]})
        trial.set_user_attr("val_ssim", best["ssim"])
        trial.set_user_attr("val_psnr", best["psnr"])
        return best["objective"]

    study = optuna.create_study(study_name=STUDY, storage=paths.optuna_storage(STUDY), load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2))
    if not study.trials:  # seed the search with the assignment's suggested starting point
        study.enqueue_trial({"lr": 1e-3, "batch_size": 32, "latent_ch": 16, "base_ch": 32, "dropout": 0.0, "alpha": 0.8})
    done = len([t for t in study.trials if t.state.is_finished()])
    with mlflow.start_run(run_name="task1_optuna_study"):
        mlflow.log_params({"n_trials_target": args.trials, "epochs_per_trial": args.epochs})
        study.optimize(objective, n_trials=max(0, args.trials - done), gc_after_trial=True)
        save_study_outputs(study, mlflow)


def save_study_outputs(study, mlflow):
    out = os.path.join(paths.RESULTS_DIR, "task1")
    os.makedirs(out, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(out, "optuna_trials.csv"), index=False)
    best = {"trial": study.best_trial.number, "objective": study.best_value, **study.best_params,
            **study.best_trial.user_attrs,
            "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
            "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials)}
    with open(os.path.join(out, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("BEST:", json.dumps(best, indent=2))
    save_optuna_plots(study, out)
    mlflow.log_artifacts(out, "optuna")


def save_optuna_plots(study, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm
    for name, fn in [("optuna_history", ovm.plot_optimization_history),
                     ("optuna_importance", ovm.plot_param_importances),
                     ("optuna_parallel", ovm.plot_parallel_coordinate)]:
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            fig.set_size_inches(8, 5)
            fig.tight_layout()
            fig.savefig(os.path.join(out, name + ".png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # importance needs >1 complete trial etc.
            print(f"skip {name}: {e}")


if __name__ == "__main__":
    main()
