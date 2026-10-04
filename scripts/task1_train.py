"""Task 1 - retrain the best Optuna configuration on the full schedule.

    python scripts/task1_train.py --epochs 40            # uses results/task1/best_params.json
    python scripts/task1_train.py --epochs 25 --skip --tag skip_ablation   # limited-skip ablation
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from genai import paths  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.evaluation import plot_history  # noqa: E402
from genai.models import ConvAutoencoder, count_params  # noqa: E402
from genai.train_utils import fit_restorer, fixed_val_batch, get_device, set_seed  # noqa: E402

DEFAULT = {"lr": 1e-3, "batch_size": 32, "latent_ch": 16, "base_ch": 32, "dropout": 0.0, "alpha": 0.8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip", action="store_true", help="add one 64x64 skip connection (ablation)")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()

    out = os.path.join(paths.RESULTS_DIR, "task1")
    os.makedirs(out, exist_ok=True)
    best_path = os.path.join(out, "best_params.json")
    cfg = dict(DEFAULT)
    if os.path.exists(best_path):
        with open(best_path) as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULT})
    else:
        print("WARNING: no best_params.json, using the assignment's default starting point")
    cfg["skip"] = args.skip
    print("config:", cfg)

    set_seed(42)
    device = get_device()
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    train_loader = data.train_loader(cfg["batch_size"], mode="uniform", num_workers=args.workers)
    val_loader = data.val_loader(num_workers=args.workers)
    model = ConvAutoencoder(cfg["base_ch"], cfg["latent_ch"], cfg["dropout"], cfg["skip"]).to(device)

    mlflow = paths.setup_mlflow("task1_universal_dae")
    with mlflow.start_run(run_name=f"task1_{args.tag}"):
        mlflow.log_params({**cfg, "epochs": args.epochs, "params": count_params(model),
                           "bottleneck_dim": model.bottleneck_size(),
                           "compression_ratio": round(128 * 128 * 3 / model.bottleneck_size(), 2)})
        best, history = fit_restorer(model, train_loader, val_loader, alpha=cfg["alpha"], lr=cfg["lr"],
                                     epochs=args.epochs, device=device, mlflow=mlflow,
                                     fixed_batch=fixed_val_batch(val_loader))
        ckpt = os.path.join(paths.CKPT_DIR, f"task1_udae_{args.tag}.pt")
        torch.save({"state_dict": model.state_dict(), "model_cfg": cfg, "best_val": best}, ckpt)
        plot_history(history, os.path.join(out, f"curves_{args.tag}.png"), "Task 1 universal DAE")
        with open(os.path.join(out, f"history_{args.tag}.json"), "w") as f:
            json.dump(history, f, indent=2)
        mlflow.log_metrics({f"best_val_{k}": v for k, v in best.items()})
        mlflow.log_artifact(os.path.join(out, f"curves_{args.tag}.png"))
        mlflow.log_artifact(ckpt, "checkpoints")
    print("best val:", best, "->", ckpt)


if __name__ == "__main__":
    main()
