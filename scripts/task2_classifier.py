"""Task 2 - corruption classifier: Optuna search, final training, test evaluation.

    python scripts/task2_classifier.py optuna --trials 12 --epochs 5
    python scripts/task2_classifier.py train --epochs 25
    python scripts/task2_classifier.py eval
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from genai import paths  # noqa: E402
from genai.classifier_utils import (classification_report_dict, fit_classifier, plot_classifier_history,  # noqa: E402
                                    plot_confusion, predict_classifier)
from genai.corruptions import CLASSES, SEVERITY_NAMES  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.models import CorruptionClassifier, count_params  # noqa: E402
from genai.train_utils import get_device, set_seed  # noqa: E402

STUDY = "task2_classifier"
CHANNEL_CONFIGS = {"small": (16, 32, 64, 128), "medium": (32, 64, 128, 256), "wide": (48, 96, 192, 256)}
DEFAULT = {"lr": 1e-3, "batch_size": 64, "channels": "medium", "dropout": 0.3, "weight_decay": 1e-4}
OUT = os.path.join(paths.RESULTS_DIR, "task2")
CKPT = os.path.join(paths.CKPT_DIR, "task2_classifier.pt")


def build(cfg):
    return CorruptionClassifier(CHANNEL_CONFIGS[cfg["channels"]], cfg["dropout"])


def stage_optuna(args, data, device, mlflow):
    val_loader = data.val_loader(num_workers=args.workers)

    def objective(trial):
        p = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),   # multiples of 4 -> exact balance
            "channels": trial.suggest_categorical("channels", list(CHANNEL_CONFIGS)),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
        set_seed(42)
        model = build(p).to(device)
        tl = data.train_loader(p["batch_size"], mode="balanced", num_workers=args.workers)
        with mlflow.start_run(run_name=f"clf_trial_{trial.number:03d}", nested=True):
            mlflow.log_params({**p, "trial": trial.number, "params": count_params(model)})
            best, _ = fit_classifier(model, tl, val_loader, lr=p["lr"], weight_decay=p["weight_decay"],
                                     epochs=args.epochs, device=device, trial=trial, mlflow=mlflow)
            mlflow.log_metrics({"best_val_macro_f1": best["macro_f1"], "best_val_acc": best["accuracy"]})
        trial.set_user_attr("val_acc", best["accuracy"])
        return best["macro_f1"]

    study = optuna.create_study(study_name=STUDY, storage=paths.optuna_storage(STUDY), load_if_exists=True,
                                direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2))
    if not study.trials:
        study.enqueue_trial(DEFAULT)
    done = len([t for t in study.trials if t.state.is_finished()])
    with mlflow.start_run(run_name="task2_classifier_optuna"):
        study.optimize(objective, n_trials=max(0, args.trials - done), gc_after_trial=True)
        study.trials_dataframe().to_csv(os.path.join(OUT, "classifier_optuna_trials.csv"), index=False)
        best = {"trial": study.best_trial.number, "val_macro_f1": study.best_value, **study.best_params,
                **study.best_trial.user_attrs,
                "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
                "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials)}
        json.dump(best, open(os.path.join(OUT, "classifier_best_params.json"), "w"), indent=2)
        print("BEST:", best)
        from task1_optuna import save_optuna_plots
        save_optuna_plots(study, OUT)
        for f in os.listdir(OUT):
            if f.startswith("optuna_"):
                os.replace(os.path.join(OUT, f), os.path.join(OUT, "classifier_" + f))
        mlflow.log_artifacts(OUT, "task2")


def load_cfg():
    cfg = dict(DEFAULT)
    p = os.path.join(OUT, "classifier_best_params.json")
    if os.path.exists(p):
        cfg.update({k: v for k, v in json.load(open(p)).items() if k in DEFAULT})
    return cfg


def stage_train(args, data, device, mlflow):
    cfg = load_cfg()
    print("config:", cfg)
    set_seed(42)
    model = build(cfg).to(device)
    tl = data.train_loader(cfg["batch_size"], mode="balanced", num_workers=args.workers)
    vl = data.val_loader(num_workers=args.workers)
    with mlflow.start_run(run_name="task2_classifier_final"):
        mlflow.log_params({**cfg, "epochs": args.epochs, "params": count_params(model)})
        best, hist = fit_classifier(model, tl, vl, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                    epochs=args.epochs, device=device, mlflow=mlflow)
        torch.save({"state_dict": model.state_dict(), "model_cfg": cfg, "channels": CHANNEL_CONFIGS[cfg["channels"]],
                    "best_val": best}, CKPT)
        plot_classifier_history(hist, os.path.join(OUT, "classifier_curves.png"))
        json.dump(hist, open(os.path.join(OUT, "classifier_history.json"), "w"), indent=2)
        mlflow.log_artifact(os.path.join(OUT, "classifier_curves.png"))
        mlflow.log_artifact(CKPT, "checkpoints")
    print("best val:", best)


def load_classifier(device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    m = build(ck["model_cfg"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def stage_eval(args, data, device, mlflow):
    model = load_classifier(device)
    loader = data.test_loader(num_workers=args.workers)
    probs, labels, levels = predict_classifier(model, loader, device)
    rep = classification_report_dict(probs, labels)
    # accuracy per (true type, severity): shows which mild corruptions get mistaken for clean
    df = pd.DataFrame({"type": [CLASSES[l] for l in labels], "level": levels, "pred": probs.argmax(1), "label": labels})
    df["severity"] = df.level.map(lambda l: "none" if l < 0 else SEVERITY_NAMES[l])
    df["correct"] = df.pred == df.label
    rep["accuracy_by_type_severity"] = {f"{t}/{s}": float(g.correct.mean()) for (t, s), g in df.groupby(["type", "severity"])}
    json.dump(rep, open(os.path.join(OUT, "classifier_test_report.json"), "w"), indent=2)
    plot_confusion(rep["confusion_normalized"], os.path.join(OUT, "classifier_confusion.png"),
                   "Task 2 classifier - test (row-normalized)")
    per_class = pd.DataFrame(rep["per_class"]).T
    per_class.loc["macro avg"] = [rep["macro_precision"], rep["macro_recall"], rep["macro_f1"], per_class.support.sum()]
    per_class.to_csv(os.path.join(OUT, "classifier_per_class.csv"))
    open(os.path.join(OUT, "classifier_per_class.tex"), "w").write(per_class.to_latex(float_format="%.3f"))
    print(per_class.round(4).to_string())
    print("accuracy", rep["accuracy"], "| by type/severity:", json.dumps(rep["accuracy_by_type_severity"], indent=1))
    with mlflow.start_run(run_name="task2_classifier_test"):
        mlflow.log_metrics({"test_accuracy": rep["accuracy"], "test_macro_f1": rep["macro_f1"],
                            "test_macro_precision": rep["macro_precision"], "test_macro_recall": rep["macro_recall"]})
        mlflow.log_artifact(os.path.join(OUT, "classifier_confusion.png"))
        mlflow.log_artifact(os.path.join(OUT, "classifier_test_report.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["optuna", "train", "eval"])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    args.epochs = args.epochs or {"optuna": 5, "train": 25, "eval": 0}[args.stage]
    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    mlflow = paths.setup_mlflow("task2_hard_routing")
    {"optuna": stage_optuna, "train": stage_train, "eval": stage_eval}[args.stage](args, data, device, mlflow)


if __name__ == "__main__":
    main()
