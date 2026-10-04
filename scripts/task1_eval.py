"""Task 1 - final evaluation on the untouched official test set (deterministic manifest).

    python scripts/task1_eval.py                      # final model
    python scripts/task1_eval.py --tag skip_ablation
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from genai import paths  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.evaluation import (bar_by_severity, evaluate_on_manifest, pick_examples, pick_failures,  # noqa: E402
                              restoration_grid, summary_tables)
from genai.models import build_autoencoder  # noqa: E402
from genai.train_utils import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(os.path.join(paths.CKPT_DIR, f"task1_udae_{args.tag}.pt"), map_location=device, weights_only=False)
    model = build_autoencoder(ck["model_cfg"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    loader = data.test_loader(num_workers=args.workers)
    out = os.path.join(paths.RESULTS_DIR, "task1")
    os.makedirs(out, exist_ok=True)

    def predict(x, label=None):
        return model(x), {}

    df = evaluate_on_manifest(predict, loader, device)
    df.to_csv(os.path.join(out, f"test_per_sample_{args.tag}.csv"), index=False)
    by_type, by_sev = summary_tables(df, out, f"test_{args.tag}")
    bar_by_severity(by_sev, os.path.join(out, f"psnr_by_severity_{args.tag}.png"), "psnr", "Task 1 test PSNR")
    bar_by_severity(by_sev, os.path.join(out, f"ssim_by_severity_{args.tag}.png"), "ssim", "Task 1 test SSIM")

    ds = loader.dataset
    ex = pick_examples(df)
    restoration_grid(ds, ex[:6], predict, device, os.path.join(out, f"examples_a_{args.tag}.png"))
    restoration_grid(ds, ex[6:], predict, device, os.path.join(out, f"examples_b_{args.tag}.png"))
    fails = pick_failures(df)
    txt = [f"SSIM {df.set_index('entry').loc[e, 'ssim']:.3f}" for e in fails]
    restoration_grid(ds, fails, predict, device, os.path.join(out, f"failures_{args.tag}.png"), extra_text=txt)

    mlflow = paths.setup_mlflow("task1_universal_dae")
    with mlflow.start_run(run_name=f"task1_test_eval_{args.tag}"):
        for t, r in by_type.iterrows():
            mlflow.log_metrics({f"test_{t}_psnr": r.psnr, f"test_{t}_ssim": r.ssim})
        mlflow.log_artifacts(out, "task1_results")
    print("saved results to", out)


if __name__ == "__main__":
    main()
