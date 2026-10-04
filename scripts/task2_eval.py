"""Task 2 - hard-routed restoration on the test manifest, oracle vs predicted routing.

    python scripts/task2_eval.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from genai import paths  # noqa: E402
from genai.corruptions import CLASSES  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.evaluation import (bar_by_severity, evaluate_on_manifest, pick_examples, pick_failures,  # noqa: E402
                              restoration_grid, summary_tables)
from genai.routing import HardRoutedRestorer  # noqa: E402
from genai.train_utils import get_device  # noqa: E402
from task2_classifier import load_classifier  # noqa: E402
from task2_specialists import EXPERTS, load_specialists  # noqa: E402

OUT = os.path.join(paths.RESULTS_DIR, "task2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    specs = load_specialists(device)
    router = HardRoutedRestorer(load_classifier(device), [specs[e] for e in EXPERTS]).to(device).eval()
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    loader = data.test_loader(num_workers=args.workers)

    def predicted(x, label=None):
        out, probs, pred = router(x)
        return out, {"prob": probs, "pred": pred}

    def oracle(x, label):
        out, probs, pred = router(x, route=label)
        return out, {"prob": probs, "pred": pred}

    dfs = {}
    for mode, fn in (("oracle", oracle), ("predicted", predicted)):
        print(f"=== {mode} routing")
        df = evaluate_on_manifest(fn, loader, device)
        df.to_csv(os.path.join(OUT, f"hard_{mode}_per_sample.csv"), index=False)
        summary_tables(df, OUT, f"hard_{mode}")
        dfs[mode] = df

    # side-by-side table + misrouting analysis
    o, p = dfs["oracle"], dfs["predicted"]
    cmp = pd.DataFrame({
        "input_psnr": o.groupby("type").in_psnr.mean(),
        "oracle_psnr": o.groupby("type").psnr.mean(), "predicted_psnr": p.groupby("type").psnr.mean(),
        "oracle_ssim": o.groupby("type").ssim.mean(), "predicted_ssim": p.groupby("type").ssim.mean(),
        "routing_acc": p.assign(ok=p.pred == p.label).groupby("type").ok.mean(),
    })
    cmp.loc["overall"] = [o.in_psnr.mean(), o.psnr.mean(), p.psnr.mean(), o.ssim.mean(), p.ssim.mean(),
                          (p.pred == p.label).mean()]
    cmp.to_csv(os.path.join(OUT, "hard_oracle_vs_predicted.csv"))
    open(os.path.join(OUT, "hard_oracle_vs_predicted.tex"), "w").write(cmp.to_latex(float_format="%.3f"))
    print(cmp.round(4).to_string())

    _, by_sev = summary_tables(p, OUT, "hard_predicted")
    bar_by_severity(by_sev, os.path.join(OUT, "hard_predicted_psnr_by_severity.png"), "psnr",
                    "Task 2 hard routing (predicted) - test PSNR")

    mis = p[p.pred != p.label].copy()
    mis = mis.merge(o[["entry", "ssim", "psnr"]].rename(columns={"ssim": "oracle_ssim", "psnr": "oracle_psnr"}), on="entry")
    mis["ssim_drop"] = mis.oracle_ssim - mis.ssim
    mis["route"] = mis.apply(lambda r: f"{CLASSES[r.label]} -> {CLASSES[r.pred]}", axis=1)
    mis.to_csv(os.path.join(OUT, "hard_misrouted.csv"), index=False)
    summary = mis.groupby("route").agg(n=("entry", "size"), mean_ssim_drop=("ssim_drop", "mean"),
                                       mean_psnr=("psnr", "mean"), mean_oracle_psnr=("oracle_psnr", "mean"))
    summary.to_csv(os.path.join(OUT, "hard_misrouting_summary.csv"))
    print(f"misrouted: {len(mis)} / {len(p)}")
    print(summary.round(4).to_string())

    ds = loader.dataset

    def label_txt(df, ids):
        d = df.set_index("entry")
        txt = []
        for e in ids:
            k = int(d.loc[e, "pred"])
            txt.append(f"pred {CLASSES[k]} p={d.loc[e, 'prob_%d' % k]:.2f}")
        return txt

    ex = pick_examples(p)
    restoration_grid(ds, ex[:6], predicted, device, os.path.join(OUT, "hard_examples_a.png"), extra_text=label_txt(p, ex[:6]))
    restoration_grid(ds, ex[6:], predicted, device, os.path.join(OUT, "hard_examples_b.png"), extra_text=label_txt(p, ex[6:]))
    worst_mis = mis.nlargest(6, "ssim_drop").entry.tolist()
    if worst_mis:
        restoration_grid(ds, worst_mis, predicted, device, os.path.join(OUT, "hard_misrouted_failures.png"),
                         extra_text=label_txt(p, worst_mis))
    fails = pick_failures(p)
    restoration_grid(ds, fails, predicted, device, os.path.join(OUT, "hard_failures.png"), extra_text=label_txt(p, fails))

    mlflow = paths.setup_mlflow("task2_hard_routing")
    with mlflow.start_run(run_name="task2_hard_routing_test"):
        for mode, df in dfs.items():
            mlflow.log_metrics({f"test_{mode}_psnr": df.psnr.mean(), f"test_{mode}_ssim": df.ssim.mean()})
        mlflow.log_metric("test_routing_accuracy", float((p.pred == p.label).mean()))
        mlflow.log_artifacts(OUT, "task2_results")


if __name__ == "__main__":
    main()
