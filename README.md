# GenAI Assignment 1 — Image Restoration (Tasks 1–3) and Face-to-Sketch cGAN (Task 4)

| Task | Status |
|---|---|
| Shared data + runtime corruption pipeline | done |
| 1. Universal denoising autoencoder | done |
| 2. Corruption classifier + hard-routed specialists | todo |
| 3. Soft mixture-of-experts | todo |
| 4. Style-conditioned face-to-sketch cGAN | todo |
| FastAPI + React app, Docker Compose | todo |

## Repository layout

```
genai/                 shared library
  corruptions.py       NumPy corruption definitions (also imported by the backend)
  data.py              Oxford-IIIT Pet loading, 80/20 split, manifests, runtime-corruption loaders
  models.py            ConvAutoencoder, CorruptionClassifier
  losses.py            L1+SSIM loss, PSNR/SSIM metrics, alpha-independent selection objective
  train_utils.py       training loop with Optuna pruning + MLflow logging
  evaluation.py        per-condition test tables, example / failure / error-map figures
  onnx_utils.py        ONNX export and PyTorch-vs-ONNX Runtime check
  paths.py             output locations (override with GENAI_DATA / GENAI_OUT)
scripts/               prepare_data.py, task1_{optuna,train,eval,export}.py, ...
manifests/             split.json, val_manifest.json, test_manifest.json (deterministic, committed)
notebooks/colab_runner.ipynb   end-to-end training on Google Colab
tests/                 corruption-spec tests
models_onnx/           exported ONNX models (download link below; not committed)
```

## Data protocol (Tasks 1–3)

* Oxford-IIIT Pet `trainval` (3,680 images) split 80/20 with seed 42 → 2,944 train / 736 val.
  The official `test` split (3,669 images) is only used by the `*_eval.py` scripts.
* RGB, resized to 128×128 (bicubic).
* **Training:** corruptions are drawn inside the DataLoader collate function every time a
  batch is built. Nothing corrupted is saved to disk. Each image gets clean / salt-and-pepper / blur / occlusion with
  probability 1/4 (Task 1), or exactly balanced batches (Task 2 classifier, Task 3 gate).
  * salt-and-pepper: p ~ U(0.02, 0.15), black/white 50/50
  * blur: kernel ∈ {3,5,7}, σ ~ U(0.5, 2.5)
  * occlusion: 1–3 non-overlapping black rectangles jointly covering U(10%, 35%)
* **Validation:** `manifests/val_manifest.json`, one fixed corruption per image with its seed and all parameters.
* **Test:** `manifests/test_manifest.json`, 10 entries per image (clean + 3 corruptions × 3 severities):
  s&p p ∈ {0.03, 0.08, 0.15}; blur (k,σ) ∈ {(3,0.7),(5,1.5),(7,2.5)}; occlusion 1/2/3 rects ≈ 10/20/35%.

The manifests only depend on the dataset sizes and fixed seeds. `prepare_data.py` regenerates exactly the same files.

## Training on Google Colab

Open `notebooks/colab_runner.ipynb` in Colab (T4 GPU), set `REPO_URL`, and run the cells in order.
Outputs are copied to `MyDrive/genai_a1` after every step. The notebook runs:

```bash
python scripts/prepare_data.py
python scripts/task1_optuna.py --trials 16 --epochs 6
python scripts/task1_train.py --epochs 40
python scripts/task1_eval.py
python scripts/task1_export.py
```

Experiment tracking uses MLflow. Runs are written to `outputs/mlruns`. To browse them, download that folder and run:

```bash
mlflow ui --backend-store-uri outputs/mlruns
```

## Task 1 design notes

* **Architecture:** 3 stride-2 conv stages (channels b, 2b, 4b) take 128² down to 16², then a 1×1 conv
  produces a 16×16×c latent (c ∈ {4,8,16,32}, a 12–48× compression of the 49,152 input values). The decoder is symmetric,
  with transposed convs and a sigmoid output. There are no skip connections, so every reconstruction must pass through the
  bottleneck. `--skip` adds a single 64×64 skip connection for the ablation study.
* **Loss:** α·L1 + (1−α)(1−SSIM), with SSIM computed in fp32 (pytorch-msssim, 11×11 Gaussian window).
* **Optuna** (TPE, seed 42, median pruner) searches lr [1e-4, 3e-3] log, batch {16,32,64},
  latent_ch {4,8,16,32}, base_ch {16,32,48,64}, dropout [0, 0.3] and α [0.5, 0.95]. The first trial is the assignment's suggested
  starting point (α = 0.8). The objective is val L1 + (1 − SSIM) with fixed equal weights. It is deliberately
  independent of α, so trials with different α are compared on the same scale.
