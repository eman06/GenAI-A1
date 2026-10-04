# RestoreLab — GenAI Assignment 1

Four generative models in one browser application:

| Workspace | Task | Model |
|---|---|---|
| **Universal Restoration** | 1 | Convolutional denoising autoencoder with a 16×16×c bottleneck and no skip connections |
| **Hard-Routed Restoration** | 2 | CNN corruption classifier → one of three specialist autoencoders (identity bypass for clean inputs) |
| **Soft Mixture-of-Experts Restoration** | 3 | Gate (initialised from the classifier) mixes identity + 3 specialists; jointly fine-tuned |
| **Face-to-Sketch Generator** | 4 | Style-conditioned U-Net generator + PatchGAN discriminator (pix2pix-style cGAN) on FS2K |

All models are trained in PyTorch, tuned with **Optuna**, tracked with **MLflow**, exported to **ONNX** (checked against PyTorch), and served by **FastAPI** + **React/Tailwind** in **Docker Compose**.

---

## 1. Run the application (one command)

Requirements: Docker Desktop (or Docker Engine + Compose v2).

```bash
git clone https://github.com/eman06/GenAI-A1.git
cd GenAI-A1
# put the 7 ONNX files into ./models_onnx  (see "Model files" below)
docker compose up --build
```

Open **http://localhost:8080**. The API docs are at http://localhost:8000/docs.
If those ports are busy, run `FRONTEND_PORT=8088 BACKEND_PORT=8010 docker compose up --build`.

### Model files

The trained ONNX models are not committed to git. Download `models_onnx.zip` from **https://github.com/eman06/GenAI-A1/releases/download/v1.0/models_onnx.zip** and unzip it into `models_onnx/`:

```
models_onnx/
  task1_universal_dae.onnx
  task2_classifier.onnx
  task2_specialist_salt_pepper.onnx
  task2_specialist_blur.onnx
  task2_specialist_occlusion.onnx
  task3_soft_moe.onnx
  task4_generator.onnx
```

The **System** page in the app (and `GET /api/health`) shows which models are loaded.

### Using the app
* **Restoration workspaces:** upload an image (or click a sample), choose a runtime corruption (salt & pepper / blur / occlusion at low, medium or high test severity, with a seed), or choose *None* if the upload is already corrupted. Click **Run model** to see the clean reference, model input, restored output, error map, PSNR/L1 and inference time. Hard routing also shows the 4 classifier probabilities, the predicted corruption and the selected expert. Soft MoE shows the 4 routing weights and which experts contributed most.
* **Face-to-Sketch:** upload a photo or use the webcam, pick Style 1/2/3, click **Generate sketch**, then **Download**.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, loaded/missing models, runtime versions |
| GET | `/api/info` | ONNX input/output signatures, corruption settings |
| POST | `/api/restore/universal` | Task 1 |
| POST | `/api/restore/hard` | Task 2 (probabilities, predicted class, selected expert) |
| POST | `/api/restore/soft` | Task 3 (routing weights, dominant expert) |
| POST | `/api/sketch` | Task 4 (`style` = 0/1/2) |
| POST | `/api/corrupt` | apply a corruption only |

The restore endpoints take multipart form fields: `file` or `sample`, plus `corruption` (`none|clean|salt_pepper|blur|occlusion`), `level` (0–2) and `seed`.

---

## 2. Reproduce the training (Google Colab, T4)

Everything after the data download is one resumable script. Each stage is skipped if its output already exists, so after a disconnect you just run it again.

```bash
pip install -r requirements.txt
export GENAI_DATA=/content/data GENAI_OUT=/content/outputs BACKUP=/content/drive/MyDrive/genai_a1
bash scripts/run_all.sh
```

`notebooks/colab_runner.ipynb` contains the same steps as notebook cells. Individual stages:

```bash
python scripts/prepare_data.py                       # Oxford-IIIT Pet, 80/20 split (seed 42), manifests
python scripts/task1_optuna.py --trials 12 --epochs 6
python scripts/task1_train.py --epochs 35
python scripts/task1_eval.py && python scripts/task1_export.py
python scripts/task2_classifier.py optuna|train|eval
python scripts/task2_specialists.py optuna|train
python scripts/task2_eval.py && python scripts/task2_export.py
python scripts/task3_moe.py optuna|train|eval|export
python scripts/task4_cgan.py prepare|optuna|train|eval|export   # needs FS2K unzipped under $GENAI_DATA
python -m pytest tests -q                            # corruption-spec unit tests
```

* **Datasets:** Oxford-IIIT Pet is downloaded automatically by torchvision. FS2K comes from the official Google Drive link in the [FS2K repo](https://github.com/DengPingFan/FS2K) (`run_all.sh` downloads it with `gdown`).
* **Optuna studies:** stored as SQLite files in `outputs/optuna/*.db`. Trial tables and plots are in `outputs/results/task*/`.
* **MLflow:** view with `mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db`. Parameters, per-epoch losses and metrics, sample image grids, checkpoints and result files are all logged.

## 3. Repository layout

```
genai/            shared library: corruptions (NumPy, also used by the backend), data + manifests,
                  models, losses/metrics, training loops, evaluation/figures, routing (hard/soft MoE),
                  FS2K loader, cGAN, ONNX export/verification
scripts/          data prep, train/optuna/eval/export per task, run_all.sh
manifests/        split.json, val/test corruption manifests, fs2k_split.json (deterministic)
tests/            corruption specification tests
backend/          FastAPI app + Dockerfile
frontend/         React + Tailwind (Vite) app + nginx Dockerfile
report/           IEEE LaTeX report, figures, diagrams
docker-compose.yml
```

## 4. Data protocol (Tasks 1–3)

* Official trainval set (3,680 images) split 80/20 with seed 42 → 2,944 train / 736 val. The official test set (3,669 images) is used only by the `*_eval.py` scripts.
* RGB images resized to 128×128.
* **Training:** a fresh corruption type and severity is drawn every time an image is loaded, inside the DataLoader. Nothing corrupted is stored on disk.
* **Validation:** a fixed manifest storing the type, parameters, rectangle coordinates and seed for every image.
* **Test:** 10 entries per image: clean, plus 3 corruptions × 3 fixed severities. Salt & pepper p = 0.03/0.08/0.15; blur (k, σ) = (3, 0.7)/(5, 1.5)/(7, 2.5); occlusion 1/2/3 rectangles covering ≈ 10/20/35%.

## 5. Acknowledgements

Oxford-IIIT Pet (Parkhi et al., CC BY-SA 4.0), FS2K (Fan et al.). PatchGAN/U-Net design follows pix2pix (Isola et al.). AI-assistant use is described in the report appendix.
