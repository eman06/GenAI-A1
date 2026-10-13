# WHERE IS WHAT — VS Code navigation

**How to use:** `Ctrl+P` → type the file name → Enter. Then `Ctrl+F` → type the **search** text.
`Ctrl+G` → line number. `Ctrl+Shift+F` = search all files. `F12` on a name = go to its definition.

---

## 1. Data & corruption
| Rubric item | File (Ctrl+P) | Ctrl+F | Line |
|---|---|---|---|
| Load Pet, RGB, resize 128 | `genai/data.py` | `def load_pet_arrays` | 19 |
| Image size 128 | `genai/data.py` | `IMG_SIZE = 128` | 12 |
| 80/20 split, seed 42 | `genai/data.py` | `def make_split` | 37 |
| Seed value | `genai/data.py` | `SPLIT_SEED = 42` | 13 |
| Saved split (reused T1–3) | `manifests/split.json` | `"seed"` | – |
| **Runtime corruption in data loader** | `genai/data.py` | `class RuntimeCorruptCollate` | 96 |
| Equal probability / balanced batches | `genai/data.py` | `mode == "balanced"` | ~110 |
| Label returned | `genai/data.py` | `labels.append` | ~118 |
| Training corruption ranges | `genai/corruptions.py` | `TRAIN_SP_RANGE` | 25–29 |
| Salt & pepper (50/50 black/white) | `genai/corruptions.py` | `def salt_pepper` | 33 |
| Gaussian blur | `genai/corruptions.py` | `def gaussian_blur` | 51 |
| Rectangles 10–35% | `genai/corruptions.py` | `def sample_rects` | 76 |
| Random training spec | `genai/corruptions.py` | `def sample_train_spec` | 124 |
| **Fixed test severities** | `genai/corruptions.py` | `TEST_SEVERITIES` | 16 |
| Val manifest builder | `genai/data.py` | `def build_val_manifest` | 44 |
| Test manifest builder (10/image) | `genai/data.py` | `def build_test_manifest` | 54 |
| Manifests (type, params, rects, seed) | `manifests/test_manifest.json` | `"rects"` | – |
| Unit tests | `tests/test_corruptions.py` | `def test_` | – |
| Run data prep | `scripts/prepare_data.py` | `def main` | – |

## 2. Task 1 – Universal autoencoder
| Rubric item | File | Ctrl+F | Line |
|---|---|---|---|
| **Encoder / bottleneck / decoder** | `genai/models.py` | `class ConvAutoencoder` | 14 |
| Bottleneck (1×1 conv) | `genai/models.py` | `self.to_latent` | ~33 |
| Skip only if enabled | `genai/models.py` | `if self.skip` | ~47 |
| Transposed conv | `genai/models.py` | `def up_bn_act` | 10 |
| Param count | `genai/models.py` | `def count_params` | 79 |
| **Loss α·L1 + (1−α)(1−SSIM)** | `genai/losses.py` | `class L1SSIMLoss` | 24 |
| Optuna objective (fixed weights) | `genai/losses.py` | `def selection_objective` | 38 |
| PSNR / SSIM | `genai/losses.py` | `def psnr_per_image` | 15 |
| **Training loop, best checkpoint** | `genai/train_utils.py` | `def fit_restorer` | 58 |
| Optimizer / LR schedule | `genai/train_utils.py` | `AdamW` | ~64 |
| Best checkpoint kept | `genai/train_utils.py` | `best_state = copy.deepcopy` | ~98 |
| Final training + save | `scripts/task1_train.py` | `torch.save` | 60 |
| Skip ablation flag | `scripts/task1_train.py` | `--skip` | – |
| **Optuna search space** | `scripts/task1_optuna.py` | `suggest_float("lr"` | 40–45 |
| Pruner / sampler | `scripts/task1_optuna.py` | `MedianPruner` | 68 |
| Optuna plots | `scripts/task1_optuna.py` | `def save_optuna_plots` | 94 |
| Test eval (per type + severity) | `scripts/task1_eval.py` | `summary_tables` | – |
| 12 examples / failures / error map | `genai/evaluation.py` | `def restoration_grid` | 73 |
| Choosing failures | `genai/evaluation.py` | `def pick_failures` | 115 |
| Results files | `experiments/results/task1/` | `test_final_by_type_severity.csv` | – |
| ONNX export | `scripts/task1_export.py` | `export_onnx` | – |

## 3. Task 2 – Classifier + specialists
| Rubric item | File | Ctrl+F | Line |
|---|---|---|---|
| **Classifier CNN** | `genai/models.py` | `class CorruptionClassifier` | 59 |
| Cross-entropy training | `genai/classifier_utils.py` | `F.cross_entropy` | ~55 |
| Classifier train loop | `genai/classifier_utils.py` | `def fit_classifier` | 40 |
| Balanced batches used | `scripts/task2_classifier.py` | `mode="balanced"` | – |
| Classifier Optuna space | `scripts/task2_classifier.py` | `suggest_float("weight_decay"` | 42–46 |
| Accuracy, macro P/R/F1, per class | `genai/classifier_utils.py` | `def classification_report_dict` | 26 |
| **Normalized confusion matrix** | `genai/classifier_utils.py` | `cm_norm` | ~33 |
| Confusion plot | `genai/classifier_utils.py` | `def plot_confusion` | 86 |
| Specialists (own corruption only) | `scripts/task2_specialists.py` | `types=[expert]` | – |
| Shared specialist Optuna | `scripts/task2_specialists.py` | `def stage_optuna` | 37 |
| Specialist training | `scripts/task2_specialists.py` | `def stage_train` | 103 |
| **Hard routing (argmax + identity)** | `genai/routing.py` | `class HardRoutedRestorer` | 7 |
| Oracle vs predicted | `scripts/task2_eval.py` | `def oracle` | – |
| Misrouting analysis | `scripts/task2_eval.py` | `mis = p[p.pred != p.label]` | – |
| Results files | `experiments/results/task2/` | `hard_oracle_vs_predicted.csv` | – |
| ONNX export | `scripts/task2_export.py` | `export_onnx` | – |

## 4. Task 3 – Soft MoE
| Rubric item | File | Ctrl+F | Line |
|---|---|---|---|
| **Gate softmax(G/τ), weighted sum** | `genai/routing.py` | `class SoftMoERestorer` | 31 |
| Temperature in code | `genai/routing.py` | `logits / self.tau` | 46 |
| Init from Task 2 (not random) | `scripts/task3_moe.py` | `def build_moe` | 49 |
| **Warm-up: freeze experts** | `scripts/task3_moe.py` | `set_experts_trainable(moe, False)` | 145 |
| Joint fine-tune (unfreeze) | `scripts/task3_moe.py` | `set_experts_trainable(moe, True)` | 148 |
| **Joint loss (L1+SSIM+CE+balance)** | `scripts/task3_moe.py` | `lambda_c"] * ce` | ~107 |
| Balance loss | `scripts/task3_moe.py` | `def balance_loss` | 54 |
| Expert BN frozen | `scripts/task3_moe.py` | `e.eval()` | 96 |
| Optuna space (lr, τ, λc, λb, λ1) | `scripts/task3_moe.py` | `suggest_float("tau"` | 159–163 |
| Routing-collapse pruning | `scripts/task3_moe.py` | `routing collapse` | 132 |
| Avg weights per type × severity | `scripts/task3_moe.py` | `w_by = df.groupby` | – |
| Routing heatmap | `scripts/task3_moe.py` | `def routing_heatmap` | 259 |
| Weight distribution | `scripts/task3_moe.py` | `def weight_distribution_plot` | 286 |
| Inactive-expert check | `scripts/task3_moe.py` | `inactive_branches` | – |
| Compare with T1/T2 | `scripts/task3_moe.py` | `comparison_tasks123` | – |
| Mixed-corruption experiment | `scripts/task3_moe.py` | `def mixed_corruption_experiment` | 304 |
| ONNX export | `scripts/task3_moe.py` | `def stage_export` | 414 |
| **Live τ change** | `scripts/export_moe_with_tau.py` | `moe.tau = a.tau` | – |

## 5. Task 4 – cGAN
| Rubric item | File | Ctrl+F | Line |
|---|---|---|---|
| FS2K split 15% stratified seed 42 | `genai/fs2k.py` | `def prepare` | 51 |
| Stratify | `genai/fs2k.py` | `stratify=` | ~56 |
| Pairing + resize 128 | `genai/fs2k.py` | `def load_split` | 37 |
| **Same augmentation for pair** | `genai/fs2k.py` | `class PairDataset` | 67 |
| **U-Net generator + style embedding** | `genai/gan.py` | `class StyleUNetGenerator` | 21 |
| Embedding used in G | `genai/gan.py` | `self.style_to_bottleneck` | ~41 |
| **PatchGAN discriminator + style** | `genai/gan.py` | `class StylePatchDiscriminator` | 65 |
| D loss real / fake (BCE logits) | `scripts/task4_cgan.py` | `d_real = bce` | 100 |
| G loss adv + λ·L1 | `scripts/task4_cgan.py` | `lambda_l1"] * g_l1` | 110 |
| Losses logged separately | `scripts/task4_cgan.py` | `d_real_loss` | – |
| Fixed val photos every 5 epochs | `scripts/task4_cgan.py` | `val_samples/epoch_` | 125 |
| GAN Optuna (7 params) | `scripts/task4_cgan.py` | `suggest_float("lr_g"` | 140–146 |
| Final training | `scripts/task4_cgan.py` | `def stage_train` | 214 |
| Test eval per style + grids | `scripts/task4_cgan.py` | `def stage_eval` | 240 |
| ONNX wrapper ([0,1] in/out) | `genai/gan.py` | `class GeneratorExport` | 92 |

## 6. ONNX
| Rubric item | File | Ctrl+F | Line |
|---|---|---|---|
| Export function | `genai/onnx_utils.py` | `def export_onnx` | 9 |
| **PyTorch vs ONNX check** | `genai/onnx_utils.py` | `def verify_onnx` | 23 |
| Numbers (max diff) | `experiments/results/task1/onnx_check.json` | `max_abs_diff` | – |
| Download link (models) | `README.md` | `releases/download/v1.2` | – |

## 7. Backend (FastAPI)
| Rubric item | File | Ctrl+F | Line |
|---|---|---|---|
| Health endpoint | `backend/app/main.py` | `/api/health` | 92 |
| Universal endpoint | `backend/app/main.py` | `/api/restore/universal` | 130 |
| Hard routing endpoint | `backend/app/main.py` | `/api/restore/hard` | 148 |
| Soft MoE endpoint | `backend/app/main.py` | `/api/restore/soft` | 171 |
| Sketch endpoint | `backend/app/main.py` | `/api/sketch` | 188 |
| **File validation (415/413)** | `backend/app/main.py` | `def read_image` | 35 |
| Upload size limit | `backend/app/main.py` | `MAX_BYTES` | 25 |
| Runtime corruption in app | `backend/app/main.py` | `def corrupt` | 56 |
| Preprocess (EXIF, RGB, 128) | `backend/app/inference.py` | `def load_rgb` | 64 |
| Load ONNX models | `backend/app/inference.py` | `class Registry` | 29 |
| Model file names | `backend/app/inference.py` | `MODEL_FILES` | 17 |
| Timing info | `backend/app/main.py` | `timing_ms` | – |

## 8. Frontend (React + Tailwind)
| Rubric item | File | Ctrl+F | Line |
|---|---|---|---|
| 4 workspaces (sidebar) | `frontend/src/App.jsx` | `const NAV` | – |
| Universal / Hard / Soft pages | `frontend/src/pages/Restoration.jsx` | `const MODES` | 5 |
| Corruption dropdown + severity | `frontend/src/pages/Restoration.jsx` | `Runtime corruption` | – |
| Skip toggle | `frontend/src/pages/Restoration.jsx` | `With limited skip` | 108 |
| Classifier probabilities + expert | `frontend/src/pages/Restoration.jsx` | `mode === 'hard'` | 151 |
| 4 routing weights + strongest | `frontend/src/pages/Restoration.jsx` | `mode === 'soft'` | 171 |
| Inference time | `frontend/src/pages/Restoration.jsx` | `Inference time` | – |
| Face-to-sketch page | `frontend/src/pages/Sketch.jsx` | `Face-to-Sketch Generator` | – |
| **Webcam** | `frontend/src/components/ui.jsx` | `getUserMedia` | 87 |
| Download button | `frontend/src/components/ui.jsx` | `Download` | 11 |
| Compare 3 styles | `frontend/src/pages/Sketch.jsx` | `Compare all 3 styles` | 69 |
| Colours (Stitch palette) | `frontend/tailwind.config.js` | `#0b1020` | – |
| Stitch design images | `report/figures/stitch_2.png` | – | – |

## 9. Docker
| Rubric item | File | Ctrl+F |
|---|---|---|
| Compose (2 services, one command) | `docker-compose.yml` | `services:` |
| Model path / volume | `docker-compose.yml` | `MODELS_DIR` |
| Backend image | `backend/Dockerfile` | `uvicorn` |
| Corruption file copied into backend | `backend/Dockerfile` | `corruptions.py` |
| Frontend image (node → nginx) | `frontend/Dockerfile` | `FROM nginx` |
| /api proxy | `frontend/nginx.conf` | `proxy_pass` |
| Run instructions | `README.md` | `docker compose up --build` |

## 10. Repo, Optuna, MLflow, report
| Rubric item | Where (Ctrl+P) | Ctrl+F |
|---|---|---|
| Optuna studies (all 4 tasks) | `experiments/optuna/` (5 `.db` files) | – |
| Optuna trials / best params | `experiments/results/task1/best_params.json` (and `task2/3/4`) | `"trial"` |
| MLflow database | `experiments/mlflow/mlflow.db` | – |
| MLflow setup in code | `genai/paths.py` | `def setup_mlflow` |
| Final configs | `configs/final_configs.json` | `"tau"` |
| Seeds | `genai/train_utils.py` | `def set_seed` |
| Full pipeline script | `scripts/run_all.sh` | `stage t1_optuna` |
| Dependencies | `requirements.txt`, `backend/requirements.txt` | – |
| Report PDF | `report/GenAI_Assignment1_Report_EmanIhsan.pdf` | – |
| Report source | `report/main.tex` | `\section{` |
| Video link | `report/main.tex` | `youtu.be` |
| AI-use appendix | `report/main.tex` | `AI-Use Statement` |
| Viva Q&A | `VIVA_PREP.md` | – |
