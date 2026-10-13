# Rubric → where it is

**Report** = `report/GenAI_Assignment1_Report_EmanIhsan.pdf` (Sec = section, Fig/Tab = figure/table number in the PDF).
**Repo** = https://github.com/eman06/GenAI-A1 · **App** = http://localhost:8080 (on your PC: 8088).

## Data & corruption (Tasks 1–3)
| Rubric item | Code | Report |
|---|---|---|
| Load Pet, RGB, 128×128, stats + samples | `genai/data.py` (`load_pet_arrays`), `scripts/prepare_data.py` | Sec III, Tab I, Fig 2 |
| 80/20 split seed 42, test untouched, no overlap | `genai/data.py` `make_split`, `manifests/split.json` | Sec III, Tab I |
| Runtime corruption, equal prob., label returned | `genai/data.py` `RuntimeCorruptCollate` | Sec III |
| Training corruption ranges | `genai/corruptions.py` lines 25–29, `sample_train_spec` | Sec III |
| Val/test manifests (type, params, rects, seed) | `manifests/val_manifest.json`, `test_manifest.json`; `genai/data.py` `build_*_manifest` | Sec III |
| Fixed test severities + visual examples | `genai/corruptions.py` line 16 `TEST_SEVERITIES` | Sec III, Fig 3 |
| (proof) unit tests | `tests/test_corruptions.py` | Appendix |

## Task 1 – Universal DAE
| Rubric item | Code | Report |
|---|---|---|
| Encoder/bottleneck/decoder, no corruption label | `genai/models.py` `ConvAutoencoder` | Sec V, Fig 4 |
| No unrestricted skips; skip ablation; params + latent | `models.py` (`skip=False`), `scripts/task1_train.py --skip` | Tab II, Tab VI, Fig 5 |
| Loss α·L1 + (1−α)(1−SSIM) | `genai/losses.py` `L1SSIMLoss` | Sec V-B |
| Training loop, best checkpoint, curves, overfitting | `genai/train_utils.py` `fit_restorer`, `scripts/task1_train.py` | Sec V, Fig 8 |
| Optuna (lr, batch, bottleneck, channels, dropout, α) | `scripts/task1_optuna.py` | Sec IV, Tab III |
| Trials, best, plots | `experiments/optuna/task1_udae.db`, `experiments/results/task1/` | Tab III, Fig 9 |
| Results per corruption + per severity | `scripts/task1_eval.py` | Tab IV, Tab V |
| 12 examples with error maps | `genai/evaluation.py` `restoration_grid` | Fig 6 + Fig 10 |
| 4 failure cases explained | same | Fig 7, Sec V-C |
| Strengths/weaknesses discussion | – | Sec V-C, XIII |

## Task 2 – Classifier + hard routing
| Rubric item | Code | Report |
|---|---|---|
| CNN classifier, CE, auto labels, balanced batches | `genai/models.py` `CorruptionClassifier`, `genai/classifier_utils.py`, `RuntimeCorruptCollate(mode="balanced")` | Sec VI |
| Classifier Optuna | `scripts/task2_classifier.py optuna` | Tab III, Fig 13 |
| Accuracy, macro P/R/F1, per class | `scripts/task2_classifier.py eval` | Tab VIII |
| Normalized confusion matrix + interpretation | same | Fig 15, Sec VI |
| 3 specialists, own corruption only, params | `scripts/task2_specialists.py train` | Sec VI, Tab II |
| Shared specialist Optuna | `scripts/task2_specialists.py optuna` | Tab III |
| Specialist curves + per-severity results | `experiments/results/task2/specialist_*` | Fig 14, Tab VII |
| Hard routing (argmax, identity bypass) | `genai/routing.py` `HardRoutedRestorer`; `backend/app/main.py` `/api/restore/hard` | Fig 11 |
| Oracle vs predicted by corruption + severity | `scripts/task2_eval.py` | Tab VII, Tab IX |
| Misrouting failures with visuals | same | Fig 16, Sec VI |
| Comparison with Task 1 | – | Tab V, Fig 22–23 |

## Task 3 – Soft MoE
| Rubric item | Code | Report |
|---|---|---|
| Gate softmax(G/τ), 4 weights | `genai/routing.py` `SoftMoERestorer` | Sec VII |
| Weighted sum, differentiable, diagram | same | Fig 17 |
| Init from Task 2 (not random) | `scripts/task3_moe.py` `build_moe` | Sec VII |
| Warm-up (gate only) → joint, smaller lr | `scripts/task3_moe.py` `train_moe` / `run_stage` | Sec VII, Fig 26 |
| Joint loss incl. balance (justified) | `scripts/task3_moe.py` `run_stage`, `balance_loss` (line 54) | Sec VII + refs [5][6] |
| Optuna (lr, τ, λc, λb, λ1) + collapse pruning | `scripts/task3_moe.py optuna` | Tab III, Fig 21 |
| Results vs Task 1 / Task 2 | `scripts/task3_moe.py eval` | Tab V, Tab X, Fig 22–23 |
| Avg weights per type × severity | `experiments/results/task3/moe_weights_by_type_severity.csv` | Tab XI |
| Heatmap, dominant + spread examples | same script | Fig 24, Fig 18, Fig 25, Fig 19 |
| Inactive experts check, failures | `experiments/results/task3/moe_expert_usage.json` | Sec VII, Fig 20 |

## Task 4 – Face-to-sketch cGAN
| Rubric item | Code | Report |
|---|---|---|
| FS2K official split + 15% stratified val seed 42 | `genai/fs2k.py` `prepare`, `manifests/fs2k_split.json` | Sec VIII, Tab I, Fig 28 |
| 128×128, pairing, identical augmentation, sample pairs | `genai/fs2k.py` `PairDataset` | Sec VIII, Fig 33 |
| U-Net G with style embedding | `genai/gan.py` `StyleUNetGenerator` | Fig 27, Tab II |
| PatchGAN D with style embedding | `genai/gan.py` `StylePatchDiscriminator` | Fig 27, Tab II |
| BCE-with-logits + λ·L1, param counts | `scripts/task4_cgan.py` `train_gan` | Sec VIII, Tab II |
| Separate losses logged, fixed val samples | `train_gan` (MLflow `val_samples/`) | Fig 30, Fig 41 |
| GAN Optuna (7 params, short trials) | `scripts/task4_cgan.py optuna` | Tab III, Fig 31 |
| Best config retrained + curves | `scripts/task4_cgan.py train` | Sec VIII, Fig 30 |
| Test metrics overall + per style | `scripts/task4_cgan.py eval` | Tab XII |
| Grids all styles + failures | same | Fig 29, Fig 33, Fig 32 |

## App, ONNX, Docker
| Rubric item | Where |
|---|---|
| Google Stitch evidence | Report Fig 35, Fig 36; files `report/figures/stitch_*.png` |
| 4 workspaces React/Tailwind | `frontend/src/App.jsx`, `pages/Restoration.jsx`, `pages/Sketch.jsx`; Report Fig 37–40 |
| Universal (upload/sample, corruption, time) | `pages/Restoration.jsx` (mode `universal`) |
| Hard-routed (4 probs, expert, time) | `pages/Restoration.jsx` (mode `hard`) |
| Soft MoE (4 weights, strongest expert, time) | `pages/Restoration.jsx` (mode `soft`) |
| Face-to-sketch (upload/webcam, styles, download) | `pages/Sketch.jsx`, webcam in `components/ui.jsx` |
| FastAPI (validation, endpoints, timing) | `backend/app/main.py`, `backend/app/inference.py` |
| ONNX export (all models) + download link | `genai/onnx_utils.py`, `scripts/task*_export.py`; models: GitHub Release v1.2 |
| ONNX vs PyTorch difference | `experiments/results/task*/onnx_check.json`; Report Abstract, Sec XIII |
| Dockerfiles + compose, one command | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`; `docker compose up --build` |
| Clone → models → run (no VS Code) | `README.md` section 1 |

## Repo, tracking, report, video
| Rubric item | Where |
|---|---|
| Code, configs, requirements, scripts | `genai/`, `scripts/`, `configs/final_configs.json`, `requirements.txt` |
| README + model link, no big files, URL in report | `README.md`; Release v1.2; report title block |
| Optuna studies all 4 tasks | `experiments/optuna/*.db`, `experiments/results/task*/…optuna*` |
| MLflow records | `experiments/mlflow/mlflow.db`; Report Sec X, Fig 41–42 |
| Code quality, seeds, commit history | seeds in `genai/data.py`, `genai/train_utils.py`; GitHub commits |
| Abstract / Intro / Related work + refs | Report Abstract, Sec I, II, References |
| Methodology, Optuna design, decisions, difficulties | Sec III–IX, Tab III, Sec XI |
| Results + figures interpreted | Sec V–IX, Fig 1–42 |
| Conclusions, limitations, future work per task | Sec XII, XIII |
| Video link | Report title block → https://youtu.be/YC3jGST1eOY |
| Live changes (τ, severity, model path) | `VIVA_PREP.md` section 11; `scripts/export_moe_with_tau.py` |
| AI-use appendix | Report Appendix A |
