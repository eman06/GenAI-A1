# Viva / Live-Evaluation Prep — GenAI Assignment 1 (RestoreLab)

Read top to bottom once, then use **section 12 (numbers)** and **section 11 (live changes)** as your cheat sheet.

---

## 0. Your 60-second pitch (memorise this)

> "I built four generative models and deployed them in one web app.
> **Tasks 1–3** restore Oxford-IIIT Pet images corrupted at runtime with salt-and-pepper noise, Gaussian blur or black rectangles.
> **Task 1** is one universal denoising autoencoder with a real bottleneck (no skip connections). It removes heavy damage but caps quality around 22 dB.
> **Task 2** splits the problem: a CNN classifier (99.7% accurate) detects the corruption and sends the image to one of three specialist autoencoders; clean images skip restoration. That gives +5.3 dB.
> **Task 3** turns the hard router into a soft mixture-of-experts: the gate gives a weight to every expert plus an identity branch, everything is fine-tuned jointly, and it gets the best SSIM, 0.832.
> **Task 4** is a pix2pix-style conditional GAN on FS2K: a U-Net generator and a PatchGAN discriminator, both conditioned on a learned style embedding, turning a face photo into one of three sketch styles.
> Everything is tuned with Optuna, tracked in MLflow, exported to ONNX (checked against PyTorch to ~10⁻⁶), and served by FastAPI + React in Docker Compose."

---

## 1. Big picture — how the pieces connect

```
Oxford-IIIT Pet ──► split (seed 42) ──► runtime corruption (DataLoader collate) ──► T1, T2 ──► T3 (initialised from T2)
FS2K ──► official split + 15% val ──────────────────────────────────────────────────► T4
All training: Optuna (search) + MLflow (logging) ──► best config retrained ──► ONNX export + check
ONNX models ──► FastAPI backend (ONNX Runtime) ◄── React/Tailwind frontend (nginx) ── Docker Compose
```

**Where things live in the code**

| What | File |
|---|---|
| Corruptions (shared with backend) | `genai/corruptions.py` |
| Dataset, split, manifests, runtime-corruption collate | `genai/data.py` |
| Autoencoder + classifier | `genai/models.py` |
| Hard router + soft MoE | `genai/routing.py` |
| Losses / PSNR / SSIM | `genai/losses.py` |
| Training loop (restorers) | `genai/train_utils.py` |
| GAN (generator, discriminator) | `genai/gan.py`, FS2K loader `genai/fs2k.py` |
| Per-task scripts | `scripts/task1_*.py`, `task2_*.py`, `task3_moe.py`, `task4_cgan.py` |
| ONNX export + verification | `genai/onnx_utils.py` |
| Backend | `backend/app/main.py`, `backend/app/inference.py` |
| Frontend | `frontend/src/pages/Restoration.jsx`, `Sketch.jsx`, `System.jsx` |
| Docker | `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`, `frontend/nginx.conf` |
| Optuna studies / MLflow DB / results | `experiments/optuna/`, `experiments/mlflow/`, `experiments/results/` |
| Final hyper-parameters | `configs/final_configs.json` |

---

## 2. Data & corruption pipeline (Tasks 1–3)

**Q: How did you split the data?**
Official `trainval` (3,680 images) → shuffled with `np.random.default_rng(42).permutation` → 2,944 train / 736 val (80/20). Official `test` (3,669) is only used by the `*_eval.py` scripts. The split is saved once in `manifests/split.json` and reused by Tasks 1, 2 and 3. Train ∩ val = ∅ (checked).

**Q: Preprocessing?** RGB conversion, resize to 128×128 (bicubic), values scaled to [0, 1].

**Q: Where exactly does the corruption happen? Why not save corrupted images?**
In `RuntimeCorruptCollate` (`genai/data.py`), the DataLoader's `collate_fn`. Every time a batch is built, each image gets a **freshly sampled** corruption type and severity, so the model never sees the same corrupted version twice (infinite augmentation, less overfitting, nothing stored on disk). It also returns the **label** (0 clean, 1 s&p, 2 blur, 3 occlusion), which the classifier and gate train on.

**Q: Equal probability of the four conditions?**
- Task 1: `mode="uniform"`: each image independently draws one of 4 types with p = 1/4.
- Tasks 2–3: `mode="balanced"`: every batch has *exactly* the same number of each type (batch sizes are multiples of 4). This is what "balanced batches" means for the classifier.

**Q: Exact training corruption definitions?**
- **Salt & pepper:** p ~ U(0.02, 0.15); each selected pixel (all 3 channels) becomes black or white with 50/50 probability.
- **Blur:** kernel ∈ {3, 5, 7}, σ ~ U(0.5, 2.5); a separable Gaussian with reflect padding. A unit test proves it equals torchvision's blur to 10⁻⁵.
- **Occlusion:** 1–3 non-overlapping black rectangles jointly covering U(10%, 35%) of the image. Area is split between rectangles with a Dirichlet draw, aspect ratio log-uniform in [½, 2], placement random with rejection of overlaps, accepted if coverage is within ±1.5% of target.

**Q: Why are validation/test deterministic?**
To compare models fairly: every model sees *exactly the same* corrupted images. `val_manifest.json` stores one fixed corruption per val image. `test_manifest.json` stores, per test image, clean + 3 corruptions × 3 severities = **10 entries → 36,690 test cases**. Each entry stores type, parameters, rectangle coordinates and the seed.

**Q: Fixed test severities?**
s&p p = 0.03 / 0.08 / 0.15; blur (k, σ) = (3, 0.7) / (5, 1.5) / (7, 2.5); occlusion 1 / 2 / 3 rectangles ≈ 10 / 20 / 35%.

**Q: How does the app reproduce the exact corruptions?**
`genai/corruptions.py` is pure NumPy (no PyTorch). The backend Dockerfile copies *the same file* into the image, so the app's corruptions are identical to training/test.

**Q: Metrics?**
- **PSNR** = 10·log₁₀(1/MSE) for images in [0, 1]. Capped at 50 dB, because an identical image (clean input + identity bypass) gives MSE ≈ 0, i.e. ~100 dB, which would distort averages.
- **SSIM** (11×11 Gaussian window, `pytorch_msssim`) compares local luminance, contrast and structure; 1 = identical.
- **L1** = mean absolute pixel error.

---

## 3. Task 1 — Universal denoising autoencoder

**Architecture (2.25 M parameters)**
- Encoder: Conv3×3(3→64) → Conv stride 2 (128→64) → Conv; Conv stride 2 (64→128 ch, 64→32 px) → Conv; Conv stride 2 (128→256 ch, 32→16 px) → Conv. Each conv is followed by BatchNorm + ReLU, with Dropout2d (p = 0.25).
- **Bottleneck:** a 1×1 conv to **32 channels at 16×16 = 8,192 numbers**. The input has 128×128×3 = 49,152, so that's **6× compression**.
- Decoder: mirror image with ConvTranspose (4×4, stride 2): 16→32→64→128, then a 3×3 conv to 3 channels + **sigmoid** (output in [0, 1]).
- The model is **not told** the corruption type.

**Q: Why no skip connections?**
With unrestricted skips (as in a U-Net), the network can copy the input straight to the output and skip the bottleneck, so it's not a real autoencoder and can't learn a compressed representation. The assignment requires a genuine bottleneck.

**Q: You have a skip variant. Explain.**
Ablation: one skip at 64×64 (`--skip`). It raises overall PSNR from **22.19 → 27.95 dB** and SSIM from **0.792 → 0.873**, with the biggest gains on clean (+10.2 dB) and s&p (+7.7 dB) and the smallest on occlusion (+2.5 dB). Why: the skip carries high-resolution detail of the *visible* pixels, but it can't invent hidden content. It weakens the autoencoder constraint, so the **submitted model is skip-free**; the skip version is a toggle in the app to show the trade-off.

**Loss:** `L = α·L1 + (1−α)·(1−SSIM)`, with **α = 0.60** chosen by Optuna (the suggested start was 0.8). L1 keeps pixel accuracy; SSIM keeps edges, structure and contrast. SSIM is computed in fp32 because it was unstable in fp16 mixed precision.

**Training:** AdamW (weight decay 1e-5), one-cycle LR (max 3.6e-4), batch 16, 35 epochs, mixed precision. The **best checkpoint is kept by validation objective**.

**Q: Why is the Optuna objective not the training loss?**
The objective is val **L1 + (1 − SSIM) with fixed equal weights**. If each trial were scored with its *own* α-weighted loss, trials with different α would be on different scales, and Optuna would favour whatever α makes the number smallest, not the best model.

**Optuna:** 12 trials × 6 epochs, TPE sampler (seed 42), median pruner, **8 pruned / 4 complete**. Search space: lr [1e-4, 3e-3] log, batch {16, 32, 64}, latent channels {4, 8, 16, 32}, base channels {16, 32, 48, 64}, dropout [0, 0.3], α [0.5, 0.95]. Best: lr 3.6e-4, batch 16, latent 32, base 64, dropout 0.25, α 0.60.

**Convergence / overfitting:** train loss 0.434 → 0.122; best val SSIM 0.814 at epoch 33 of 35. Val tracks train, so there's **no overfitting**, because runtime corruption is strong augmentation. The plateau shows the model is limited by **bottleneck capacity**, not training time.

**Results (test PSNR / SSIM):** clean 23.29 / 0.849 · s&p 23.16 / 0.834 · blur 22.77 / 0.791 · occlusion 20.29 / 0.734 · **overall 22.19 / 0.792**.

**Q: Strengths and weaknesses?**
- Strong on heavy s&p (13 → 23 dB) and occlusion (10.7 → 18.9 dB at high severity).
- Output quality is almost **constant ~23 dB**, so it *damages* clean images (50 → 23 dB) and mild blur (32 → 23 dB).
- **Hardest corruption: occlusion**, because missing content must be hallucinated.
- Failure cases:
  - cats on pure black backgrounds get a grey "average" background;
  - saturated scenes lose their colour (the model hedges toward the mean);
  - large occlusions are filled with blur.
- This motivates Task 2: clean images should simply bypass the model.

---

## 4. Task 2 — Classifier + hard-routed specialists

**Classifier (1.68 M parameters):** 4 VGG-style blocks (two 3×3 conv + BN + ReLU, then max-pool) with channels **48–96–192–256**, global average pooling, dropout 0.39, linear layer → 4 logits. Loss: **multiclass cross-entropy**. Labels come automatically from the runtime pipeline. **Batches are exactly balanced.**

**Optuna:** 12 trials × 5 epochs (4 complete / 8 pruned). Objective: **validation macro-F1**. Search: lr, batch {32, 64, 128}, channel config {small, medium, wide}, dropout [0, 0.5], weight decay [1e-6, 1e-2]. Best: wide, lr 5.4e-4, batch 64, dropout 0.39, wd 6e-6.

**Q: Why macro-F1 and not accuracy?**
Macro-F1 averages F1 over the 4 classes equally, so a weak class can't hide behind strong ones. The test set is also imbalanced (3,669 clean vs 11,007 per corruption), so accuracy alone would be dominated by the corruption classes.

**Results:** accuracy **99.71%**, macro P/R/F1 0.994 / 0.996 / **0.995**. Per class: s&p 1.000; blur 0.999; occlusion recall 0.993; clean precision 0.978, recall 0.993.

**Confusion matrix interpretation:**
- 2.1% of **low** occlusions are called **clean**: one small black box landing on a dark background looks clean.
- 0.7% of clean images are called occlusion or blur: dark photos, or out-of-focus photos.
- Salt & pepper is perfect: its isolated black/white pixels are very distinctive.

**Specialists:** three copies of the Task 1 architecture (64 base channels, latent 32), each trained **only on its own corruption** with the clean image as target, with **independent weights**. A **shared Optuna search** (8 trials; each trial trains all 3 for 4 epochs, scored by their mean objective) picked α = 0.87, lr 3.6e-4, batch 16. Then each was trained for 25 epochs. Val SSIM: s&p 0.855, blur 0.850, occlusion 0.735 (still rising slightly at epoch 25).

**Hard routing:** `r = argmax(p)`. If r = clean → **identity bypass** (no network runs). Otherwise run only that specialist.

**Oracle vs predicted:** oracle routes with the true label from the test manifest, predicted with the classifier. Overall **27.48 vs 27.46 dB**, a 0.02 dB difference, because the classifier is almost perfect. Both are **+5.3 dB over Task 1**.

**Misrouting failures (108 cases):**
- **clean → occlusion** (15): an unnecessary inpainting pass blurs a perfect image (SSIM −0.32).
- **clean → blur** (9).
- **occlusion → clean** (76) actually *raised* SSIM on average: a small black box costs less than the occlusion expert smoothing the whole image.

**Q: When do specialists help or fail?**
- Help: s&p (+3.5 dB at every severity) and clean (bypass).
- Fail: **mild blur**, where every learned restorer (~27 dB) is worse than doing nothing (32.3 dB input).
- Fix: bypass mildly blurred images, or use the skip variant.

---

## 5. Task 3 — Soft mixture-of-experts

**Equations**
- `w = softmax(G(x̃) / τ)` gives 4 weights: identity, s&p expert, blur expert, occlusion expert.
- `x̂ = w₀·x̃ + w₁·A_sp(x̃) + w₂·A_blur(x̃) + w₃·A_occ(x̃)`, a weighted sum, so it's fully **differentiable** and gate and experts train together.

**Initialisation (not random):** gate = the Task 2 classifier's weights; experts = the 3 Task 2 specialists.

**Training:**
1. **Warm-up** (2 epochs): experts frozen, gate only, lr 1e-3.
2. **Joint fine-tuning** (12 epochs): everything unfrozen, lr **3.6e-5** (28× smaller, so the pretrained experts aren't destroyed), cosine schedule, batch 32. The experts' BatchNorm statistics are kept frozen (standard practice when fine-tuning).

**Loss:** `λ₁·L1 + (1−λ₁)·(1−SSIM) + λ_c·CE(gate logits, label) + λ_b·Σ_k (w̄_k − ¼)²`
- **CE** keeps the gate meaningful: it must still recognise the corruption.
- **Balance loss** uses w̄_k = the *batch-average* weight of branch k. Because batches are exactly balanced (¼ of each class), the *average* routing should be uniform; individual images can still be routed sharply. This is the idea of load-balancing losses in Shazeer et al. 2017 and Switch Transformers (Fedus et al. 2022). A per-image entropy penalty would wrongly force *every* image to spread its weights.

**Optuna:** 6 trials (4 complete / 2 pruned). Search: joint lr [1e-5, 3e-4], τ [0.5, 2], λ_c [0.01, 0.5], λ_b [1e-3, 0.1], λ₁ [0.5, 0.95]. Trials are also **pruned on routing collapse** (any branch with mean weight < 2%). Best: τ **1.93**, λ_c 0.175, λ_b 0.016, λ₁ 0.57, lr 3.6e-5.

**Q: What does temperature τ do?**
It divides the logits before the softmax. τ → 0 gives argmax, i.e. hard routing; large τ gives near-uniform weights. Optuna chose a high τ (1.93), which keeps gradients flowing to non-selected experts during joint training. *(Live demo below: τ = 0.3 gives weights [0, 0, 0, 1]; τ = 20 gives roughly uniform.)*

**Results:** overall **27.48 dB / SSIM 0.832**, the **best SSIM of all systems**. Clean 48.7 dB via the identity branch. Gains over hard routing on s&p (26.90 vs 26.71) and blur (26.43 vs 26.10).

**Gate analysis:**
- Almost diagonal: s&p images give 0.96–0.98 to the s&p expert; blur 0.91–0.93; medium/high occlusion ≥ 0.99.
- **Low occlusion: 0.23 identity + 0.75 occlusion expert.** This is a real *soft blend*: don't smooth the 90% of the image that isn't covered.
- **No inactive expert** (mean weights 0.14 / 0.29 / 0.28 / 0.28; identity is lower only because clean images are 10% of the test set). Off-target weight is ≤ 8.8%, so no expert dominates unrelated inputs.
- 88% of images have max weight > 0.9 (dominant); 1.6% have max weight < 0.6 (spread).

**Failures:**
- A **clean cat on a black background** gets 0.94 weight on the **occlusion** expert: the gate mistakes the black background for a mask.
- Saturated green/purple scenes get a brownish colour shift.

**Negative result:** with **blur + salt-and-pepper at once**, both hard and soft routing put ~all weight on the s&p expert (24.6 vs 24.7 dB). The gate was trained only on single corruptions, so it's overconfident on mixtures. The fix: train on mixtures with soft targets.

---

## 6. Task 4 — Style-conditioned face-to-sketch cGAN

**Data:** FS2K, 2,104 photo–sketch pairs, 3 styles (the `style` field in `anno_*.json`). The official train set (1,058) is split 85/15 → **899 / 159**, **stratified by style**, seed 42. Official test **1,046** is never used for training or tuning. Photos RGB, sketches grayscale, 128×128, scaled to [−1, 1]. The test set is imbalanced (619 / 381 / 46 per style).

**Q: Augmentation, and why "identical for both"?**
A random horizontal flip, plus an 8-pixel reflect-pad and random crop. **The same random numbers are applied to photo and sketch.** Otherwise the pixel correspondence breaks and the L1 loss would teach the wrong mapping.

**Generator (29.3 M params), a U-Net:**
- 6 down blocks (Conv 4×4 stride 2, BN, LeakyReLU 0.2): 128 → 2 px, channels 64-128-256-512-512-512.
- 5 up blocks (ConvTranspose, BN, ReLU, dropout 0.30 in the first three) with **skip connections**, ending in ConvTranspose → 1 channel + **tanh**.
- **Style embedding:** `nn.Embedding(3, 32)`, a learned vector per style, used **twice**:
  1. tiled to 32×128×128 and **concatenated with the photo** at the input;
  2. projected and **added at the 2×2 bottleneck**.
- (Skips are fine here: a GAN generator *should* pass fine detail through; only Task 1 forbids them.)

**Discriminator (2.8 M), a PatchGAN:**
- Input: photo (3) + sketch (1) + tiled style embedding (32) → C64 (no BN) → C128 → C256 → C512 → 1 channel.
- Output: **14×14 logits**, each judging a ~70×70 patch as real or fake. It judges local texture and strokes rather than the whole image: fewer parameters and sharper output.

**Losses (BCE with logits):**
- `L_D = ½ [BCE(D(x, y, s), 1) + BCE(D(x, G(x, s), s), 0)]`
- `L_G = BCE(D(x, G(x, s), s), 1) + λ_L1 · |y − G(x, s)|₁`
- λ_L1 = **121** (Optuna; pix2pix uses 100). The adversarial term gives sharpness; L1 keeps the output close to the real sketch.
- Adam (β₁ 0.5, β₂ 0.999), weights initialised N(0, 0.02).

**Optuna:** 4 trials × 8 epochs (3 complete / 1 pruned; shortened because of GPU time). Search: lr_G, lr_D, batch {4, 8, 16}, base channels {32, 48, 64}, dropout, embedding dim {8, 16, 32}, λ_L1 [10, 200]. Best: lr_G 1.2e-4, lr_D 4.3e-4, batch 4, base 64, dropout 0.30, emb 32, λ 121. Retrained for **40 epochs** (constant LR, then linear decay).

**Training dynamics (logged separately in MLflow):**
- D real/fake loss 0.76 → 0.16; G adversarial loss 0.78 → 2.64. The discriminator gets stronger towards the end.
- L1 0.29 → 0.21. Val SSIM peaked at epoch 22 (0.477), final 0.452.
- **The final generator is deployed**, because pixel metrics prefer blurry output while later epochs have crisper strokes.
- The same 8 validation photos are logged as an image grid every 5 epochs.

**Results (test SSIM / PSNR):** Style 1: 0.472 / 16.7 · **Style 2: 0.348 / 11.8 (hardest: dense pencil hatching)** · Style 3: 0.582 / 18.3 · overall 0.432 / 15.0.

**Q: Why are the SSIM numbers "low"?**
Sketching isn't a single correct answer. Strokes in a slightly different place are still a good sketch but get a low pixel score. That's why the **style-control grid** (same photo in 3 styles) is the main qualitative evidence. FID or LPIPS would be better metrics (future work).

**Failure cases:** a repeated inverted-V "nose" stroke appears on the worst cases. That's a template artefact, **mild mode collapse**, caused by the small data (899 pairs). Long hair is drawn as vertical strokes. Fixes: more data, a perceptual (VGG) loss, feature matching, spectral normalisation.

---

## 7. Optuna & MLflow

**Optuna terms:**
- **TPE** (Tree-structured Parzen Estimator): models "good" and "bad" regions of past trials and samples where the good-to-bad ratio is high. It's smarter than grid or random search.
- **Median pruner:** stops a trial early if its intermediate value is worse than the median of previous trials at the same step.
- Studies are stored as SQLite files in `experiments/optuna/*.db`, so they can be resumed.
- The first trial of each study was the assignment's suggested defaults (`enqueue_trial`).

**MLflow:** logs params, per-epoch metrics, test metrics, image grids and checkpoints. **61 runs in 4 experiments.** It uses an SQLite backend (the plain-file store is in maintenance mode in recent MLflow). To view it:
```
mlflow ui --backend-store-uri sqlite:///experiments/mlflow/mlflow.db --port 5050
```

**Reproducibility:** fixed seeds (split 42, Optuna sampler 42, training 42, per-image manifest seeds); `configs/final_configs.json`; `scripts/run_all.sh` (resumable: skips finished stages).

---

## 8. ONNX

**Q: Why ONNX?** It's framework-independent, so the backend doesn't need PyTorch (a much smaller image). It's fast on CPU with ONNX Runtime.

**Export:** `torch.onnx.export` (opset 17, dynamic batch dimension).

**Wrappers:**
- classifier → softmax probabilities;
- MoE → (image, weights);
- generator → takes a photo in [0, 1] and a style int64, converts internally to [−1, 1], and returns a sketch in [0, 1].

**Verification:** run the same real test images through PyTorch and ONNX Runtime and compare outputs. **Max absolute difference ≤ 1.5×10⁻⁶ for all 8 models** (files: `experiments/results/task*/onnx_check.json`).

**8 models:** universal DAE, its skip variant, classifier, 3 specialists, the complete soft-MoE pipeline (gate + experts in one graph), and the generator. The discriminator isn't exported; it's only needed for training.

---

## 9. Application

**Backend (FastAPI):**
- Endpoints: `/api/health`, `/api/info`, `/api/restore/universal` (with `variant=bottleneck|skip`), `/api/restore/hard`, `/api/restore/soft`, `/api/sketch`, `/api/corrupt`, `/api/samples`.
- **Validation:** content type must be PNG/JPEG/WebP/BMP (otherwise **415**); max 10 MB (**413**); undecodable image (**400**); bad level, style or variant (**422**); missing model (**503**).
- Preprocessing: fix EXIF rotation → RGB → resize to 128×128 bicubic (same as training).
- If the user applies a corruption in the app, the backend knows the clean image, so it returns PSNR/L1 and an error map against it. For an uploaded already-corrupted image there's no clean reference.
- Returns base64 PNGs + JSON (probabilities, weights, timing in ms per stage).

**Frontend (React 18 + Vite + Tailwind 3):**
- Sidebar with the 4 workspaces + System; health polled every 15 s.
- Restoration pages: upload or sample, corruption + severity + seed, 4 image panels, stats, probability bars (Task 2), weight bars + contribution strip (Task 3), variant toggle (Task 1).
- Sketch page: upload / **webcam** (`getUserMedia`), Style 1/2/3, side by side, compare all styles, download.
- Designed first in **Google Stitch**; the implementation follows its layout and colours (#0b1020, #111833, #38bdf8).

**Docker:**
- Two containers. **backend** (python:3.11-slim + onnxruntime + uvicorn, with a healthcheck). **frontend** (a multi-stage build: Node builds the React app, then nginx serves it and proxies `/api` to `backend:8000`).
- Models mounted **read-only** from `./models_onnx`.
- One command: `docker compose up --build` → http://localhost:8080. Ports are configurable with `.env` (on your PC: 8088/8010).

---

## 10. Theory questions they may throw at you

| Question | Short answer |
|---|---|
| Autoencoder vs PCA? | An AE is non-linear (convolutions + ReLU); a linear AE with MSE loss learns the same subspace as PCA. |
| Denoising AE idea? | Input corrupted, target clean, so it learns robust features (Vincent et al. 2008). |
| L1 vs L2 loss? | L2 punishes big errors more and gives blurrier results; L1 is more robust to outliers and sharper. |
| SSIM formula idea? | Compares local means (luminance), variances (contrast) and covariance (structure) in a sliding window. |
| PSNR? | 10·log₁₀(MAX²/MSE); higher is better; ~+6 dB means about half the error amplitude. |
| BatchNorm? | Normalises activations per batch; faster, more stable training; uses running statistics in eval mode. |
| Dropout / Dropout2d? | Randomly zero activations (whole channels for 2d), a regulariser. |
| Cross-entropy? | −log(probability of the true class); combined with softmax for multiclass problems. |
| Softmax temperature? | softmax(z/τ): small τ → peaked, large τ → flat. |
| Mixture of experts? | A gate weights several expert networks (Jacobs et al. 1991); "sparse" MoE picks top-k. |
| Routing collapse? | The gate sends everything to one expert, the others stop learning; fixed with a balance loss. |
| GAN objective? | A min-max game: D tells real from fake, G tries to fool D. |
| Conditional GAN? | Both G and D receive the condition (here: photo + style). |
| Why BCE with logits? | Sigmoid + BCE in one numerically stable function. |
| U-Net? | Encoder–decoder with skip connections at each resolution, preserving fine detail. |
| PatchGAN? | D outputs a grid of real/fake decisions, one per patch: a texture critic. |
| Mode collapse? | G produces limited variety and ignores input differences (our repeated nose stroke). |
| Why 2 learning rates in a GAN? | To balance the D/G training speed; Optuna picked lr_D ≈ 3.6 × lr_G. |
| Macro vs micro F1? | Macro = average over classes (equal weight); micro = computed over all samples. |
| Row-normalised confusion matrix? | Each row (true class) sums to 1, so the diagonal = recall per class. |
| Overfitting signs? | Train loss falls while val loss rises. Ours: val tracks train (runtime augmentation). |
| Why AdamW? | Adam with decoupled weight decay (better regularisation). |
| One-cycle LR? | Warm up to max LR, then anneal: fast, stable convergence. |
| Mixed precision? | fp16 forward pass for speed/memory; we compute the SSIM loss in fp32 for stability. |

---

## 11. Live changes they might ask for (practise these!)

All commands run from the project folder in PowerShell. After a backend change, restart it: `docker compose restart backend` (or `docker compose up --build -d backend` if you changed Python code).

1. **Change MoE temperature τ** (best demo):
   ```
   python scripts/export_moe_with_tau.py --tau 0.3
   docker compose restart backend
   ```
   → Soft MoE weights become ~[0, 0, 0, 1] (hard routing). Try `--tau 20` → almost uniform. Undo: `python scripts/export_moe_with_tau.py --restore` then restart.
   *Explain:* τ only rescales the gate logits; the weights don't change.

2. **Change corruption severity:** easiest in the UI (Low/Medium/High + seed). In code: `genai/corruptions.py` line 16, `TEST_SEVERITIES` (e.g. s&p `0.15` → `0.30`), then `docker compose up --build -d backend` (the Dockerfile copies this file).

3. **Change the model path:** `docker-compose.yml`, `volumes: ./models_onnx:/models:ro` and `MODELS_DIR=/models`. Point the volume at another folder; `/api/health` and the System page show which models loaded or are missing.

4. **Change the upload limit:** `backend/app/main.py` line 25, `MAX_BYTES = 10 * 2 ** 20` (10 MB). Allowed types are on line 26.

5. **Change the CPU threads used by ONNX Runtime:** `docker-compose.yml` `ORT_THREADS=2`.

6. **Change ports:** the `.env` file (`FRONTEND_PORT`, `BACKEND_PORT`).

7. **Swap the Task 1 model:** in the app, choose **Model variant → With limited skip**; in code, the `variant` form field of `/api/restore/universal`.

8. **Training-side changes** (explain; no need to run): training corruption ranges at `genai/corruptions.py` lines 25–29; Optuna search spaces in each script's `objective()` (e.g. `scripts/task1_optuna.py`); the MoE balance loss is `balance_loss()` in `scripts/task3_moe.py` (line 54).

9. **Show an API call directly:** open http://localhost:8010/docs (FastAPI's automatic Swagger UI) → `/api/restore/hard` → Try it out.

---

## 12. Numbers cheat sheet

| Item | Value |
|---|---|
| Pet split | 2,944 / 736 / 3,669 test → 36,690 test cases |
| FS2K split | 899 / 159 / 1,046 test |
| T1 params / latent | 2.25 M / 16×16×32 = 8,192 (6× compression) |
| T1 best | lr 3.6e-4, batch 16, base 64, dropout 0.25, **α 0.60**, 35 epochs |
| T1 test | **22.19 dB / 0.792**; skip variant 27.95 / 0.873 |
| Classifier | 1.68 M, wide 48-96-192-256, **99.71% acc, macro-F1 0.995** |
| Specialists | α 0.87, 25 epochs, val SSIM 0.855 / 0.850 / 0.735 |
| Hard routing | predicted 27.46 vs oracle 27.48 dB (+5.3 dB vs T1); 108 misrouted |
| Soft MoE | 8.42 M, τ 1.93, λ_c 0.175, λ_b 0.016, λ₁ 0.57, **27.48 dB / 0.832** |
| MoE weights | low occlusion 0.23 identity / 0.75 occlusion; no inactive expert |
| Generator / D | 29.3 M / 2.8 M; 14×14 patches; emb 32; λ_L1 121; 40 epochs |
| T4 test SSIM | 0.472 / 0.348 / 0.582 (styles 1/2/3) |
| Optuna trials | T1 12 (8 pruned) · clf 12 (8) · spec 8 (5) · MoE 6 (2) · GAN 4 (1) |
| ONNX check | max diff ≤ 1.5×10⁻⁶, 8 models |
| MLflow | 61 runs, 4 experiments |

---

## 13. Honest weaknesses (say these before they find them)

- **Small Optuna budgets** (4–12 trials) because of free Colab GPU time; the configurations are good, not optimal.
- **128×128 resolution**, so fine fur is lost.
- **Universal DAE hurts clean and mild-blur images**, the motivation for routing.
- **The MoE gate is overconfident on mixed corruptions** (trained on single corruptions).
- **Black-background confusion** (clean → occlusion).
- **GAN template artefacts / mild mode collapse** on the small dataset; pixel metrics are a weak proxy for sketch quality (FID/LPIPS would be better).
- **Two Colab disconnections**, handled by a resumable, Drive-backed `run_all.sh`.
- **AI use:** Claude was used for code and report drafting. *I verified it* by running everything end to end on synthetic data first, with unit tests, ONNX numerical checks and API tests with bad uploads. **Be ready to explain any file**: open it and walk through it.
