# Submission guide — what you still need to do

## A. Push the code to GitHub (5 min)
1. Create an empty **public** repo on github.com named `GenAI_Assignment1`. Don't add a README.
2. In this folder:
   ```bash
   git remote add origin https://github.com/<your-username>/GenAI_Assignment1.git
   git push -u origin main
   ```
3. Upload `models_onnx.zip` (the 7 ONNX files) as a **GitHub Release** asset: repo → Releases → "Draft a new release" → tag `v1.0` → attach the zip. Copy the asset link into README.md (`<MODEL-DOWNLOAD-LINK>`) and push again.

## B. Google Stitch design evidence (10 min)
1. Open https://stitch.withgoogle.com, choose **Web**, and paste this prompt:

> Design a dark-themed web dashboard called "RestoreLab" for an image-restoration and face-to-sketch AI app. Left sidebar with the logo, five navigation items ("Universal Restoration – Task 1", "Hard-Routed Restoration – Task 2", "Soft Mixture-of-Experts Restoration – Task 3", "Face-to-Sketch Generator – Task 4", "System") and an API status pill at the bottom. Main area: page title and a one-line description. Left column of control cards: "1 · Input image" with a dashed upload drop zone and a 4×2 grid of sample thumbnails; "2 · Runtime corruption" with a dropdown (None, Clean, Salt & pepper, Gaussian blur, Occlusion), Low/Medium/High segmented buttons and a seed field; a large sky-blue "Run model" button. Right area: four square image cards in a row (Clean reference, Model input, Restored output, Absolute error map) with Download links, then a row of stat tiles (Inference time, Input PSNR, Output PSNR, Model). Below them, a card with horizontal probability bars for Clean / Salt & pepper / Gaussian blur / Occlusion with the top one highlighted, and a "Routing decision" card. Colours: navy background #0b1020, cards #111833, accent sky blue #38bdf8, Inter font, rounded 16px corners.

2. Generate the screens, then make variations for the **Soft MoE page** (bars for 4 routing weights plus a stacked contribution strip) and the **Face-to-Sketch page** (upload/webcam, Style 1/2/3 buttons, photo and sketch side by side).
3. Take screenshots of the Stitch canvas with the Stitch UI visible, so it's clear they come from Stitch. Save them to `report/figures/stitch_1.png` and `stitch_2.png`.

## C. App screenshots for the report
With the app running (`docker compose up --build`, http://localhost:8080), screenshot each workspace after a run and save them as
`report/figures/app_universal.png`, `app_hard.png`, `app_soft.png`, `app_sketch.png`, `app_system.png`.

## D. Report (Overleaf, 10 min)
1. Zip the `report/` folder, then on Overleaf: New Project → Upload Project → choose the zip.
2. Fill in your name, roll number, repo link and video link at the top of `main.tex`.
3. Compile and download the PDF.

## E. Demo video (5–7 min) → YouTube (unlisted is fine)
Record with OBS or the Windows Game Bar (Win+Alt+R). Suggested script:

| Time | Show |
|---|---|
| 0:00–0:40 | Repo on GitHub, README. In a terminal: `docker compose up --build` → containers start → open http://localhost:8080 |
| 0:40–1:00 | **System** page: 7/7 models loaded, ONNX signatures |
| 1:00–2:00 | **Universal Restoration**: upload a pet photo → apply salt & pepper (high) → Run. Point out restored output, error map, PSNR, inference time. Repeat with occlusion (medium). |
| 2:00–3:10 | **Hard-Routed**: same image with blur → classifier probabilities, predicted class, selected expert. Then "Clean": identity bypass. Upload an already-corrupted image with corruption = None. |
| 3:10–4:10 | **Soft MoE**: salt & pepper → 4 routing weights and the contribution strip; try blur and occlusion to show the weights move |
| 4:10–5:10 | **Face-to-Sketch**: upload a face (or webcam) → Style 1, 2, 3 → **Download** the result |
| 5:10–6:30 | **MLflow** (`mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db`): experiments, a run's metrics curves, logged sample images. Optuna results figures. |
| 6:30–7:00 | Wrap-up |

## F. What goes into Google Classroom
* The **report PDF**. It must contain the GitHub link and the YouTube link. Don't upload the video itself.
