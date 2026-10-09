"""Report v2: video link, dataset/corruption figures, model + Optuna summary tables, skip ablation,
oracle-vs-predicted by severity, comparison charts, extra curves/failure figures, Stitch, MLflow, tracking."""
p = 'report/main.tex'
s = open(p, encoding='utf-8').read()


def rep(old, new, count=1):
    global s
    assert old in s, f'anchor not found: {old[:70]}'
    s = s.replace(old, new, count)


def fig(name, cap, label, width=r'\linewidth', star=False):
    env = 'figure*' if star else 'figure'
    return f"\\begin{{{env}}}[t]\\centering\\includegraphics[width={width}]{{{name}}}\\caption{{{cap}}}\\label{{{label}}}\\end{{{env}}}\n"


# ---- video link
rep(r'\newcommand{\video}{YouTube link to be added}', r'\newcommand{\video}{\url{https://youtu.be/YC3jGST1eOY}}')

# ---- pipeline overview in the introduction
rep(r'\section{Related Work}', fig('pipeline.png', 'End-to-end pipeline: data preparation and runtime corruption, training of the four tasks with Optuna and MLflow, ONNX export and the containerised application.', 'fig:pipeline', star=True) +
    r"Fig.~\ref{fig:pipeline} summarises the complete system. All code, Optuna studies, the MLflow tracking database and every result file are in the repository (\repo), and the demonstration video is at \video." + "\n\n" + r'\section{Related Work}')

# ---- dataset statistics, sample images, corruption gallery, no-overlap check
rep(r"\textbf{Metrics.} PSNR (capped",
    r"""\textbf{Dataset statistics.} Table~\ref{tab:data} lists the image counts of every split; the train and validation index sets were checked to be disjoint (intersection $=\emptyset$, union $=3{,}680$), and the same \texttt{split.json} is reused by Tasks 1--3. Fig.~\ref{fig:petsamples} shows test images after RGB conversion and resizing, and Fig.~\ref{fig:gallery} shows one test image under every corruption at every fixed test severity, generated from its entries in the committed test manifest.
\begin{table}[t]\centering\caption{Dataset splits used in this work.}\label{tab:data}\setlength{\tabcolsep}{4pt}
\begin{tabular}{llr}\toprule
Dataset & Split & Images \\ \midrule
Oxford-IIIT Pet & train (80\% of trainval, seed 42) & 2{,}944 \\
 & validation (20\%) & 736 \\
 & official test (untouched until evaluation) & 3{,}669 \\
 & test cases (clean + 3 corruptions $\times$ 3 severities) & 36{,}690 \\ \midrule
FS2K & train (85\% of official train) & 899 \\
 & validation (15\%, stratified by style) & 159 \\
 & official test & 1{,}046 \\ \bottomrule
\end{tabular}\end{table}
""" + fig('data_pet_samples.png', 'Oxford-IIIT Pet test images after RGB conversion and resizing to $128\\times128$ (the same images are bundled as samples in the application).', 'fig:petsamples') +
    fig('data_corruption_gallery.png', 'Every corruption at every fixed test severity for one test image (parameters and rectangle coordinates taken from the deterministic test manifest; the occlusion titles report the exact covered area).', 'fig:gallery') +
    r"\textbf{Metrics.} PSNR (capped")

# ---- model summary + Optuna summary before Task 1
rep(r'\section{Task 1: Universal Denoising Autoencoder}',
    r"""\section{Models and Hyper-parameter Search Overview}
Table~\ref{tab:models} summarises the architectures with their learnable parameters, and Table~\ref{tab:optuna} the five Optuna studies (TPE sampler with seed 42 and median pruning; all studies are stored as SQLite databases in \texttt{experiments/optuna/}). Every study used shortened trials, and the selected configuration was then retrained for the full schedule.
\begin{table}[t]\centering\caption{Model summary (final configurations).}\label{tab:models}\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrl}\toprule
Model & Params & Key dimensions \\ \midrule
T1 universal DAE & 2.25\,M & latent $16{\times}16{\times}32=8{,}192$ (6$\times$ compression) \\
T1 + limited skip (ablation) & 2.28\,M & one $64{\times}64$ skip into the decoder \\
T2 classifier & 1.68\,M & channels 48--96--192--256, GAP, 4 logits \\
T2 specialist ($\times3$) & 2.25\,M each & same as T1, trained per corruption \\
T3 soft MoE (complete) & 8.42\,M & gate + 3 experts + identity branch \\
T4 generator $G$ & 29.29\,M & U-Net, $128\rightarrow2\rightarrow128$, emb.\ dim 32 \\
T4 discriminator $D$ & 2.80\,M & PatchGAN, $14{\times}14$ logits \\ \bottomrule
\end{tabular}\end{table}
\begin{table*}[t]\centering\caption{Optuna studies: search spaces, trials and selected configurations.}\label{tab:optuna}\setlength{\tabcolsep}{3pt}\footnotesize
\begin{tabular}{p{2.1cm}p{6.6cm}p{1.6cm}p{2.3cm}p{4.3cm}}\toprule
Study & Search space & Trials (compl./pruned) & Objective & Selected configuration \\ \midrule
T1 universal DAE & lr $[10^{-4},3{\cdot}10^{-3}]$ log; batch \{16,32,64\}; latent ch.\ \{4,8,16,32\}; base ch.\ \{16,32,48,64\}; dropout $[0,0.3]$; $\alpha\in[0.5,0.95]$; 6 epochs/trial & 12 (4/8) & val L1 + (1$-$SSIM) & lr $3.6{\cdot}10^{-4}$, batch 16, latent 32, base 64, dropout 0.25, $\alpha=0.60$ \\
T2 classifier & lr $[10^{-4},5{\cdot}10^{-3}]$ log; batch \{32,64,128\}; channels \{small, medium, wide\}; dropout $[0,0.5]$; weight decay $[10^{-6},10^{-2}]$ log; 5 epochs/trial & 12 (4/8) & val macro-F1 & wide, lr $5.4{\cdot}10^{-4}$, batch 64, dropout 0.39, wd $6{\cdot}10^{-6}$ \\
T2 specialists (shared) & lr; batch \{16,32,64\}; latent ch.; base ch.; $\alpha$; each trial trains all three experts for 4 epochs & 8 (3/5) & mean val obj.\ of 3 experts & lr $3.6{\cdot}10^{-4}$, batch 16, latent 32, base 64, $\alpha=0.87$ \\
T3 soft MoE & joint lr $[10^{-5},3{\cdot}10^{-4}]$ log; $\tau\in[0.5,2]$; $\lambda_c\in[0.01,0.5]$; $\lambda_b\in[10^{-3},0.1]$; $\lambda_1\in[0.5,0.95]$ ($\lambda_s=1-\lambda_1$); 1+3 epochs, collapse pruning & 6 (4/2) & val L1 + (1$-$SSIM) & lr $3.6{\cdot}10^{-5}$, $\tau=1.93$, $\lambda_c=0.175$, $\lambda_b=0.016$, $\lambda_1=0.57$ \\
T4 cGAN & lr$_G$ $[5{\cdot}10^{-5},5{\cdot}10^{-4}]$; lr$_D$ $[2{\cdot}10^{-5},5{\cdot}10^{-4}]$; batch \{4,8,16\}; base ch.\ \{32,48,64\}; dropout $[0,0.5]$; emb.\ dim \{8,16,32\}; $\lambda_{L1}\in[10,200]$ log; 8 epochs/trial & 4 (3/1) & val L1 + (1$-$SSIM) & lr$_G$ $1.2{\cdot}10^{-4}$, lr$_D$ $4.3{\cdot}10^{-4}$, batch 4, base 64, dropout 0.30, emb.\ 32, $\lambda_{L1}=121$ \\ \bottomrule
\end{tabular}\end{table*}

\section{Task 1: Universal Denoising Autoencoder}""")

# ---- Task 1: convergence + skip ablation
rep(r"The best configuration was retrained for 35 epochs (AdamW, one-cycle LR, AMP).",
    r"The best configuration was retrained for 35 epochs (AdamW with weight decay $10^{-5}$, one-cycle LR schedule, mixed precision, batch 16); the checkpoint with the best validation objective is kept. \textbf{Convergence.} The training loss falls from 0.434 to 0.122 and validation SSIM reaches its best value of 0.814 at epoch 33 (0.813 at epoch 35, Fig.~\ref{fig:t1curve}). Validation tracks training closely, so there is no sign of overfitting: runtime corruption gives the model a new corrupted version of every image in every epoch, which acts as strong data augmentation. The small gain over the last epochs suggests that the model is capacity-limited by the bottleneck rather than under-trained.")
rep(r"An ablation with a single $64{\times}64$ skip connection (flag \texttt{--skip}) in our first Colab run raised clean SSIM from 0.88 to 0.97 and salt-and-pepper PSNR from 26.7 to 33.2\,dB, but it lets high-frequency content bypass the bottleneck and weakens the autoencoder constraint, so the skip-free model is the submitted Task~1 system.",
    r"""\textbf{Limited-skip ablation.} We investigated one limited skip connection (encoder features at $64{\times}64$ concatenated into the decoder; flag \texttt{--skip}), trained with the same configuration for 20 epochs. Table~\ref{tab:skip} shows that it raises overall test PSNR from 22.19 to 27.95\,dB and SSIM from 0.792 to 0.873. The gain is largest where fine detail matters (clean $+10.2$\,dB, salt-and-pepper $+7.7$\,dB) and smallest for occlusion ($+2.5$\,dB): the skip carries the high-resolution content of the \emph{visible} pixels but cannot invent the masked content, which still has to come from the bottleneck. Because the skip lets detail bypass the compressed latent and weakens the autoencoder constraint required by the assignment, the skip-free model remains the submitted Task~1 system; the skip variant is exported to ONNX as well (max.\ deviation $8.9\cdot10^{-7}$) and selectable in the application so that the effect can be demonstrated live (Fig.~\ref{fig:skipex}).
\begin{table}[t]\centering\caption{Effect of one limited $64{\times}64$ skip connection (test set).}\label{tab:skip}\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}\toprule
 & \multicolumn{2}{c}{Bottleneck only} & \multicolumn{2}{c}{+ limited skip} \\
Input & PSNR & SSIM & PSNR & SSIM \\ \midrule
Clean & 23.29 & 0.849 & 33.53 & 0.970 \\
S\&P & 23.16 & 0.834 & 30.84 & 0.925 \\
Blur & 22.77 & 0.791 & 28.34 & 0.844 \\
Occlusion & 20.29 & 0.734 & 22.82 & 0.816 \\
Overall & 22.19 & 0.792 & 27.95 & 0.873 \\ \bottomrule
\end{tabular}\end{table}
""" + fig('t1_skip_examples_a.png', 'Limited-skip ablation examples (same layout as Fig.~\\ref{fig:t1ex}): fine fur and background texture are preserved, while large occluded areas remain blurry.', 'fig:skipex'))
rep(r"\begin{figure}[t]\centering\includegraphics[width=0.9\linewidth]{t1_optuna_history.png}\caption{Task 1 Optuna optimisation history.}\label{fig:t1opt}\end{figure}",
    r"\begin{figure}[t]\centering\includegraphics[width=0.49\linewidth]{t1_optuna_history.png}\hfill\includegraphics[width=0.49\linewidth]{t1_optuna_importance.png}\caption{Task 1 Optuna optimisation history (left) and hyper-parameter importance (right).}\label{fig:t1opt}\end{figure}" + "\n" +
    fig('t1_examples_b_final.png', 'Task 1 examples, continued (six further cases; twelve representative examples in total with Fig.~\\ref{fig:t1ex}).', 'fig:t1exb'))

# ---- Task 2: curves, Optuna history, specialists, oracle vs predicted by severity
rep(r"\textbf{Specialists.} The shared study",
    r"""The classifier trained for 25 epochs reaches a best validation macro-F1 of 0.995 at epoch 17 (Fig.~\ref{fig:t2curves}); its final training accuracy (0.999) is slightly above validation (0.993), a small generalisation gap that dropout 0.39 keeps from growing.
""" + fig('t2_classifier_curves.png', 'Task 2 classifier: training/validation cross-entropy (left) and accuracy and macro-F1 (right).', 'fig:t2curves') +
    fig('t2_classifier_optuna_history.png', 'Task 2 classifier Optuna history (objective: validation macro-F1).', 'fig:t2opt', width=r'0.85\linewidth') +
    r"\textbf{Specialists.} The shared study")
rep(r"i.e.\ the specialists prefer a larger pixel-wise L1 share than the universal model.",
    r"""i.e.\ the specialists prefer a larger pixel-wise L1 share than the universal model. Each specialist was then trained independently for 25 epochs on its own corruption only (2.25\,M parameters each). Fig.~\ref{fig:t2spec} shows their curves: validation SSIM reaches 0.855 (salt-and-pepper), 0.850 (blur) and 0.735 (occlusion) and is still rising slightly at epoch 25, so longer training would help, while occlusion is clearly the hardest problem because the content behind the masks must be hallucinated. Table~\ref{tab:t2sev} gives each specialist's performance by severity (oracle routing).
""" + fig('t2_specialists_curves.png', 'Task 2 specialists: training loss and validation SSIM of the three independently trained experts (each validated on its own corruption only).', 'fig:t2spec') +
    r"""\begin{table}[t]\centering\caption{Hard routing by corruption and severity: oracle (true label) vs.\ predicted (classifier) routing on the test set.}\label{tab:t2sev}\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrr}\toprule
Input & Sev. & PSNR$_{in}$ & PSNR$_{orc}$ & PSNR$_{pred}$ & SSIM$_{orc}$ & SSIM$_{pred}$ \\ \midrule
Clean & -- & 50.00 & 50.00 & 49.83 & 1.000 & 0.998 \\
S\&P & low & 20.13 & 26.85 & 26.85 & 0.856 & 0.856 \\
S\&P & med. & 15.87 & 26.75 & 26.75 & 0.852 & 0.852 \\
S\&P & high & 13.14 & 26.52 & 26.52 & 0.842 & 0.842 \\
Blur & low & 32.34 & 26.80 & 26.81 & 0.857 & 0.857 \\
Blur & med. & 26.77 & 26.78 & 26.78 & 0.846 & 0.846 \\
Blur & high & 24.35 & 24.73 & 24.73 & 0.742 & 0.742 \\
Occl. & low & 16.70 & 23.87 & 23.84 & 0.795 & 0.797 \\
Occl. & med. & 13.27 & 22.18 & 22.18 & 0.746 & 0.746 \\
Occl. & high & 10.75 & 20.32 & 20.32 & 0.671 & 0.671 \\ \bottomrule
\end{tabular}\end{table}
""")

# ---- Task 2: when do specialists help or fail
rep(r"\textbf{Two simultaneous corruptions.}",
    r"""\textbf{Comparison of the three systems.} Figs.~\ref{fig:cmpbars} and~\ref{fig:cmpsev} compare all systems on identical test cases. Specialists help most for salt-and-pepper ($+3.5$\,dB over the universal DAE at every severity) and for clean inputs (identity bypass), but they \emph{fail} for mild blur: at low blur every learned restorer (about 27\,dB) is worse than doing nothing (32.3\,dB input PSNR), because the autoencoders cannot reproduce fine detail exactly. A routing rule that bypasses mildly blurred inputs, or the limited-skip variant, would fix this. Occlusion remains the hardest corruption for all systems.
""" + fig('cmp_systems_by_type.png', 'Test PSNR and SSIM by input condition for the universal DAE, hard routing (predicted and oracle) and the soft MoE.', 'fig:cmpbars', star=True) +
    fig('cmp_psnr_by_severity.png', 'Test PSNR versus severity for every corruption; the dashed line is the corrupted input. Hard (predicted) and hard (oracle) overlap almost exactly because the classifier is 99.7\\% accurate.', 'fig:cmpsev', star=True) +
    r"\textbf{Two simultaneous corruptions.}")

# ---- Task 3: warm-up effect, Optuna plot, dominant examples, distribution, failures
rep(r"\textbf{Optuna.} Of 6 trials, 2 were pruned;",
    r"\textbf{Joint training.} Starting from the Task 2 classifier and specialists, validation SSIM rose from 0.862 to 0.864 and routing accuracy from 97.2\% to 98.6\% over the two warm-up epochs (experts frozen); joint fine-tuning with the 28$\times$ smaller learning rate then improved validation SSIM to 0.868 while the mean validation weights stayed close to $\frac14$ (0.27/0.22/0.26/0.25 on the class-balanced validation set; Fig.~\ref{fig:t3curve}). " + "\n" + r"\textbf{Optuna.} Of 6 trials, 2 were pruned;")
rep(r"Fig.~\ref{fig:t3dist} shows the most distributed examples.",
    r"""Fig.~\ref{fig:t3dom} shows examples where one expert dominates, Fig.~\ref{fig:t3dist} the most distributed examples, and Fig.~\ref{fig:t3box} the full weight distribution per true class.
\textbf{Failure cases.} The lowest-SSIM test cases (Fig.~\ref{fig:t3fail}) reveal a gate error: a \emph{clean} cat on a pure black background receives 0.94 weight on the \emph{occlusion} expert, which mistakes the black background for a mask and paints it grey. The same image under high occlusion is the worst occlusion case. The salt-and-pepper and blur failures are saturated green or purple scenes where the experts shift colour towards brown, showing that the experts learned a dataset colour prior.
""" + fig('t3_moe_examples_dominant.png', 'Soft MoE examples where a single branch dominates (weights printed as identity/s\\&p/blur/occlusion).', 'fig:t3dom') +
    fig('t3_moe_weight_distribution.png', 'Distribution of the four routing weights for each true input condition (test set).', 'fig:t3box', star=True) +
    fig('t3_moe_failures.png', 'Soft MoE failure cases (lowest SSIM per input condition) with their routing weights.', 'fig:t3fail') +
    fig('t3_moe_optuna_history.png', 'Task 3 Optuna history.', 'fig:t3opt', width=r'0.85\linewidth'))

# ---- Task 4: style distribution, sample pairs, Optuna history, MLflow logging
rep(r"FS2K's official train/test definitions are used;",
    r"FS2K's official train/test definitions are used (Table~\ref{tab:data}, Fig.~\ref{fig:t4dist});")
rep(r"\textbf{Optuna.} Four 8-epoch trials were run",
    fig('t4_style_distribution.png', 'FS2K photo--sketch pairs per style and split. Training and validation are balanced; the official test split is not.', 'fig:t4dist', width=r'0.75\linewidth') +
    r"\textbf{Optuna.} Four 8-epoch trials were run")
rep(r"Validation SSIM peaked at epoch 22 (0.477) and ended at 0.452.",
    r"Validation SSIM peaked at epoch 22 (0.477) and ended at 0.452. All four losses and the validation metrics are recorded as separate MLflow metrics (Fig.~\ref{fig:mlflowt4}), and the same eight validation photographs are rendered and logged as an image grid after epoch 1 and every 5 epochs, so that the generator's progress can be inspected over time in MLflow.")
rep(r"\begin{figure}[t]\centering\includegraphics[width=0.62\linewidth]{t4_cgan_failures.png}",
    fig('t4_cgan_optuna_history.png', 'Task 4 Optuna history.', 'fig:t4opt', width=r'0.85\linewidth') +
    r"\begin{figure}[t]\centering\includegraphics[width=0.62\linewidth]{t4_cgan_failures.png}")

# ---- Application: Stitch evidence and design-implementation correspondence, skip toggle
rep(r"\IfFileExists{figures/stitch_1.png}{\begin{figure}[t]\centering\includegraphics[width=\linewidth]{stitch_1.png}\caption{Original Google Stitch design of the restoration workspace.}\label{fig:stitch}\end{figure}}{}",
    r"""\textbf{Design in Google Stitch.} The interface was designed in Google Stitch before implementation (Fig.~\ref{fig:stitch}): first a single-column layout, then the desktop restoration workspace and the Face-to-Sketch workspace. The implementation follows this design: the same left sidebar with the four workspaces and an API status pill, a 320\,px column of control cards (input image with samples, runtime corruption with Low/Medium/High buttons and a seed, Run model), four image panels, stat tiles, probability bars and a routing-decision card, the same palette (background \texttt{\#0b1020}, cards \texttt{\#111833}, accent \texttt{\#38bdf8}), the Inter font and 16\,px rounded cards. Tailwind breakpoints make the layout responsive: the sidebar becomes a horizontal tab bar and the panels stack on narrow screens. The Universal workspace additionally offers a \emph{model variant} switch (bottleneck only vs.\ limited skip; Fig.~\ref{fig:appskip}).
\begin{figure*}[t]\centering\includegraphics[width=0.49\linewidth]{stitch_2.png}\hfill\includegraphics[width=0.49\linewidth]{stitch_3.png}\caption{Original Google Stitch designs: desktop restoration workspace (left) and Face-to-Sketch workspace (right).}\label{fig:stitch}\end{figure*}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{stitch_1.png}\caption{The Stitch canvas with the design prompts (evidence of the original design).}\label{fig:stitchui}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{app_universal_skip.png}\caption{Universal Restoration workspace with the limited-skip variant selected.}\label{fig:appskip}\end{figure}""")
rep(r"\IfFileExists{figures/app_hard.png}{\begin{figure}[t]\centering\includegraphics[width=\linewidth]{app_hard.png}\caption{Implemented Hard-Routed Restoration workspace (React + Tailwind).}\label{fig:app}\end{figure}}{}", "")
rep(r"\section{Limitations}",
    r"""\section{Experiment Tracking and Reproducibility}
Every training, Optuna-trial and evaluation run is logged to MLflow (61 runs in four experiments): hyper-parameters, per-epoch losses and metrics, test metrics, sample-image grids, result files and checkpoints (Fig.~\ref{fig:mlflow}). The tracking database is committed in \texttt{experiments/mlflow/}, the Optuna studies in \texttt{experiments/optuna/}, the selected hyper-parameters in \texttt{configs/final\_configs.json}, and all result tables and figures in \texttt{experiments/results/}. All randomness is seeded (split seed 42, Optuna sampler seed 42, training seed 42, per-image seeds in the manifests), and \texttt{scripts/run\_all.sh} reproduces the whole training pipeline stage by stage, skipping stages whose outputs already exist (which made the pipeline robust to two Colab disconnections).
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{mlflow_task4_metrics.png}\caption{MLflow: separately logged discriminator real/fake, generator adversarial and generator L1 losses and validation L1 of the final Task 4 run.}\label{fig:mlflowt4}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{mlflow_task4_runs.png}\caption{MLflow runs of the Task 4 experiment (Optuna parent run with nested trials, final training and test evaluation).}\label{fig:mlflow}\end{figure}

\section{Difficulties and How They Were Resolved}
(1) \emph{Bottleneck vs.\ fidelity}: the skip-free autoencoder plateaued at about 23\,dB; we investigated a limited skip (Table~\ref{tab:skip}) and used specialists with an identity bypass instead of a larger universal model. (2) \emph{Incomparable Optuna objectives}: scoring trials with their own $\alpha$-weighted loss would bias the search, so all restoration studies use a fixed L1 + (1$-$SSIM) objective. (3) \emph{Numerical issues}: SSIM was unstable in fp16 and is computed in fp32; identical images gave infinite PSNR, so PSNR is capped at 50\,dB. (4) \emph{Routing collapse} in MoE training was guarded against by a batch-level balance loss and by pruning collapsed trials. (5) \emph{Infrastructure}: the MLflow file store is in maintenance mode in recent releases (switched to SQLite), PyTorch~2.6 changed checkpoint loading, two Colab runtimes disconnected and the free GPU's memory required sequential jobs; a resumable, Drive-backed runner solved all of these.

\section{Limitations}""")

# ---- conclusions + future work per task
rep(r"\section{Conclusion}" + "\n",
    r"""\section{Conclusion}
\textbf{Task 1.} A single universal autoencoder removes heavy salt-and-pepper noise and occlusion but caps fidelity at about 23\,dB and degrades clean and mildly blurred images; a limited skip lifts it to 27.95\,dB. \emph{Future work:} larger latents or a perceptual loss. \textbf{Task 2.} A 99.7\%-accurate classifier with specialists and an identity bypass adds 5.3\,dB; nearly all residual errors are low-severity occlusions. \emph{Future work:} a bypass rule for mild blur and longer specialist training. \textbf{Task 3.} The jointly trained soft MoE achieves the best SSIM (0.832) with interpretable, mostly sharp routing, blending only where blending helps. \emph{Future work:} training on corruption mixtures with soft targets, and calibrating the gate (the black-background failure). \textbf{Task 4.} The style-conditioned cGAN produces three clearly different sketch styles from the same photograph; the hardest style shows template artefacts. \emph{Future work:} more data, perceptual or feature-matching losses, and spectral normalisation. """ + "\n")

open(p, 'w', encoding='utf-8').write(s)
print('report v2 written', len(s.splitlines()), 'lines')
