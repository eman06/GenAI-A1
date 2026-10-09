"""Replace the Task 4 'status' text with the real results (numbers from drive_results/t4)."""
p = 'report/main.tex'
s = open(p, encoding='utf-8').read()

old_abs = "The face-to-sketch cGAN pipeline (data split, conditioning, Optuna search) is complete; its final training run was still in progress at submission time."
new_abs = (r"The style-conditioned cGAN produces visibly different sketch styles from the same photograph and reaches a test SSIM of 0.58 / 0.47 / 0.35 for styles 3 / 1 / 2 "
           r"(Style~2, dense pencil hatching, is the hardest).")
assert old_abs in s
s = s.replace(old_abs, new_abs)

i = s.index(r"\textbf{Status.}")
j = s.index(r"\section{Application Design and Deployment}")
t4 = r"""\textbf{Optuna.} Four 8-epoch trials were run (3 complete, 1 pruned; the budget was reduced after two Colab disconnections). The best configuration used $\mathrm{lr}_G=1.2\cdot10^{-4}$, $\mathrm{lr}_D=4.3\cdot10^{-4}$, batch 4, 64 base channels, dropout 0.30, a 32-dimensional style embedding and $\lambda_{L1}=121$ (validation SSIM 0.477). Optuna kept $\lambda_{L1}$ close to the pix2pix value of 100 but preferred a discriminator learning rate $3.6\times$ larger than the generator's, which compensated for the weak early discriminator signal of the small-batch regime. The configuration was retrained for 40 epochs.
\textbf{Training dynamics.} Fig.~\ref{fig:t4curve} shows the separately logged losses. The discriminator's real and fake losses fall from 0.76/0.75 to 0.16/0.15 while the generator's adversarial loss rises from 0.78 to 2.64: towards the end the discriminator increasingly wins, but the L1 term keeps the generator anchored (train L1 0.29$\rightarrow$0.21, validation L1 0.113$\rightarrow$0.104). Validation SSIM peaked at epoch 22 (0.477) and ended at 0.452. Following pix2pix practice we deploy the final generator, because later epochs produce crisper strokes even when pixel SSIM is slightly lower.
\textbf{Results.} Table~\ref{tab:t4} reports test metrics per style. FS2K's official test split is strongly unbalanced (619 / 381 / 46 images for styles 1/2/3), whereas our training split is nearly balanced (303 / 298 / 298), so we report every style separately. Style~3 (soft shading) is reproduced best (SSIM 0.582, PSNR 18.3\,dB) and Style~2 (dense pencil hatching) worst (0.348, 11.8\,dB): hatching patterns are stochastic, and a pixel-wise metric penalises any plausible but differently placed stroke. The style-control grid (Fig.~\ref{fig:t4style}) shows that the learned embedding genuinely controls the output: the same photograph is rendered with light contours (Style~1), heavy dark hatching (Style~2) or soft grey shading (Style~3), while identity, glasses, hats and hairstyles are preserved. This confirms that the condition is used by the network and is not just an interface label.
\textbf{Failure cases.} The six lowest-SSIM test cases (Fig.~\ref{fig:t4fail}) are all Style~1/2 portraits of people with long, light-coloured hair. They share a repeated inverted-V ``nose'' stroke that ignores the actual nose, a template-like artefact typical of mild mode collapse in conditional GANs trained on small datasets (fewer than 900 training pairs). Long hair is also drawn as dense vertical strokes rather than the artist's flowing contours. More data, a perceptual (VGG) loss or feature matching, and spectral normalisation in $D$ would be the next steps.
\begin{table}[t]\centering\caption{Task 4 test results per sketch style (FS2K official test split).}\label{tab:t4}
\begin{tabular}{lrrrr}\toprule
Style & $n$ & L1 & SSIM & PSNR (dB)\\ \midrule
Style 1 & 619 & 0.087 & 0.472 & 16.66 \\
Style 2 & 381 & 0.165 & 0.348 & 11.80 \\
Style 3 & 46 & 0.068 & 0.582 & 18.31 \\
Overall & 1046 & 0.114 & 0.432 & 14.96 \\
\bottomrule\end{tabular}\end{table}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{t4_cgan_style_control.png}\caption{Style control: each test photograph rendered with all three style conditions.}\label{fig:t4style}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{t4_cgan_curves.png}\caption{Task 4 training: discriminator real/fake loss, generator adversarial loss, train/validation L1 and validation SSIM.}\label{fig:t4curve}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=0.8\linewidth]{t4_cgan_failures.png}\caption{Task 4 failure cases (lowest test SSIM): photo, generated sketch, ground truth.}\label{fig:t4fail}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=0.8\linewidth]{t4_cgan_test_examples.png}\caption{Task 4 test examples, four per style: photo, generated sketch (true style), ground truth.}\label{fig:t4ex}\end{figure}

"""
s = s[:i] + t4 + s[j:]

# app section: add the sketch screenshot next to the others
old_fig = r"\IfFileExists{figures/app_system.png}"
add = r"""\begin{figure}[t]\centering\includegraphics[width=\linewidth]{app_sketch.png}\caption{Face-to-Sketch Generator workspace: generated sketch and the three-style comparison for a test photograph.}\label{fig:appsketch}\end{figure}
"""
if 'app_sketch.png' not in s:
    s = s.replace(old_fig, add + old_fig, 1)

s = s.replace("(2) Optuna budgets were small (4--12 trials, 4--15 epochs per trial)", "(2) Optuna budgets were small (4--12 trials, 4--8 epochs per trial)")
s = s.replace("The main open problem is calibrated routing for unseen corruption mixtures.",
              "The style-conditioned cGAN shows clear, controllable style transfer but suffers from template artefacts on the hardest style. The main open problems are calibrated routing for unseen corruption mixtures and data-efficient sketch synthesis.")
s = s.replace("a PyTorch~2.6 checkpoint-loading change (\\texttt{weights\\_only}) that broke the Task~4 evaluation, ", "")
s = s.replace("CRLF line endings that broke the shell scripts on Linux, and memory limits",
              "CRLF line endings that broke the shell scripts on Linux, a PyTorch~2.6 checkpoint-loading change (\\texttt{weights\\_only}) that broke the Task~4 evaluation, and memory limits")
open(p, 'w', encoding='utf-8').write(s)
print('ok')
