#!/usr/bin/env bash
# Resumable sequential runner for everything after Task 1 (Colab).
# Each stage is skipped if its output already exists, so the script can simply be re-run after a disconnect.
# Logs go to $GENAI_OUT/logs/*.log (keeps the notebook output small).
set -u
cd "$(dirname "$0")/.."
O="$GENAI_OUT"; R="$O/results"; C="$O/checkpoints"; M=models_onnx
mkdir -p "$O/logs"
backup() { [ -n "${BACKUP:-}" ] && rsync -a "$O/" "$BACKUP/outputs/" && rsync -a manifests/ "$BACKUP/manifests/" \
           && mkdir -p "$BACKUP/models_onnx" && rsync -a $M/ "$BACKUP/models_onnx/" \
           && cp -n "$GENAI_DATA"/*.npz "$BACKUP/" 2>/dev/null; echo "[$(date +%T)] backup done"; }
stage() {  # stage <name> <done-marker-file> <python args...>
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then echo "[$(date +%T)] skip $name (done)"; return; fi
  echo "[$(date +%T)] START $name"
  if python "$@" > "$O/logs/$name.log" 2>&1; then echo "[$(date +%T)] OK    $name"; else echo "[$(date +%T)] FAIL  $name (see logs/$name.log)"; tail -5 "$O/logs/$name.log"; fi
  backup > /dev/null
}

stage t2_clf_optuna  "$R/task2/classifier_best_params.json"  scripts/task2_classifier.py optuna --trials 12 --epochs 5
stage t2_clf_train   "$C/task2_classifier.pt"                scripts/task2_classifier.py train --epochs 25
stage t2_clf_eval    "$R/task2/classifier_test_report.json"  scripts/task2_classifier.py eval
stage t2_spec_optuna "$R/task2/specialists_best_params.json" scripts/task2_specialists.py optuna --trials ${SPEC_TRIALS:-8} --epochs 4
stage t2_spec_train  "$C/task2_specialist_occlusion.pt"      scripts/task2_specialists.py train --epochs ${SPEC_EPOCHS:-25}
stage t2_eval        "$R/task2/hard_oracle_vs_predicted.csv" scripts/task2_eval.py
stage t2_export      "$M/task2_specialist_occlusion.onnx"    scripts/task2_export.py
stage t3_optuna      "$R/task3/moe_best_params.json"         scripts/task3_moe.py optuna --trials ${MOE_TRIALS:-6}
stage t3_train       "$C/task3_moe.pt"                       scripts/task3_moe.py train --warmup 2 --epochs ${MOE_EPOCHS:-10}
stage t3_eval        "$R/task3/moe_expert_usage.json"        scripts/task3_moe.py eval
stage t3_export      "$M/task3_soft_moe.onnx"                scripts/task3_moe.py export
if [ ! -f "$GENAI_DATA/fs2k/FS2K.zip" ]; then
  mkdir -p "$GENAI_DATA/fs2k" && (cd "$GENAI_DATA/fs2k" && gdown -q --fuzzy 'https://drive.google.com/file/d/1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp/view' -O FS2K.zip && unzip -q -o FS2K.zip)
fi
stage t4_prepare     "$GENAI_DATA/fs2k_128.npz"              scripts/task4_cgan.py prepare
stage t4_optuna      "$R/task4/cgan_best_params.json"        scripts/task4_cgan.py optuna --trials ${GAN_TRIALS:-6} --epochs ${GAN_TRIAL_EPOCHS:-12}
stage t4_train       "$C/task4_generator.pt"                 scripts/task4_cgan.py train --epochs ${GAN_EPOCHS:-60}
stage t4_eval        "$R/task4/cgan_test_by_style.csv"       scripts/task4_cgan.py eval
stage t4_export      "$M/task4_generator.onnx"               scripts/task4_cgan.py export
echo "[$(date +%T)] ALL DONE"
