#!/usr/bin/env bash
# Task 4 end to end (Colab). Expects FS2K unzipped somewhere under $GENAI_DATA.
#   cd $GENAI_DATA && gdown --fuzzy 'https://drive.google.com/file/d/1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp/view' -O FS2K.zip && unzip -q FS2K.zip
set -u
cd "$(dirname "$0")/.."
backup() { [ -n "${BACKUP:-}" ] && rsync -a "$GENAI_OUT/" "$BACKUP/outputs/" && rsync -a manifests/ "$BACKUP/manifests/" \
           && mkdir -p "$BACKUP/models_onnx" && rsync -a models_onnx/ "$BACKUP/models_onnx/" && echo "[backup] done"; }
run() { echo "=== $(date +%T) $*"; python "$@" || echo "!!! FAILED: $*"; }

run scripts/task4_cgan.py prepare
run scripts/task4_cgan.py optuna --trials ${GAN_TRIALS:-10} --epochs ${GAN_TRIAL_EPOCHS:-15} --workers 1
backup
run scripts/task4_cgan.py train --epochs ${GAN_EPOCHS:-100} --workers 1
run scripts/task4_cgan.py eval --workers 1
run scripts/task4_cgan.py export
backup
echo "=== $(date +%T) TASK 4 DONE"
