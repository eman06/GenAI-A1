#!/usr/bin/env bash
# Tasks 2 and 3 end to end (Colab). Requires GENAI_DATA / GENAI_OUT; BACKUP (optional) = Drive folder.
set -u
cd "$(dirname "$0")/.."
backup() { [ -n "${BACKUP:-}" ] && rsync -a "$GENAI_OUT/" "$BACKUP/outputs/" && rsync -a manifests/ "$BACKUP/manifests/" \
           && mkdir -p "$BACKUP/models_onnx" && rsync -a models_onnx/ "$BACKUP/models_onnx/" && echo "[backup] done"; }
run() { echo "=== $(date +%T) $*"; python "$@" || echo "!!! FAILED: $*"; }

run scripts/task2_classifier.py optuna --trials ${CLF_TRIALS:-12} --epochs 5
run scripts/task2_classifier.py train --epochs ${CLF_EPOCHS:-25}
run scripts/task2_classifier.py eval
backup
run scripts/task2_specialists.py optuna --trials ${SPEC_TRIALS:-10} --epochs 4
run scripts/task2_specialists.py train --epochs ${SPEC_EPOCHS:-30}
run scripts/task2_eval.py
run scripts/task2_export.py
backup
run scripts/task3_moe.py optuna --trials ${MOE_TRIALS:-8}
run scripts/task3_moe.py train --warmup 2 --epochs ${MOE_EPOCHS:-12}
run scripts/task3_moe.py eval
run scripts/task3_moe.py export
backup
echo "=== $(date +%T) TASKS 2-3 DONE"
