#!/usr/bin/env bash
# Builds dist/report_overleaf.zip (upload to Overleaf) and dist/models_onnx.zip (attach to a GitHub release).
set -e
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/report_overleaf.zip dist/models_onnx.zip
(cd report && python -c "
import zipfile, os
z = zipfile.ZipFile('../dist/report_overleaf.zip', 'w', zipfile.ZIP_DEFLATED)
for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith(('.tex', '.png', '.jpg', '.pdf', '.bib')):
            z.write(os.path.join(root, f))
z.close()")
python -c "
import zipfile, glob
z = zipfile.ZipFile('dist/models_onnx.zip', 'w', zipfile.ZIP_DEFLATED)
for f in sorted(glob.glob('models_onnx/*.onnx')): z.write(f)
z.close()"
ls -la dist/*.zip
