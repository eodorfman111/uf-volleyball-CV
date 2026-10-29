#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
REF=https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/master/models
for m in VballNetV4c_seq9_grayscale_20260908_213829; do
  for ext in onnx json; do
    [ -s "models/$m.$ext" ] || curl -fL -o "models/$m.$ext" "$REF/$m.$ext"
  done
done
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("grdesignbuild/volleyball-person-ball-detector", allow_patterns=["*.json", "*.safetensors"])
PY
ls -la models
