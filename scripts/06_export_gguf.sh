#!/usr/bin/env bash
# Step 6: convert a HF checkpoint dir to GGUF and produce Q4_K_M.
# Usage: bash scripts/06_export_gguf.sh models/qwen3-4b-awq-w4-g128
# Requires llama.cpp cloned at ../llama.cpp (or set LLAMA_CPP).
set -euo pipefail

SRC="${1:?usage: 06_export_gguf.sh <hf-model-dir>}"
LLAMA_CPP="${LLAMA_CPP:-../llama.cpp}"
NAME="$(basename "$SRC")"
OUT_DIR="models/gguf"
mkdir -p "$OUT_DIR"

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" \
  --outfile "$OUT_DIR/$NAME-f16.gguf" --outtype f16

"$LLAMA_CPP/build/bin/llama-quantize" \
  "$OUT_DIR/$NAME-f16.gguf" "$OUT_DIR/$NAME-Q4_K_M.gguf" Q4_K_M

ls -lh "$OUT_DIR"
