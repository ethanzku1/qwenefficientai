#!/usr/bin/env bash
# Step 7: deployment benchmarks — full-GPU vs CPU-only, plus KV-cache quant.
# Usage: bash scripts/07_bench_llamacpp.sh models/gguf/<name>-Q4_K_M.gguf
set -euo pipefail

GGUF="${1:?usage: 07_bench_llamacpp.sh <path-to-gguf>}"
LLAMA_CPP="${LLAMA_CPP:-../llama.cpp}"
BENCH="$LLAMA_CPP/build/bin/llama-bench"

echo "== full GPU offload =="
"$BENCH" -m "$GGUF" -ngl 99 -p 512 -n 128

echo "== CPU only =="
"$BENCH" -m "$GGUF" -ngl 0 -p 512 -n 128

echo "== long-context memory with quantized KV cache =="
"$LLAMA_CPP/build/bin/llama-cli" -m "$GGUF" -ngl 99 -c 16384 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -p "Summarize the tradeoffs of 4-bit quantization." -n 64 --no-display-prompt

echo
echo "Copy the tok/s numbers into results/results.csv (or extend make_report to parse them)."
