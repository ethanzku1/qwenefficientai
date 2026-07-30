# EfficientML Lab — Compressing Qwen3-4B for Laptop Deployment

TinyML-style lab: apply post-training quantization (RTN / AWQ / GPTQ) to Qwen3-4B, then deploy as GGUF via llama.cpp.
Every experiment appends to a single results table for the final tradeoff analysis.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 0 | `scripts/00_check_env.py` | environment sanity check |
| 1 | `scripts/01_baseline.py` | BF16 baseline: PPL, tasks, memory, tok/s |
| 2 | `scripts/02_quantize_rtn.py` | naive round-to-nearest INT4 (strawman) |
| 3 | `scripts/03_quantize_awq.py` | AWQ W4/W3, group-size ablations |
| 4 | `scripts/04_quantize_gptq.py` | GPTQ W4 comparison |
| 6 | `scripts/06_export_gguf.sh` | GGUF conversion + Q4_K_M |
| 7 | `scripts/07_bench_llamacpp.sh` | GPU vs CPU deployment benchmarks |
| — | `scripts/make_report.py` | renders `results/results.csv` → markdown table |

All Python steps share the measurement harness in `src/effml/` and append one
row per configuration to `results/results.csv` (tracked in git, so numbers sync
across machines automatically).

## What syncs via git, what doesn't

**Tracked:** all code, configs, `results/results.csv`, the report, notebooks.

**Not tracked (see `.gitignore`):** model weights, quantized checkpoints,
GGUF files, the HF cache. These are multi-GB and machine-local. Each machine
re-downloads the base model on first run (cached by HF automatically) and
regenerates or re-downloads quantized artifacts as needed. If you want
quantized checkpoints to follow you, push them to a private HF Hub repo
(`scripts/03_quantize_awq.py --push-to-hub <repo>`), not to git.

## Multi-machine workflow

1. `git pull` before starting work on either machine.
2. Run experiments; results append to `results/results.csv`.
3. `git add -A && git commit -m "awq w3 g64 run" && git push`.
4. On the other machine: `git pull` — code and numbers are current; heavy
   artifacts regenerate locally when needed.

Machines with different hardware (e.g. 4070 laptop vs iGPU laptop) record
their hostname and GPU in each results row, so deployment numbers from
different devices coexist in one table — useful for the writeup.

## VS Code

Open the folder; recommended extensions will be suggested automatically
(Python, Jupyter, Remote-WSL). Preconfigured tasks (Terminal → Run Task):
baseline, AWQ, report. Debug configs in `.vscode/launch.json` let you
breakpoint into any pipeline step.

On Windows, open the repo **inside WSL** (`code .` from the WSL shell) so
CUDA-enabled PyTorch and llama.cpp builds behave.
