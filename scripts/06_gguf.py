"""Step 6: convert HF checkpoints to GGUF and quantize to Q4_K_M.

Runs on the POD (RAM headroom, weights already in /workspace/hf_cache, fast
local disk). Produces the ~2.5GB Q4_K_M files that 07 benchmarks on the
laptop. Files move by scp, never git (100MB limit).

GGUF Q4_K_M is INDEPENDENT of the AWQ/GPTQ track: convert_hf_to_gguf.py
reads the original BF16 HF weights and llama.cpp re-quantizes from scratch
with its own k-quant scheme. Nothing here consumes 03/04 output.

Converts BOTH checkpoints:
  - Qwen/Qwen3-4B                (so 07's row is comparable to results.csv)
  - Qwen/Qwen3-4B-Instruct-2507  (the agent showpiece backend)

Prereqs (one-time, see notes):
  /workspace/llama.cpp cloned, llama-quantize built (CPU build)
  pip install gguf   # deps are numpy/pyyaml/tqdm only -- torch-safe

Usage:
  /usr/bin/python -u scripts/06_gguf.py                 # both models
  /usr/bin/python -u scripts/06_gguf.py --only base     # or --only instruct
  /usr/bin/python -u scripts/06_gguf.py --keep-bf16     # keep intermediates
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODELS = {
    "base": "Qwen/Qwen3-4B",
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",
}
QUANT_TYPE = "Q4_K_M"


def run(cmd, log_prefix):
    print(f"[06] {log_prefix}: {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def resolve_hf_dir(model_id: str) -> Path:
    """Local snapshot path; downloads only if not already in HF_HOME cache."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def convert_one(model_id: str, out_root: Path, llama_cpp: Path,
                keep_bf16: bool) -> Path:
    short = model_id.split("/")[-1].lower()
    out_dir = out_root / short
    done = out_dir / "DONE"
    q_file = out_dir / f"{short}-{QUANT_TYPE}.gguf"

    if done.exists():
        print(f"[skip] {q_file} already complete")
        return q_file
    if out_dir.exists():
        print(f"[warn] {out_dir} exists but no DONE marker -- partial, redoing")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    hf_dir = resolve_hf_dir(model_id)
    print(f"[06] {model_id} resolved to {hf_dir}")

    bf16_file = out_dir / f"{short}-bf16.gguf"
    convert_py = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    for p in (convert_py, quantize_bin):
        if not p.exists():
            sys.exit(f"[06] missing {p} -- clone/build llama.cpp first")

    # HF BF16 -> GGUF BF16 (lossless container change)
    run([sys.executable, convert_py, hf_dir,
         "--outfile", bf16_file, "--outtype", "bf16"],
        f"convert {short}")

    # GGUF BF16 -> Q4_K_M (llama.cpp's own k-quant, from scratch)
    run([quantize_bin, bf16_file, q_file, QUANT_TYPE],
        f"quantize {short}")

    if not keep_bf16:
        bf16_file.unlink()
        print(f"[06] removed intermediate {bf16_file.name}")

    done.touch()
    gb = q_file.stat().st_size / 1e9
    print(f"[06] DONE {q_file}  ({gb:.2f} GB)")
    return q_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(MODELS),
                    help="convert a single checkpoint")
    ap.add_argument("--out-root", default="/workspace/gguf",
                    help="output dir (local ephemeral disk, NOT /mnt, NOT the repo)")
    ap.add_argument("--llama-cpp", default="/workspace/llama.cpp")
    ap.add_argument("--keep-bf16", action="store_true",
                    help="keep the ~8GB BF16 intermediate GGUFs")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    llama_cpp = Path(args.llama_cpp)

    targets = {args.only: MODELS[args.only]} if args.only else MODELS
    produced = []
    for key, model_id in targets.items():
        print(f"=== GGUF {key}: {model_id} ===")
        produced.append(convert_one(model_id, out_root, llama_cpp,
                                    args.keep_bf16))

    print("\n[06] all done. scp these to the laptop (never git):")
    for p in produced:
        print(f"  {p}")


if __name__ == "__main__":
    main()