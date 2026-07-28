"""Step 7: benchmark GGUF Q4_K_M via llama.cpp on the laptop (CPU, WSL2).

Logs one row per model to results/results.csv via the shared harness schema.

Comparability contract:
  - tok_per_s: from llama-bench text generation (tg) -- the headline number.
  - disk_gb:   GGUF file size.
  - ppl_wikitext2: LEFT BLANK. That column means "our harness, seq 2048,
    stride 512, HF implementation". llama-perplexity uses non-overlapping
    chunks on a different implementation; its number goes in `notes` when
    --ppl is passed, never in the column.
  - peak_vram_gb: 0.0 (CPU).

n_ctx is pinned EXPLICITLY everywhere. Qwen3 Instruct-2507 has a 262k
native context; llama.cpp defaults to the model's full trained context and
will allocate ~9GB of KV/compute buffers for it, which suffocates a 12GB
WSL2 VM (empirically: 11.1GB RSS, swap thrash). Context length is a
deployment parameter on constrained hardware -- report finding.

Usage:
  python scripts/07_llamacpp_bench.py                # bench both models
  python scripts/07_llamacpp_bench.py --only base    # or --only instruct
  python scripts/07_llamacpp_bench.py --ppl          # add llama-perplexity
                                                     # (slow on CPU: ~1-2h/model)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from effml import measure as M

N_CTX = 4096          # agent workloads fit; keeps RAM sane on 12GB WSL2
BENCH_PROMPT = 512    # llama-bench pp test size
BENCH_GEN = 128       # llama-bench tg test size

MODELS = {
    "base": ("Qwen/Qwen3-4B", "qwen3-4b-Q4_K_M.gguf"),
    "instruct": ("Qwen/Qwen3-4B-Instruct-2507",
                 "qwen3-4b-instruct-2507-Q4_K_M.gguf"),
}


def llama_bench(bin_dir: Path, gguf: Path):
    """Returns (prompt_tps, gen_tps) from llama-bench JSON output."""
    cmd = [bin_dir / "llama-bench", "-m", gguf,
           "-p", str(BENCH_PROMPT), "-n", str(BENCH_GEN), "-o", "json"]
    print(f"[07] {' '.join(str(c) for c in cmd)}", flush=True)
    out = subprocess.run([str(c) for c in cmd], check=True,
                         capture_output=True, text=True).stdout
    # llama-bench may print non-JSON warmup lines; take the JSON array.
    payload = out[out.index("["):out.rindex("]") + 1]
    results = json.loads(payload)
    pp = tg = None
    for r in results:
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            pp = r["avg_ts"]
        if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            tg = r["avg_ts"]
    if tg is None:
        raise RuntimeError(f"no tg result in llama-bench output:\n{out}")
    return pp, tg


def wikitext_test_file(cache_dir: Path) -> Path:
    """Materialize wikitext-2-raw-v1 test split as raw text for llama-perplexity."""
    f = cache_dir / "wiki.test.raw"
    if f.exists():
        return f
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(ds["text"]))
    return f


def llama_ppl(bin_dir: Path, gguf: Path, textfile: Path) -> float:
    cmd = [bin_dir / "llama-perplexity", "-m", gguf,
           "-f", textfile, "-c", str(2048)]
    print(f"[07] {' '.join(str(c) for c in cmd)}  (slow on CPU)", flush=True)
    proc = subprocess.run([str(c) for c in cmd], check=True,
                          capture_output=True, text=True)
    blob = proc.stdout + proc.stderr
    m = re.search(r"Final estimate: PPL = ([\d.]+)", blob)
    if not m:
        raise RuntimeError(f"could not parse PPL from:\n{blob[-2000:]}")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(MODELS))
    ap.add_argument("--gguf-dir", default=None,
                    help="default: <repo>/models/gguf")
    ap.add_argument("--llama-cpp", default=str(Path.home() / "llama.cpp"))
    ap.add_argument("--ppl", action="store_true",
                    help="also run llama-perplexity (goes in notes, not the "
                         "ppl column -- different windowing, not comparable)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    gguf_dir = Path(args.gguf_dir) if args.gguf_dir else repo / "models" / "gguf"
    bin_dir = Path(args.llama_cpp) / "build" / "bin"

    cfg = M.load_config()

    targets = [args.only] if args.only else sorted(MODELS)
    for key in targets:
        model_id, fname = MODELS[key]
        gguf = gguf_dir / fname
        if not gguf.exists():
            sys.exit(f"[07] missing {gguf} -- scp/download it first (06)")

        print(f"=== llama.cpp bench {key}: {gguf.name} ===")
        pp_tps, tg_tps = llama_bench(bin_dir, gguf)
        print(f"[07] prompt {pp_tps:.1f} t/s, gen {tg_tps:.1f} t/s")

        notes = (f"llama.cpp Q4_K_M CPU, WSL2, n_ctx={N_CTX}, "
                 f"pp={pp_tps:.1f} t/s @ {BENCH_PROMPT}")
        if args.ppl:
            ppl = llama_ppl(bin_dir, gguf,
                            wikitext_test_file(repo / "models" / "gguf"))
            notes += (f", llama-perplexity(c=2048, non-overlapping)={ppl:.3f} "
                      f"-- NOT comparable to ppl_wikitext2 column")

        # model column autofills from cfg; override in-memory for instruct.
        cfg["model_id"] = model_id
        M.log_result(
            cfg,
            config_name=f"gguf-q4km-{key}",
            method="gguf-q4km",
            bits=4,
            disk_gb=round(gguf.stat().st_size / 1e9, 2),
            peak_vram_gb=0.0,
            ppl_wikitext2="",
            mmlu="",
            gsm8k="",
            tok_per_s=round(tg_tps, 1),
            notes=notes,
        )
        print(f"[07] row logged: gguf-q4km-{key}")


if __name__ == "__main__":
    main()