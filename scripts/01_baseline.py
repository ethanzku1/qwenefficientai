"""Step 1: BF16 baseline. Every later configuration is compared to this row."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from effml import measure as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="override config model")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--dryrun", action="store_true",
        help="small/fast settings for CPU plumbing checks")
    args = parser.parse_args()

    cfg = M.load_config()
    model_id = args.model or cfg["model_id"]          # <-- override actually used
    hw = cfg["hardware"]
    seq_len = 512 if args.dryrun else None      # None = use config value
    max_windows = 8 if args.dryrun else None
    dtype = torch.bfloat16

    use_cuda = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"

    tok = AutoTokenizer.from_pretrained(model_id)

    if use_cuda:
        M.reset_vram_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",  # 4B in BF16 ~8GB: partially offloads on an 8GB card
            max_memory={0: hw["max_gpu_mem"], "cpu": hw["max_cpu_mem"]},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if args.dryrun else torch.float32,
        )
    model.eval()
    model.config.use_cache = False

    ppl = M.perplexity(model, tok, cfg, seq_len=seq_len, max_windows=max_windows)
    tps = M.throughput(model, tok, cfg)
    vram = M.peak_vram_gb() if use_cuda else 0.0

    del model
    if use_cuda:
        torch.cuda.empty_cache()

    tasks = {} if args.dryrun else M.run_lm_eval(model_id, cfg)

    M.log_result(
        cfg,
        config_name="baseline-cpu-dryrun" if args.dryrun else ("baseline-bf16" if use_cuda else "baseline-cpu"),
        method="none",
        bits=16,
        disk_gb="",  # HF cache; record quantized dirs from step 2 onward
        peak_vram_gb=round(vram, 2),
        ppl_wikitext2=round(ppl, 3),
        mmlu=tasks.get("mmlu", ""),
        gsm8k=tasks.get("gsm8k", ""),
        tok_per_s=round(tps, 1),
        notes="device_map=auto (partial CPU offload on 8GB card)" if use_cuda
              else f"CPU dry-run on {model_id}",
    )


if __name__ == "__main__":
    main()