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
    parser.add_argument("--tag", default=None,
        help="suffix appended to config_name, e.g. --tag gold -> baseline-bf16-gold")
    args = parser.parse_args()

    cfg = M.load_config()
    model_id = args.model or cfg["model_id"]          # <-- override actually used
    hw = cfg["hardware"]
    seq_len = 512 if args.dryrun else None      # None = use config value
    max_windows = 8 if args.dryrun else None

    use_cuda = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"

    tok = AutoTokenizer.from_pretrained(model_id)

    if use_cuda:
        M.reset_vram_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
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
    model.config.use_cache = True
    tps = M.throughput(model, tok, cfg)
    vram = M.peak_vram_gb() if use_cuda else 0.0

    if use_cuda:
        devmap = getattr(model, "hf_device_map", None)
        if devmap and any(v in ("cpu", "disk") for v in devmap.values()):
            placement = "device_map=auto with partial CPU offload"
        else:
            placement = "device_map=auto, fully on GPU"
        note = f"{placement}; max_gpu_mem={hw['max_gpu_mem']}"
    else:
        note = f"CPU dry-run on {model_id}" if args.dryrun else f"CPU run on {model_id}"

    del model
    if use_cuda:
        torch.cuda.empty_cache()

    tasks = {} if args.dryrun else M.run_lm_eval(model_id, cfg)

    config_name = ("baseline-cpu-dryrun" if args.dryrun
            else ("baseline-bf16" if use_cuda else "baseline-cpu"))
    if args.tag:
        config_name += f"-{args.tag}"

    M.log_result(
        cfg,
        config_name=config_name,
        method="none",
        bits=16,
        disk_gb="",  # HF cache; record quantized dirs from step 2 onward
        peak_vram_gb=round(vram, 2),
        ppl_wikitext2=round(ppl, 3),
        mmlu=tasks.get("mmlu", ""),
        gsm8k=tasks.get("gsm8k", ""),
        tok_per_s=round(tps, 1),
        notes=note,
    )


if __name__ == "__main__":
    main()