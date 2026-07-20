"""Step 3: AWQ quantization with ablations over bit-width and group size.

Each variant is quantized, saved under models/, fully re-measured with the
shared harness, and logged as its own row.

Usage:
    python scripts/03_quantize_awq.py                 # all variants in lab.yaml
    python scripts/03_quantize_awq.py --only w4-g128  # a single variant
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from effml import measure as M


def quantize_variant(cfg, w_bit: int, group: int) -> Path:
    from awq import AutoAWQForCausalLM

    name = f"awq-w{w_bit}-g{group}"
    out_dir = Path(cfg["paths"]["models_dir"]) / f"qwen3-4b-{name}"
    if out_dir.exists():
        print(f"[skip] {out_dir} already exists")
        return out_dir

    model = AutoAWQForCausalLM.from_pretrained(cfg["model_id"], safetensors=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model.quantize(
        tok,
        quant_config={
            "w_bit": w_bit,
            "q_group_size": group,
            "zero_point": True,
            "version": "GEMM",
        },
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del model
    torch.cuda.empty_cache()
    return out_dir


def evaluate_dir(cfg, model_dir: Path, name: str, w_bit: int):
    tok = AutoTokenizer.from_pretrained(model_dir)
    M.reset_vram_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    ppl = M.perplexity(model, tok, cfg)
    tps = M.throughput(model, tok, cfg)
    vram = M.peak_vram_gb()
    del model
    torch.cuda.empty_cache()

    tasks = M.run_lm_eval(str(model_dir), cfg)
    M.log_result(
        cfg,
        config_name=name,
        method="awq",
        bits=w_bit,
        disk_gb=round(M.dir_size_gb(model_dir), 2),
        peak_vram_gb=round(vram, 2),
        ppl_wikitext2=round(ppl, 3),
        mmlu=tasks.get("mmlu", ""),
        gsm8k=tasks.get("gsm8k", ""),
        tok_per_s=round(tps, 1),
        notes="fully on GPU",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="e.g. w4-g128", default=None)
    args = ap.parse_args()

    cfg = M.load_config()
    for v in cfg["awq"]["variants"]:
        tag = f"w{v['w_bit']}-g{v['q_group_size']}"
        if args.only and args.only != tag:
            continue
        print(f"=== AWQ variant {tag} ===")
        out = quantize_variant(cfg, v["w_bit"], v["q_group_size"])
        evaluate_dir(cfg, out, f"awq-{tag}", v["w_bit"])


if __name__ == "__main__":
    main()
