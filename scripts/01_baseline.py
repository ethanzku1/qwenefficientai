"""Step 1: BF16 baseline. Every later configuration is compared to this row."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from effml import measure as M


def main():
    cfg = M.load_config()
    model_id = cfg["model_id"]
    hw = cfg["hardware"]

    tok = AutoTokenizer.from_pretrained(model_id)
    M.reset_vram_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 4B in BF16 ~8GB: partially offloads on an 8GB card
        max_memory={0: hw["max_gpu_mem"], "cpu": hw["max_cpu_mem"]},
    )
    model.eval()

    ppl = M.perplexity(model, tok, cfg)
    tps = M.throughput(model, tok, cfg)
    vram = M.peak_vram_gb()

    del model
    torch.cuda.empty_cache()

    tasks = M.run_lm_eval(model_id, cfg)

    M.log_result(
        cfg,
        config_name="baseline-bf16",
        method="none",
        bits=16,
        disk_gb="",  # HF cache; record quantized dirs from step 2 onward
        peak_vram_gb=round(vram, 2),
        ppl_wikitext2=round(ppl, 3),
        mmlu=tasks.get("mmlu", ""),
        gsm8k=tasks.get("gsm8k", ""),
        tok_per_s=round(tps, 1),
        notes="device_map=auto (partial CPU offload on 8GB card)",
    )


if __name__ == "__main__":
    main()
