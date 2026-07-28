"""Step 2: naive round-to-nearest INT4/INT3 fake-quant (the strawman).
Per-group asymmetric quantization of all Linear layers except lm_head.
Same measurement suite as the baseline; logs with method="rtn"."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from effml import measure as M


@torch.no_grad()
def rtn_quantize_(linear: torch.nn.Linear, n_bits: int = 4, group_size: int = 128) -> bool:
    """Fake-quantize a Linear's weight in place. Returns True if per-group,
    False if it fell back to per-channel (in_features not divisible)."""
    W = linear.weight.data
    out_f, in_f = W.shape
    orig_dtype = W.dtype

    per_group = in_f % group_size == 0
    g = group_size if per_group else in_f  # fallback: one group per row

    Wg = W.float().reshape(out_f, in_f // g, g)   # fp32 math

    w_max = Wg.amax(dim=-1, keepdim=True)
    w_min = Wg.amin(dim=-1, keepdim=True)

    qmax = 2 ** n_bits - 1
    scale = (w_max - w_min).clamp(min=1e-8) / qmax
    zero = (-w_min / scale).round()

    Wq = (Wg / scale + zero).round().clamp(0, qmax)
    Wdq = (Wq - zero) * scale

    W.copy_(Wdq.reshape(out_f, in_f).to(orig_dtype))
    return per_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="override config model")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--dryrun", action="store_true",
        help="small/fast settings for CPU plumbing checks")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--tag", default=None, help="suffix for config_name")
    args = parser.parse_args()

    cfg = M.load_config()
    model_id = args.model or cfg["model_id"]
    hw = cfg["hardware"]
    seq_len = 512 if args.dryrun else None
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

    # Quantize every Linear except lm_head. NB: Qwen3 ties lm_head to the
    # input embeddings, so touching lm_head would corrupt embeddings too.
    n_quant, n_fallback = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            per_group = rtn_quantize_(module, n_bits=args.bits, group_size=args.group_size)
            n_quant += 1
            n_fallback += 0 if per_group else 1
    print(f"quantized {n_quant} Linear layers "
          f"({n_fallback} fell back to per-channel), "
          f"w{args.bits} g{args.group_size}")

    model.config.use_cache = False
    ppl = M.perplexity(model, tok, cfg, seq_len=seq_len, max_windows=max_windows)

    model.config.use_cache = True
    tps = M.throughput(model, tok, cfg)

    vram = M.peak_vram_gb() if use_cuda else 0.0

    # NOTE: lm_eval reloads from model_id, i.e. the UNQUANTIZED weights.
    # Fake-quant lives only in this process, so skip tasks here until the
    # harness can evaluate an in-memory model (or we save quantized weights).
    tasks = {}
    if not args.dryrun and not args.skip_tasks:
        tasks = M.run_lm_eval(model, tok, cfg)   # <-- match your real signature
    
    del model
    if use_cuda:
        torch.cuda.empty_cache()

    parser.add_argument("--skip-tasks", action="store_true")

    config_name = f"rtn-w{args.bits}-g{args.group_size}" + ("-dryrun" if args.dryrun else "")
    if args.tag:
        config_name += f"-{args.tag}"

    M.log_result(
        cfg,
        config_name=config_name,
        method="rtn",
        bits=args.bits,
        disk_gb="",
        peak_vram_gb=round(vram, 2),
        ppl_wikitext2=round(ppl, 3),
        mmlu=tasks.get("mmlu", ""),
        gsm8k=tasks.get("gsm8k", ""),
        tok_per_s=round(tps, 1),
        notes=f"fake-quant, {n_quant} linears, {n_fallback} per-channel fallback"
              + (f", CPU dry-run on {model_id}" if args.dryrun else ""),
    )


if __name__ == "__main__":
    main()