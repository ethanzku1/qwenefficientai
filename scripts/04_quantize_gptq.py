"""Step 4: GPTQ INT4/INT3 fake-quant (Hessian-aware, error-compensating).

Same measurement suite as 02; logs with method="gptq". Quantization is
in-memory fake-quant -- no auto-gptq / gptqmodel, whose torch-pinned CUDA
kernels would displace the pod's NGC torch 2.8.0. Consequence: disk_gb is
blank and peak_vram_gb measures the BF16 container, exactly as for 02's RTN
rows. The GPTQ-vs-AWQ comparison here is quality-at-matched-bits.

Algorithm lives in src/effml/gptq.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from effml import measure as M
from effml.gptq import gptq_quantize_model


def get_calib(tok, nsamples: int, seqlen: int, seed: int = 0):
    """Calibration token windows.

    C4, deliberately: AWQ calibrated on pileval, and calibrating GPTQ on
    wikitext2-train while reporting wikitext2 perplexity would hand GPTQ an
    in-domain advantage and contaminate the comparison against awq-w4-g128.
    Falls back to wikitext2 only if C4 can't be fetched -- the fallback is
    recorded in the results notes so the contamination is never silent.
    """
    from datasets import load_dataset

    calib_name = "c4"
    try:
        ds = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        texts = ds["text"]
    except Exception as e:                                    # noqa: BLE001
        print(f"[04] C4 unavailable ({type(e).__name__}: {e}); "
              f"falling back to wikitext2-train -- IN-DOMAIN, note it")
        calib_name = "wikitext2-train-INDOMAIN"
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = ["\n\n".join(ds["text"])]

    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)

    # Pack: concatenate shuffled docs into a rolling buffer and slice fixed
    # windows. C4's median doc is far shorter than 2048 tokens, so requiring
    # one doc per window would both fail and bias toward atypical long docs.
    out, buf, have = [], [], 0
    for i in order:
        ids = tok(texts[i], return_tensors="pt").input_ids[0]
        buf.append(ids)
        have += ids.numel()
        if have >= seqlen:
            cat = torch.cat(buf)
            while cat.numel() >= seqlen and len(out) < nsamples:
                out.append(cat[:seqlen].unsqueeze(0))
                cat = cat[seqlen:]
            buf, have = [cat], cat.numel()
        if len(out) >= nsamples:
            break

    if len(out) < nsamples:
        raise RuntimeError(f"only packed {len(out)}/{nsamples} calib windows")
    return out, calib_name + "-packed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="override config model")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--dryrun", action="store_true",
        help="small/fast settings for CPU plumbing checks")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=None,
        help="calibration windows (default 128, or 8 with --dryrun)")
    parser.add_argument("--seqlen", type=int, default=None,
        help="calibration window length (default 2048, or 512 with --dryrun)")
    parser.add_argument("--percdamp", type=float, default=0.01,
        help="Hessian dampening; raise to 0.05 if Cholesky fails")
    parser.add_argument("--tag", default=None, help="suffix for config_name")
    parser.add_argument("--skip-tasks", action="store_true",
        help="skip lm_eval (dry-runs / quick plumbing checks)")
    args = parser.parse_args()

    cfg = M.load_config()
    model_id = args.model or cfg["model_id"]
    hw = cfg["hardware"]
    seq_len = 512 if args.dryrun else None
    max_windows = 8 if args.dryrun else None

    nsamples = args.nsamples if args.nsamples is not None else (8 if args.dryrun else 128)
    calib_seqlen = args.seqlen if args.seqlen is not None else (512 if args.dryrun else 2048)

    use_cuda = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"

    tok = AutoTokenizer.from_pretrained(model_id)

    if use_cuda:
        M.reset_vram_counter()
        # NB: device_map={"": 0}, NOT "auto". GPTQ walks decoder blocks
        # sequentially, feeding block i's outputs into block i+1; if "auto"
        # shards blocks across GPUs those activations land on the wrong
        # device. The 4B in BF16 (~8GB) plus activation buffers (~3GB) fits
        # one A100-80GB with room to spare.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map={"": 0},
            max_memory={0: hw["max_gpu_mem"]},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if args.dryrun else torch.float32,
        )
    model.eval()

    devs = {p.device for p in model.parameters()}
    print(f"[04] model on {devs}")
    if len(devs) > 1:
        raise RuntimeError(f"GPTQ needs a single device, got {devs}")

    calib, calib_name = get_calib(tok, nsamples, calib_seqlen)
    print(f"[04] calib: {nsamples} x {calib_seqlen} tokens from {calib_name}")

    # Quantizes every Linear inside the decoder blocks. lm_head is outside
    # model.model.layers and so is never touched -- same reason as 02: Qwen3
    # ties lm_head to the input embeddings.
    gptq_quantize_model(
        model,
        calib,
        n_bits=args.bits,
        group_size=args.group_size,
        percdamp=args.percdamp,
    )
    del calib

    model.config.use_cache = False
    ppl = M.perplexity(model, tok, cfg, seq_len=seq_len, max_windows=max_windows)

    model.config.use_cache = True
    tps = M.throughput(model, tok, cfg)

    vram = M.peak_vram_gb() if use_cuda else 0.0

    # Fake-quant lives only in this process, so lm_eval must run against the
    # live model object rather than reloading from model_id -- same path AWQ
    # forced on us (transformers won't load AWQ checkpoints without
    # gptqmodel, which we don't install).
    tasks = {}
    if not args.dryrun and not args.skip_tasks:
        try:
            tasks = M.run_lm_eval(model, cfg)
            print(f"[04] tasks: {tasks}")
        except Exception as e:                      # noqa: BLE001
            print(f"[04] lm_eval FAILED ({type(e).__name__}: {e}) "
                  f"-- logging ppl/tps anyway")
            tasks = {}

    del model
    if use_cuda:
        torch.cuda.empty_cache()

    config_name = f"gptq-w{args.bits}-g{args.group_size}" + ("-dryrun" if args.dryrun else "")
    if args.tag:
        config_name += f"-{args.tag}"

    M.log_result(
        cfg,
        config_name=config_name,
        method="gptq",
        bits=args.bits,
        disk_gb="",
        peak_vram_gb=round(vram, 2),
        ppl_wikitext2=round(ppl, 3),
        mmlu=tasks.get("mmlu", ""),
        gsm8k=tasks.get("gsm8k", ""),
        tok_per_s=round(tps, 1),
        notes=f"fake-quant, calib={calib_name} n={nsamples} len={calib_seqlen}, "
              f"percdamp={args.percdamp}, no act_order"
              + (f", CPU dry-run on {model_id}" if args.dryrun else ""),
    )


if __name__ == "__main__":
    main()