"""Shared measurement harness.

Every pipeline step imports from here so all configurations are measured
identically and land in the same results/results.csv.
"""

from __future__ import annotations

import csv
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

RESULT_FIELDS = [
    "timestamp", "host", "gpu","model", "config_name", "method", "bits",
    "disk_gb", "peak_vram_gb", "ppl_wikitext2",
    "mmlu", "gsm8k", "tok_per_s", "notes",
]


def load_config(path: str = "configs/lab.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"


# ---------------------------------------------------------------- perplexity
@torch.no_grad()
def perplexity(model, tokenizer, cfg: dict,
               seq_len: int | None = None,
               max_windows: int | None = None) -> float:
    """Sliding-window perplexity on WikiText-2 test."""
    from datasets import load_dataset

    ecfg = cfg["eval"]
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    seq_len = seq_len or ecfg["ppl_seq_len"]
    stride = min(ecfg["ppl_stride"], seq_len)
    n_tokens = input_ids.size(1)

    nlls, counted = [], 0
    prev_end = 0
    windows = 0
    device = next(model.parameters()).device
    for begin in range(0, n_tokens, stride):
        end = min(begin + seq_len, n_tokens)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        targets = ids.clone()
        targets[:, :-trg_len] = -100
        out = model(ids, labels=targets)
        nlls.append(out.loss * trg_len)
        counted += trg_len
        prev_end = end
        windows += 1
        if end == n_tokens or (max_windows and windows >= max_windows):
            break
    return torch.exp(torch.stack(nlls).sum() / counted).item()


# ---------------------------------------------------------------- throughput
@torch.no_grad()
def throughput(model, tokenizer, cfg: dict) -> float:
    """Tokens/sec on a fixed generation prompt (greedy)."""
    ecfg = cfg["eval"]
    device = next(model.parameters()).device
    ids = tokenizer(ecfg["gen_prompt"], return_tensors="pt").input_ids.to(device)
    # warmup
    model.generate(ids, max_new_tokens=16, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=ecfg["gen_tokens"], do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (out.size(1) - ids.size(1)) / dt


# ---------------------------------------------------------------- memory/disk
def peak_vram_gb() -> float:
       if not torch.cuda.is_available():
           return 0.0
       return sum(torch.cuda.max_memory_allocated(d)
                  for d in range(torch.cuda.device_count())) / 1e9

def reset_vram_counter():
   if torch.cuda.is_available():
       for d in range(torch.cuda.device_count()):
           torch.cuda.reset_peak_memory_stats(d)


def dir_size_gb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


# ---------------------------------------------------------------- task evals
def run_lm_eval(model_or_path, cfg: dict, tokenizer=None) -> dict:
    """Run lm-eval harness; returns {task: acc}.
    Accepts either a HF model id / local path (str) or an already-loaded
    model object (pass tokenizer too in that case).
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    ecfg = cfg["eval"]
    if isinstance(model_or_path, str):
        lm = HFLM(pretrained=model_or_path, dtype="bfloat16", batch_size=1)
    else:
        lm = HFLM(pretrained=model_or_path, tokenizer=tokenizer, batch_size=1)

    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=ecfg["tasks"],
        limit=ecfg["task_limit"],
    )
    out = {}
    for task in ecfg["tasks"]:
        metrics = res["results"].get(task, {})
        acc = metrics.get("acc,none") or metrics.get("exact_match,strict-match") or 0.0
        out[task] = round(float(acc), 4)
    return out

# ---------------------------------------------------------------- reset?
def reset_vram_counter():
    if not torch.cuda.is_available():
        return
    torch.cuda.init()          # allocator isn't sized until CUDA initializes
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)    

# ---------------------------------------------------------------- results log
def log_result(cfg: dict, **row):
    """Append one configuration's numbers to the shared CSV (git-tracked)."""
    path = Path(cfg["paths"]["results_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row.setdefault("host", socket.gethostname() or platform.node())
    row.setdefault("gpu", gpu_name())
    row.setdefault("model", cfg.get("model_id", ""))
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
    print(f"[logged] {row.get('config_name')} -> {path}")
