"""Sanity-check the environment before running anything expensive."""
import shutil
import sys

import torch


def main():
    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram        : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    for pkg in ("transformers", "datasets", "lm_eval", "awq", "llmcompressor"):
        try:
            mod = __import__(pkg)
            print(f"{pkg:<12}: {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"{pkg:<12}: MISSING")
    print(f"llama.cpp   : {'found' if shutil.which('llama-bench') else 'not on PATH (needed for steps 6-7)'}")
    print(f"disk free   : {shutil.disk_usage('.').free/1e9:.0f} GB")


if __name__ == "__main__":
    main()
