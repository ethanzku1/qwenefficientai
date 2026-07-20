"""Render results/results.csv into a markdown tradeoff table + deltas vs baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from effml import measure as M


def main():
    cfg = M.load_config()
    csv_path = Path(cfg["paths"]["results_csv"])
    if not csv_path.exists():
        sys.exit("No results yet — run scripts/01_baseline.py first.")

    df = pd.read_csv(csv_path)
    # keep latest run per config
    df = df.sort_values("timestamp").groupby("config_name", as_index=False).last()

    base = df[df.config_name == "baseline-bf16"]
    if len(base):
        b = base.iloc[0]
        df["ppl_delta"] = (df.ppl_wikitext2 - b.ppl_wikitext2).round(3)
        df["speedup"] = (df.tok_per_s / b.tok_per_s).round(2)

    cols = [c for c in [
        "config_name", "bits", "disk_gb", "peak_vram_gb",
        "ppl_wikitext2", "ppl_delta", "mmlu", "gsm8k",
        "tok_per_s", "speedup", "gpu", "host",
    ] if c in df.columns]

    out = Path("results/REPORT.md")
    out.write_text(
        "# EfficientML Lab — Results\n\n"
        + df[cols].to_markdown(index=False)
        + "\n"
    )
    print(df[cols].to_string(index=False))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
