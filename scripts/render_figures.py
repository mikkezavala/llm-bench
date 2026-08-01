"""Regenerate every figure and headline number used by the README.

Run after appending new results::

    python scripts/render_figures.py

Figures land in ``figures/``; the printed summary is what the README quotes, so
re-reading it after a data refresh shows immediately whether the write-up has
gone stale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from llmbench import (
    audit_runs,
    backend_contrast,
    depth_contrast,
    kv_contrast,
    load_runs,
    model_catalog,
    summarize,
    to_wide,
)
from llmbench.loading import DEFAULT_RESULTS
from llmbench.plots import (
    plot_backend_ratio,
    plot_depth_scaling,
    plot_kv_penalty,
    plot_kv_sensitivity,
    savefig,
    use_study_style,
)

FIGURES = Path("figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    use_study_style()

    runs = load_runs(args.results)
    wide = to_wide(runs)
    backend = backend_contrast(wide)
    kv = kv_contrast(wide)
    depth = depth_contrast(wide)

    written = [
        savefig(plot_kv_sensitivity(wide), args.out / "kv-sensitivity.png"),
        savefig(plot_kv_penalty(kv), args.out / "kv-penalty.png"),
        savefig(plot_backend_ratio(backend), args.out / "backend-ratio.png"),
        savefig(plot_depth_scaling(depth), args.out / "depth-scaling.png"),
    ]

    span = f"{runs['test_time'].min()} .. {runs['test_time'].max()}"
    print(f"=== snapshot: {len(runs)} tests, {len(wide)} configs, {span} ===\n")
    print(model_catalog(runs).to_string(index=False), "\n")
    print(audit_runs(runs), "\n")
    print("=== Vulkan / ROCm, grouped by KV class ===")
    print(summarize(backend, ["metric", "kv_class"]).to_string(index=False), "\n")
    print("=== Quantised KV vs f16/f16, per backend ===")
    kv_summary = summarize(kv, ["bench_backend", "metric", "level"])
    print(kv_summary.to_string(index=False), "\n")
    print("=== 16k -> 32k context, per backend ===")
    print(summarize(depth, ["bench_backend", "metric"]).to_string(index=False), "\n")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
