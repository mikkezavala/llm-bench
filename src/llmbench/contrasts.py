"""Paired contrasts: change one factor, hold every other factor fixed.

Because the design is unbalanced (see :func:`llmbench.validation.audit_runs`),
averaging a metric over a whole factor would compare different sets of
conditions. Every comparison here is instead computed *within* a configuration
and only where both sides of the pair were actually measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from llmbench.loading import kv_order, metric_columns
from llmbench.schema import BACKENDS, CONFIG_COLS, KV_BASELINE, KV_CLASSES


def _factor_levels(wide: pd.DataFrame, factor: str) -> list:
    """Levels of ``factor`` in interpretable order."""
    if factor == "bench_kv":
        return kv_order(wide)
    if factor == "bench_backend":
        present = set(wide[factor].unique())
        known = [b for b in BACKENDS if b in present]
        return known + sorted(present - set(known))
    return sorted(wide[factor].unique())


def contrast(
    wide: pd.DataFrame,
    factor: str,
    baseline: object | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compare every level of ``factor`` against ``baseline``, pairwise.

    Args:
        wide: Frame from :func:`llmbench.to_wide`.
        factor: The column to vary; one of :data:`llmbench.schema.CONFIG_COLS`.
        baseline: Reference level. Defaults to the first interpretable level.
        metrics: Metric columns to contrast. Defaults to all of them.

    Returns:
        One row per (held-fixed configuration, level, metric) with columns
        ``baseline`` / ``level`` (the two sides of the pair), ``from`` / ``to``
        (their throughputs), ``ratio`` (``to / from``), ``pct`` (percent change)
        and ``log2_ratio`` (symmetric around 0, for plotting).

    Pairs where either side is missing are dropped rather than imputed, so the
    row count reflects what was genuinely measured.
    """
    if factor not in CONFIG_COLS:
        raise ValueError(f"{factor!r} is not a design factor; expected {CONFIG_COLS}")

    levels = _factor_levels(wide, factor)
    if baseline is None:
        baseline = levels[0]
    if baseline not in levels:
        raise ValueError(f"baseline {baseline!r} not present; levels are {levels}")

    metrics = metrics or metric_columns(wide)
    hold = [col for col in CONFIG_COLS if col != factor]

    long = wide.melt(
        id_vars=[*hold, factor],
        value_vars=metrics,
        var_name="metric",
        value_name="t_s",
    )
    keys = [*hold, "metric"]
    base = long[long[factor] == baseline].drop(columns=factor)
    other = long[long[factor] != baseline]

    paired = other.merge(base, on=keys, suffixes=("_to", "_from"), how="inner")
    paired = paired.dropna(subset=["t_s_from", "t_s_to"])

    out = paired[keys].copy()
    out["baseline"] = baseline
    out["level"] = paired[factor].to_numpy()
    out["from"] = paired["t_s_from"].to_numpy()
    out["to"] = paired["t_s_to"].to_numpy()
    out["ratio"] = out["to"] / out["from"]
    out["pct"] = (out["ratio"] - 1) * 100
    out["log2_ratio"] = np.log2(out["ratio"])

    # Carry the KV classification through, so results can be grouped by the kind
    # of KV configuration whether it is the varied factor or a held-fixed one.
    kv_source = "level" if factor == "bench_kv" else "bench_kv"
    if kv_source in out.columns:
        out["kv_class"] = out[kv_source].map(KV_CLASSES)

    ranks = {level: i for i, level in enumerate(levels)}
    return (
        out.assign(_rank=out["level"].map(ranks))
        .sort_values([*hold, "_rank", "metric"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def backend_contrast(wide: pd.DataFrame, baseline: str = "rocm") -> pd.DataFrame:
    """Vulkan relative to ROCm, with model, depth and KV type held fixed."""
    return contrast(wide, "bench_backend", baseline=baseline)


def kv_contrast(wide: pd.DataFrame, baseline: str = KV_BASELINE) -> pd.DataFrame:
    """Each KV cache type relative to the ``f16/f16`` baseline, per backend."""
    return contrast(wide, "bench_kv", baseline=baseline)


def depth_contrast(wide: pd.DataFrame, baseline: int | None = None) -> pd.DataFrame:
    """Deeper context relative to the shallowest measured depth."""
    return contrast(wide, "bench_depth", baseline=baseline)


def summarize(contrasts: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median ratio and range per group, for reporting a spread not a point.

    The median is used rather than the mean because throughput ratios here span
    orders of magnitude when a configuration falls off a supported code path.
    """
    return (
        contrasts.groupby(by, dropna=False)["ratio"]
        .agg(n="size", median_ratio="median", min_ratio="min", max_ratio="max")
        .reset_index()
        .sort_values("median_ratio")
        .reset_index(drop=True)
    )


def as_table(contrasts: pd.DataFrame, decimals: int = 1) -> pd.DataFrame:
    """Render a contrast frame for reading: ratios as ``x``, changes as ``%``.

    Percentages are rounded to one decimal at most. With a single repetition per
    test there is no basis for more precision than that.
    """
    table = contrasts.copy()
    table["from"] = table["from"].round(2)
    table["to"] = table["to"].round(2)
    table["ratio"] = table["ratio"].map(lambda v: f"{v:.2f}x")
    table["pct"] = table["pct"].map(lambda v: f"{v:+.{decimals}f}%")
    return table.drop(columns="log2_ratio")
