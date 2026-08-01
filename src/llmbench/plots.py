"""Figures for the study.

Two conventions are applied everywhere, because the alternative hid the main
effect in earlier drafts of this analysis:

* Throughput axes are logarithmic. Measured values span 6 to 753 t/s; on a
  linear axis pinned to the maximum, a config that falls off a supported code
  path becomes a flat line at the bottom and reads as "no data".
* Ratios are plotted as ``log2`` around a diverging colour scale, so a 2x
  speed-up and a 2x slow-down are visually the same distance from parity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogLocator

from llmbench.loading import kv_order, metric_columns
from llmbench.schema import metric_label

#: Fixed colour per backend so it means the same thing in every figure.
BACKEND_COLORS = {"rocm": "#c1443c", "vulkan": "#2f6f9f"}

_TS_FORMATTER = FuncFormatter(lambda v, _: f"{v:g}")


def use_study_style() -> None:
    """Apply the shared visual style. Call once per notebook or script."""
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "grid.linewidth": 0.5,
        }
    )


def _log_throughput_axis(ax: plt.Axes) -> None:
    """Log y-axis labelled at decades plus the 2x and 5x steps between them.

    Decade-only labels leave too much unlabelled space to read a value off the
    plot when the data spans 6 to 750 t/s.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
    ax.yaxis.set_major_formatter(_TS_FORMATTER)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def savefig(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_kv_sensitivity(wide: pd.DataFrame) -> plt.Figure:
    """Throughput against KV cache type, one row per metric, one column per backend.

    Marker style separates context depths; colour separates models. Reading
    across a row shows how differently the two backends respond to the same KV
    configuration.
    """
    metrics = metric_columns(wide)
    backends = sorted(wide["bench_backend"].unique())
    order = kv_order(wide)
    depths = sorted(wide["bench_depth"].unique())
    models = sorted(wide["bench_model"].unique())

    # Models within a family can trace nearly identical curves. Colour alone
    # would hide one under the other, so each model also gets its own marker and
    # a slightly narrower line than the one before it, leaving the wider line
    # visible underneath as a halo.
    palette = dict(zip(models, sns.color_palette("colorblind"), strict=False))
    markers = dict(zip(models, ["o", "s", "^", "D", "v", "P"], strict=False))
    widths = {model: 3.0 - 0.7 * i for i, model in enumerate(models)}

    fig, axes = plt.subplots(
        len(metrics),
        len(backends),
        figsize=(6.2 * len(backends), 4.2 * len(metrics)),
        sharey="row",
        squeeze=False,
    )

    for row, metric in enumerate(metrics):
        for col, backend in enumerate(backends):
            ax = axes[row][col]
            sub = wide[wide["bench_backend"] == backend]
            for (model, depth), grp in sub.groupby(["bench_model", "bench_depth"]):
                grp = grp.set_index("bench_kv").reindex(order).reset_index()
                ax.plot(
                    grp["bench_kv"],
                    grp[metric],
                    marker=markers[model],
                    color=palette[model],
                    linestyle="-" if depth == depths[0] else "--",
                    markersize=5,
                    linewidth=max(widths[model], 1.0),
                    alpha=0.9,
                    label=f"{model} @ d{depth}",
                )
            ax.set_title(f"{backend} — {metric_label(metric)}")
            ax.set_xlabel("KV cache type (K / V)")
            ax.set_ylabel(metric_label(metric) if col == 0 else "")
            ax.tick_params(axis="x", rotation=20)
            _log_throughput_axis(ax)

    # Collect handles from every panel, not just the first: a series measured on
    # only one backend would otherwise be missing from the legend entirely.
    merged: dict[str, object] = {}
    for row_axes in axes:
        for ax in row_axes:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                merged.setdefault(label, handle)
    labels = sorted(merged)
    fig.legend(
        list(map(merged.get, labels)),
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Throughput by KV cache type — log scale, so collapses stay visible",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


#: Colour saturates beyond a 2x change in either direction. Ratios in this study
#: span 0.44x to 90x; letting the extremes set the scale renders every ordinary
#: difference as indistinguishable white. The exact ratio is annotated in each
#: cell, so saturation costs no information.
RATIO_CLIP_LOG2 = 1.0

_CBAR_LABEL = "log2(ratio), clipped at ±1 (2x) — red = slower, blue = faster"


def _ratio_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    vmax: float = RATIO_CLIP_LOG2,
) -> None:
    annot = matrix.map(lambda v: "" if pd.isna(v) else f"{2**v:.2f}x")
    ax.set_facecolor("#e8e8e8")  # unmeasured cells stay visibly empty
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    ax.set_title(title)


def _attach_ratio_colorbar(fig: plt.Figure, ax: plt.Axes) -> None:
    """Add the shared ratio colour bar outside the grid.

    Called after ``tight_layout``: an axes added beforehand is not part of the
    grid and makes the layout engine emit incorrect geometry.
    """
    cbar_ax = fig.add_axes((1.02, 0.15, 0.015, 0.7))
    fig.colorbar(ax.collections[0], cax=cbar_ax, label=_CBAR_LABEL)


def plot_backend_ratio(backend_contrasts: pd.DataFrame) -> plt.Figure:
    """Vulkan / ROCm throughput ratio per configuration, one panel per metric.

    Cells are annotated with the plain ratio; colour encodes ``log2(ratio)`` so
    that "twice as fast" and "half as fast" are equally far from parity.
    """
    metrics = sorted(backend_contrasts["metric"].unique())
    order = kv_order(backend_contrasts)

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(7.0 * len(metrics), 4.2), squeeze=False
    )
    for i, metric in enumerate(metrics):
        sub = backend_contrasts[backend_contrasts["metric"] == metric]
        matrix = sub.pivot_table(
            index=["bench_model", "bench_depth"],
            columns="bench_kv",
            values="log2_ratio",
        ).reindex(columns=[kv for kv in order if kv in set(sub["bench_kv"])])
        _ratio_heatmap(axes[0][i], matrix, title=metric_label(metric))
        axes[0][i].set_xlabel("KV cache type (K / V)")
        axes[0][i].set_ylabel("model, context depth" if i == 0 else "")

    baseline = backend_contrasts["baseline"].iloc[0]
    level = backend_contrasts["level"].iloc[0]
    fig.suptitle(
        f"{level} relative to {baseline}, per configuration",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _attach_ratio_colorbar(fig, axes[0][0])
    return fig


def plot_kv_penalty(kv_contrasts: pd.DataFrame) -> plt.Figure:
    """Cost of quantising the KV cache, relative to ``f16/f16``, per backend.

    Each panel is one backend and one metric; rows are configurations, columns
    are the non-baseline KV types.
    """
    metrics = sorted(kv_contrasts["metric"].unique())
    backends = sorted(kv_contrasts["bench_backend"].unique())
    order = kv_order(kv_contrasts, column="level")

    fig, axes = plt.subplots(
        len(backends),
        len(metrics),
        figsize=(6.0 * len(metrics), 3.6 * len(backends)),
        squeeze=False,
    )
    for row, backend in enumerate(backends):
        for col, metric in enumerate(metrics):
            sub = kv_contrasts[
                (kv_contrasts["bench_backend"] == backend)
                & (kv_contrasts["metric"] == metric)
            ]
            matrix = sub.pivot_table(
                index=["bench_model", "bench_depth"],
                columns="level",
                values="log2_ratio",
            ).reindex(columns=order)
            _ratio_heatmap(
                axes[row][col],
                matrix,
                title=f"{backend} — {metric_label(metric)}",
            )
            axes[row][col].set_xlabel("KV cache type (K / V)")
            axes[row][col].set_ylabel("model, context depth" if col == 0 else "")

    baseline = kv_contrasts["baseline"].iloc[0]
    fig.suptitle(
        f"Quantised KV cache relative to {baseline}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _attach_ratio_colorbar(fig, axes[0][0])
    return fig


def plot_depth_scaling(depth_contrasts: pd.DataFrame) -> plt.Figure:
    """Throughput change from the shallowest to the deeper context, per backend.

    Only configurations measured at both depths appear, so the bars are true
    paired comparisons.
    """
    metrics = sorted(depth_contrasts["metric"].unique())
    multi_level = depth_contrasts["level"].nunique() > 1
    frame = depth_contrasts.copy()
    frame["config"] = frame["bench_model"] + " · " + frame["bench_kv"]
    if multi_level:
        frame["config"] += " → d" + frame["level"].astype(str)

    # Order rows by model, then by KV type in interpretable order, so the
    # baseline sits at the top of each model's block.
    ranks = {kv: i for i, kv in enumerate(kv_order(frame))}
    row_order = (
        frame.assign(_rank=frame["bench_kv"].map(ranks))
        .sort_values(["bench_model", "_rank"])["config"]
        .unique()
        .tolist()
    )

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.4 * len(metrics), 4.6), sharey=True, squeeze=False
    )
    for i, metric in enumerate(metrics):
        ax = axes[0][i]
        sns.barplot(
            data=frame[frame["metric"] == metric],
            y="config",
            x="pct",
            hue="bench_backend",
            order=row_order,
            palette=BACKEND_COLORS,
            ax=ax,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(metric_label(metric))
        ax.set_xlabel("change vs shallowest depth (%)")
        ax.set_ylabel("model · KV cache type" if i == 0 else "")
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    fig.legend(
        handles,
        labels,
        title="backend",
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.06),
    )
    baseline = depth_contrasts["baseline"].iloc[0]
    fig.suptitle(
        f"Cost of deeper context, relative to d{baseline}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig
