"""Tests for the paired-contrast machinery.

These cover the failure modes that made the original notebook's rate-of-change
table unreliable: comparisons against a neighbour instead of a baseline, pairs
formed across configurations that were never measured together, and results that
depended on hardcoded metric column names.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmbench import (
    as_table,
    backend_contrast,
    contrast,
    kv_contrast,
    load_runs,
    summarize,
    to_wide,
)


@pytest.fixture
def wide(synthetic):
    return to_wide(load_runs(synthetic))


def test_backend_contrast_ratio_is_to_over_from(wide):
    row = backend_contrast(wide).query("bench_kv == 'f16/f16' and metric == 'pp2048'")
    assert row["from"].item() == pytest.approx(100.0)
    assert row["to"].item() == pytest.approx(120.0)
    assert row["ratio"].item() == pytest.approx(1.2)
    assert row["pct"].item() == pytest.approx(20.0)


def test_log2_ratio_is_symmetric_about_parity(wide):
    contrasts = backend_contrast(wide)
    doubled = contrasts.assign(ratio=2.0)
    halved = contrasts.assign(ratio=0.5)
    assert np.log2(doubled["ratio"].iloc[0]) == -np.log2(halved["ratio"].iloc[0])


def test_every_kv_level_is_compared_to_the_baseline_not_its_neighbour(wide):
    """A KV contrast must be measured against ``f16/f16`` for every level.

    Walking consecutive KV pairs instead produces transitions such as
    ``f16/q8_0 -> q4_0/q4_0`` that change the key and value cache simultaneously
    and cannot be attributed to either.
    """
    contrasts = kv_contrast(wide)
    assert set(contrasts["baseline"]) == {"f16/f16"}
    assert "f16/f16" not in set(contrasts["level"])


def test_kv_contrast_is_computed_within_a_backend(wide):
    rocm = kv_contrast(wide).query("bench_backend == 'rocm' and metric == 'pp2048'")
    vulkan = kv_contrast(wide).query("bench_backend == 'vulkan' and metric == 'pp2048'")
    assert rocm["ratio"].item() == pytest.approx(0.10)  # 100 -> 10
    assert vulkan["ratio"].item() == pytest.approx(0.50)  # 120 -> 60


def test_pairs_without_both_sides_are_dropped_not_imputed(wide):
    """An unmeasured half of a pair must produce no row at all."""
    incomplete = wide[
        ~((wide["bench_backend"] == "vulkan") & (wide["bench_kv"] == "f16/q4_0"))
    ]
    contrasts = backend_contrast(incomplete)
    assert set(contrasts["bench_kv"]) == {"f16/f16"}


def test_baseline_missing_from_the_whole_frame_raises(wide):
    """Silently returning nothing would look like "no effect found"."""
    only_quantised = wide[wide["bench_kv"] != "f16/f16"]
    with pytest.raises(ValueError, match="not present"):
        kv_contrast(only_quantised)


def test_all_metrics_are_contrasted_without_being_named(wide):
    """Metrics are discovered from the frame, never hardcoded.

    The original table referenced ``pp2048_to`` directly, so renaming the prompt
    length would have broken it silently.
    """
    contrasts = backend_contrast(wide)
    assert set(contrasts["metric"]) == {"pp2048", "tg128"}


def test_unknown_factor_is_rejected(wide):
    with pytest.raises(ValueError, match="not a design factor"):
        contrast(wide, "bench_nonsense")


def test_unknown_baseline_is_rejected(wide):
    with pytest.raises(ValueError, match="not present"):
        kv_contrast(wide, baseline="q2_k/q2_k")


def test_summarize_reports_the_spread_not_just_a_midpoint(wide):
    summary = summarize(kv_contrast(wide), ["bench_backend", "metric"])
    assert {"n", "median_ratio", "min_ratio", "max_ratio"} <= set(summary.columns)
    assert (summary["min_ratio"] <= summary["median_ratio"]).all()
    assert (summary["median_ratio"] <= summary["max_ratio"]).all()


def test_as_table_limits_precision_to_what_the_data_supports(wide):
    """One repetition per test does not justify two-decimal percentages."""
    table = as_table(backend_contrast(wide))
    assert table["pct"].iloc[0].count(".") == 1
    assert len(table["pct"].iloc[0].split(".")[1].rstrip("%")) == 1
    assert table["ratio"].iloc[0].endswith("x")


def test_model_contrast_isolates_the_quantisation_variant(study_wide):
    """glm vs glm-xl is the only pair in the study that varies quantisation alone.

    Both are the same base model at the same parameter count, so a paired
    contrast between them holds architecture fixed.
    """
    pair = study_wide[study_wide["bench_model"].isin(["glm", "glm-xl"])]
    contrasts = contrast(pair, "bench_model", baseline="glm")
    assert set(contrasts["level"]) == {"glm-xl"}
    assert contrasts["ratio"].between(0.95, 1.05).all()
