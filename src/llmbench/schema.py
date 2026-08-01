"""Field names, factor orderings and design constants for the benchmark study.

Every name here is a *native* ``llama-bench`` JSONL field (or a field injected by
the run harness, prefixed ``bench_``). Nothing is renamed, so any figure or
table can be traced back to a column that literally exists in
``data/results.jsonl``.
"""

from __future__ import annotations

# --- Identity of a benchmarked configuration --------------------------------

#: Harness-injected fields that together identify one configuration under test.
CONFIG_COLS: list[str] = ["bench_model", "bench_backend", "bench_depth", "bench_kv"]

#: Fields carried through from llama-bench and kept for provenance / auditing.
PROVENANCE_COLS: list[str] = [
    "build_commit",
    "build_number",
    "backends",
    "gpu_info",
    "model_filename",
    "model_type",
    "model_size",
    "model_n_params",
    "test_time",
]

#: Measurement columns: mean throughput and its dispersion across repetitions.
MEASUREMENT_COLS: list[str] = ["avg_ts", "stddev_ts", "samples_ts"]

#: Runtime knobs that were deliberately held fixed for the whole study. The
#: audit fails loudly if any of these varies, because a varying value would
#: silently confound the backend / KV-cache comparison.
#:
#: ``gpu_info`` is deliberately absent: both backends drive the same physical
#: iGPU but report its name differently (``Radeon 8060S Graphics`` under ROCm,
#: ``... (RADV STRIX_HALO)`` under the Vulkan RADV driver), so it belongs to
#: provenance rather than to the controlled set.
CONTROLLED_FIELDS: list[str] = [
    "cpu_info",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "n_gpu_layers",
    "n_cpu_moe",
    "split_mode",
    "flash_attn",
    "no_kv_offload",
    "use_mmap",
    "poll",
    "embeddings",
]

# --- Factor levels and orderings --------------------------------------------

#: KV cache type pairs, ordered by the *kind* of change relative to the
#: ``f16/f16`` baseline rather than alphabetically. Alphabetical order would
#: place ``f16/q8_0`` next to ``q4_0/q4_0``, a step that changes both the key
#: and the value cache at once and is therefore not interpretable.
KV_PAIRS: list[str] = [
    "f16/f16",
    "f16/q8_0",
    "f16/q4_0",
    "q8_0/q8_0",
    "q4_0/q4_0",
]

#: The reference KV configuration all KV contrasts are measured against.
KV_BASELINE: str = "f16/f16"

#: How each KV pair differs from the baseline. This grouping is the explanatory
#: variable for the throughput cliffs observed in the study.
KV_CLASSES: dict[str, str] = {
    "f16/f16": "baseline f16 K, f16 V",
    "f16/q8_0": "asymmetric f16 K, quantized V",
    "f16/q4_0": "asymmetric f16 K, quantized V",
    "q8_0/q8_0": "symmetric quantized K and V",
    "q4_0/q4_0": "symmetric quantized K and V",
}

#: Backends compared, in the order used for contrasts (``from`` -> ``to``).
BACKENDS: list[str] = ["rocm", "vulkan"]

#: Model families keyed by the ``model_type`` architecture token reported by
#: llama.cpp. Members share an attention implementation, which matters because
#: the KV-cache pathologies split cleanly along this boundary.
FAMILIES: dict[str, list[str]] = {
    "qwen35moe": ["qwen", "agentworld"],
    "deepseek2": ["glm", "glm-xl"],
}

# --- Metrics ----------------------------------------------------------------

#: Metric column -> human-readable axis label. Metric names are derived from the
#: record itself (``n_prompt`` > 0 -> ``pp{n_prompt}``, ``n_gen`` > 0 ->
#: ``tg{n_gen}``), so this map is looked up leniently.
METRIC_LABELS: dict[str, str] = {
    "pp2048": "prefill — pp2048 (t/s)",
    "tg128": "decode — tg128 (t/s)",
}


def metric_label(metric: str) -> str:
    """Return a readable axis label for a metric column, with a safe fallback."""
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    if metric.startswith("pp"):
        return f"prefill — {metric} (t/s)"
    if metric.startswith("tg"):
        return f"decode — {metric} (t/s)"
    return f"{metric} (t/s)"
