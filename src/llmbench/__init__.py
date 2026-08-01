"""Analysis toolkit for the llama.cpp ROCm vs Vulkan benchmark study on gfx1151.

The public surface is intentionally small: load the raw ``llama-bench`` JSONL,
audit it, reshape it, and derive paired contrasts.
"""

from llmbench.contrasts import (
    as_table,
    backend_contrast,
    contrast,
    depth_contrast,
    kv_contrast,
    summarize,
)
from llmbench.loading import (
    DEFAULT_RESULTS,
    find_results,
    kv_order,
    load_runs,
    metric_columns,
    model_catalog,
    to_wide,
)
from llmbench.schema import (
    BACKENDS,
    CONFIG_COLS,
    CONTROLLED_FIELDS,
    FAMILIES,
    KV_BASELINE,
    KV_CLASSES,
    KV_PAIRS,
    metric_label,
)
from llmbench.validation import Audit, audit_runs, coverage_matrix, missing_cells

__all__ = [
    "BACKENDS",
    "CONFIG_COLS",
    "CONTROLLED_FIELDS",
    "DEFAULT_RESULTS",
    "FAMILIES",
    "KV_BASELINE",
    "KV_CLASSES",
    "KV_PAIRS",
    "Audit",
    "as_table",
    "audit_runs",
    "backend_contrast",
    "contrast",
    "coverage_matrix",
    "depth_contrast",
    "find_results",
    "kv_contrast",
    "kv_order",
    "load_runs",
    "metric_columns",
    "metric_label",
    "missing_cells",
    "model_catalog",
    "summarize",
    "to_wide",
]
