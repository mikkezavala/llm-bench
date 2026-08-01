"""Load raw ``llama-bench`` JSONL output into tidy and wide DataFrames."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llmbench.schema import CONFIG_COLS, KV_CLASSES, KV_PAIRS

#: Location of the raw, never-edited benchmark output, relative to the repo root.
DEFAULT_RESULTS = Path("data/results.jsonl")


def find_results(start: str | Path | None = None) -> Path:
    """Locate ``data/results.jsonl`` by walking up from ``start``.

    Lets a notebook, a script and a test all reach the same file without caring
    which directory the process was launched from.
    """
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        results = candidate / DEFAULT_RESULTS
        if results.is_file():
            return results
    raise FileNotFoundError(f"no {DEFAULT_RESULTS} found at or above {here}")


def _test_type(record: dict) -> str:
    """Name the test a record represents, from its own ``n_prompt`` / ``n_gen``.

    ``llama-bench`` emits one record per test. A prompt-processing test has
    ``n_gen == 0``; a token-generation test has ``n_prompt == 0``. A combined
    prompt+generation test (``-pg``) sets both, and gets its own label so it
    surfaces in the audit instead of being silently dropped.
    """
    n_prompt, n_gen = record["n_prompt"], record["n_gen"]
    if n_prompt > 0 and n_gen == 0:
        return f"pp{n_prompt}"
    if n_gen > 0 and n_prompt == 0:
        return f"tg{n_gen}"
    if n_prompt > 0 and n_gen > 0:
        return f"pg{n_prompt}+{n_gen}"
    raise ValueError(f"record has neither n_prompt nor n_gen: {record!r}")


def _quant_label(model_filename: str) -> str:
    """Extract the quantisation tag from a GGUF filename, e.g. ``UD-Q4_K_XL``."""
    parts = Path(model_filename).stem.split("-")
    for i, part in enumerate(parts):
        if part.startswith("Q") and "_" in part:
            tag = "-".join(parts[i:])
            return f"UD-{tag}" if i and parts[i - 1] == "UD" else tag
    return Path(model_filename).stem


def load_runs(path: str | Path = DEFAULT_RESULTS) -> pd.DataFrame:
    """Read the JSONL log into one row per benchmark test.

    Native fields are preserved verbatim. Added columns are derived only from
    fields already present in the record:

    ``test_type``
        ``pp{n_prompt}`` or ``tg{n_gen}``, see :func:`_test_type`.
    ``t_s``
        Alias of ``avg_ts``, the mean throughput in tokens/second.
    ``n_reps``
        Number of timed repetitions behind ``avg_ts`` (``len(samples_ts)``).
    ``rel_stddev``
        ``stddev_ts / avg_ts``; zero when only one repetition was run.
    ``family``
        Architecture token of ``model_type`` (e.g. ``qwen35moe``).
    ``kv_class``
        Whether ``bench_kv`` is the baseline, asymmetric, or symmetric.
    ``quant``
        Quantisation tag parsed from the GGUF filename.
    ``dynamic_quant``
        Whether the file is an Unsloth Dynamic (``UD``) quantisation.
    """
    path = Path(path)
    records: list[dict] = []
    with path.open() as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON") from exc

    if not records:
        raise ValueError(f"{path} contains no records")

    runs = pd.DataFrame.from_records(records)
    runs["test_type"] = [_test_type(r) for r in records]
    runs["t_s"] = runs["avg_ts"]
    runs["n_reps"] = runs["samples_ts"].map(len)
    runs["rel_stddev"] = runs["stddev_ts"] / runs["avg_ts"]
    runs["family"] = runs["model_type"].str.split().str[0]
    runs["kv_class"] = runs["bench_kv"].map(KV_CLASSES)
    runs["quant"] = runs["model_filename"].map(_quant_label)
    runs["dynamic_quant"] = runs["quant"].str.startswith("UD-")
    return runs


def kv_order(frame: pd.DataFrame, column: str = "bench_kv") -> list[str]:
    """KV pairs present in ``frame``, in interpretable order (see ``KV_PAIRS``).

    Any pair not covered by ``KV_PAIRS`` is appended alphabetically, so a newly
    benchmarked configuration still plots — just without a curated position.
    """
    present = set(frame[column].unique())
    known = [kv for kv in KV_PAIRS if kv in present]
    return known + sorted(present - set(known))


def to_wide(runs: pd.DataFrame) -> pd.DataFrame:
    """Pivot tests into one row per configuration, one column per metric.

    The prompt-processing and token-generation records for the same
    ``(model, backend, depth, kv)`` become a single row, so a configuration can
    be read as a point in (prefill, decode) space.
    """
    wide = (
        runs.pivot_table(
            index=CONFIG_COLS,
            columns="test_type",
            values="t_s",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    ranks = {kv: i for i, kv in enumerate(kv_order(runs))}
    wide["_kv_rank"] = wide["bench_kv"].map(ranks)
    wide = (
        wide.sort_values(["bench_model", "bench_backend", "bench_depth", "_kv_rank"])
        .drop(columns="_kv_rank")
        .reset_index(drop=True)
    )
    wide["kv_class"] = wide["bench_kv"].map(KV_CLASSES)
    return wide


def metric_columns(wide: pd.DataFrame) -> list[str]:
    """Metric columns of a wide frame, prefill first then decode."""
    extras = {*CONFIG_COLS, "kv_class"}
    metrics = [c for c in wide.columns if c not in extras]
    return sorted(metrics, key=lambda m: (not m.startswith("pp"), m))


def model_catalog(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per benchmarked model, describing exactly what was loaded.

    Raises if a ``bench_model`` label maps to more than one GGUF file, which
    would mean the label is not a reliable identifier.
    """
    cols = ["model_filename", "model_type", "family", "quant", "dynamic_quant"]
    catalog = (
        runs.groupby("bench_model")
        .agg(
            **{col: (col, "unique") for col in cols},
            size_gib=("model_size", lambda s: round(s.iloc[0] / 2**30, 2)),
            params_b=("model_n_params", lambda s: round(s.iloc[0] / 1e9, 2)),
            n_tests=("t_s", "size"),
        )
        .reset_index()
    )
    for col in cols:
        ambiguous = catalog[catalog[col].map(len) > 1]
        if not ambiguous.empty:
            labels = ", ".join(ambiguous["bench_model"])
            raise ValueError(f"bench_model label(s) map to multiple {col}: {labels}")
        catalog[col] = catalog[col].str[0]
    catalog["model_filename"] = catalog["model_filename"].map(lambda p: Path(p).name)
    return catalog
