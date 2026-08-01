"""Analysis helpers for the llama.cpp ROCm vs Vulkan study on gfx1151.

Load the raw ``llama-bench`` JSONL, audit what it can support, reshape it, derive
paired contrasts, and plot them. Used by ``analysis.py`` (the notebook) and
``build_site.py``.

Run it directly to refresh ``figures/`` and print every number the README quotes::

    python llmbench.py

Field names from the benchmark log are used verbatim throughout, so any figure or
table traces back to a column that literally exists in ``data/results.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogLocator

if TYPE_CHECKING:
    from plotly import graph_objects as go

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

#: Harness-injected fields that always identify a configuration.
BENCH_COLS: list[str] = ["bench_model", "bench_backend", "bench_depth", "bench_kv"]

#: Harness fields that restate another factor rather than adding one.
#:
#: ``bench_ctk`` / ``bench_ctv`` are the two halves of ``bench_kv``, and
#: ``bench_block`` restates ``bench_depth``. Treating them as factors would not
#: change the row count, but it would make the factorial design look mostly empty:
#: 5 KV pairs would become 5 x 3 x 3 nominal cells with 40 of them impossible.
#: :func:`design_factors` drops any field that is functionally determined by a
#: factor already in the list, so this constant only documents the known cases.
DERIVED_BENCH_COLS: list[str] = ["bench_ctk", "bench_ctv", "bench_block"]

#: Runtime knobs that may legitimately be swept as an additional factor.
#:
#: A knob in this list is *not* required to be constant. Whichever ones vary in a
#: given log are added to the configuration key by :func:`design_factors`, so the
#: sweep is treated as part of the design rather than averaged over. Whichever
#: ones are constant are reported as held fixed.
SWEEPABLE_FIELDS: list[str] = [
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

#: Fields describing the machine, which must not change within one log.
#:
#: ``gpu_info`` is deliberately absent: both backends drive the same physical iGPU
#: but report its name differently (``Radeon 8060S Graphics`` under ROCm,
#: ``... (RADV STRIX_HALO)`` under the Vulkan RADV driver), so it is provenance
#: rather than evidence the hardware changed.
MACHINE_FIELDS: list[str] = ["cpu_info"]


def _determines(runs: pd.DataFrame, source: str, target: str) -> bool:
    """Whether each value of ``source`` implies exactly one value of ``target``."""
    counts = runs.groupby(source, dropna=False)[target].nunique(dropna=False)
    return bool(counts.le(1).all())


def design_factors(runs: pd.DataFrame) -> list[str]:
    """Columns that together identify one configuration in ``runs``.

    Derived from the data rather than fixed in advance, so a newly swept knob is
    picked up automatically and becomes a factor that contrasts hold fixed, rather
    than a dimension the pivot silently averages over. In order:

    1. The four core harness fields.
    2. Any other varying ``bench_*`` field, since a harness that starts recording
       a sweep of its own is recording a factor.
    3. Any varying runtime knob from :data:`SWEEPABLE_FIELDS`.

    A candidate is dropped when a factor already in the list determines it, which
    is what keeps a restatement of an existing factor — ``bench_ctk`` against
    ``bench_kv``, or a harness's ``bench_mmap`` against ``use_mmap`` — from being
    counted twice and making the design look full of impossible cells.
    """
    # Only these are ever candidates. Scanning every column would also reach
    # measurement columns, including the list-valued `samples_ts`.
    candidates = [
        col
        for col in runs.columns
        if col.startswith("bench_") and col not in (*BENCH_COLS, *DERIVED_BENCH_COLS)
    ] + [col for col in SWEEPABLE_FIELDS if col in runs.columns]

    factors = [*BENCH_COLS]
    for candidate in candidates:
        if runs[candidate].nunique(dropna=False) <= 1:
            continue
        if not any(_determines(runs, factor, candidate) for factor in factors):
            factors.append(candidate)
    return factors


def held_fixed(runs: pd.DataFrame) -> list[str]:
    """Sweepable knobs that are constant in ``runs``."""
    return [
        col
        for col in SWEEPABLE_FIELDS
        if col in runs.columns and runs[col].nunique(dropna=False) == 1
    ]

#: KV cache type pairs, ordered by the *kind* of change relative to the
#: ``f16/f16`` baseline rather than alphabetically. Alphabetical order would place
#: ``f16/q8_0`` next to ``q4_0/q4_0``, a step that changes both the key and the
#: value cache at once and is therefore not interpretable.
KV_PAIRS: list[str] = [
    "f16/f16",
    "f16/q8_0",
    "f16/q4_0",
    "q8_0/q8_0",
    "q4_0/q4_0",
]

#: The reference KV configuration all KV contrasts are measured against.
KV_BASELINE: str = "f16/f16"

#: How each KV pair differs from the baseline, as a description of the pair
#: itself. Measurements are grouped by it because they separate along this line;
#: the grouping asserts nothing about why.
KV_CLASSES: dict[str, str] = {
    "f16/f16": "baseline f16 K, f16 V",
    "f16/q8_0": "asymmetric f16 K, quantized V",
    "f16/q4_0": "asymmetric f16 K, quantized V",
    "q8_0/q8_0": "symmetric quantized K and V",
    "q4_0/q4_0": "symmetric quantized K and V",
}

#: Backends compared, in the order used for contrasts (``from`` -> ``to``).
BACKENDS: list[str] = ["rocm", "vulkan"]

#: Metric column -> readable axis label. Metric names are derived from the record
#: itself, so this map is looked up leniently by :func:`metric_label`.
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

#: Location of the raw, never-edited benchmark output, relative to the repo root.
DEFAULT_RESULTS = Path("data/results.jsonl")

#: Where :func:`main` writes figures.
DEFAULT_FIGURES = Path("figures")


def find_results(start: str | Path | None = None) -> Path:
    """Locate ``data/results.jsonl`` by walking up from ``start``.

    Lets the notebook, the site build and the tests all reach the same file
    without caring which directory the process was launched from.
    """
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        results = candidate / DEFAULT_RESULTS
        if results.is_file():
            return results
    raise FileNotFoundError(f"no {DEFAULT_RESULTS} found at or above {here}")


#: Hugging Face cache layout: ``models--ORG--REPO/snapshots/REVISION/FILE.gguf``.
_HF_CACHE = re.compile(
    r"models--(?P<org>[^/]+?)--(?P<repo>[^/]+?)/snapshots/(?P<rev>[^/]+)/(?P<file>[^/]+)$"
)


def scrub_path(model_filename: str) -> str:
    """Strip the local filesystem location from a model path.

    ``llama-bench`` records the absolute path it loaded, which on a personal
    machine contains a home directory and therefore a username. Only the location
    is removed; the identity of the model is kept, since that is what the study is
    about and what makes it reproducible:

    * A Hugging Face cache path becomes ``hf://org/repo@revision/file.gguf``,
      preserving which repository and which revision was loaded.
    * Anything else keeps its filename only, as ``local://file.gguf``.

    Idempotent, so it can be re-run over a log that is already partly scrubbed.
    """
    if model_filename.startswith(("hf://", "local://")):
        return model_filename
    match = _HF_CACHE.search(model_filename)
    if match:
        parts = match.groupdict()
        return f"hf://{parts['org']}/{parts['repo']}@{parts['rev'][:12]}/{parts['file']}"
    return f"local://{PurePosixPath(model_filename).name}"


def scrub_log(path: str | Path) -> dict[str, str]:
    """Rewrite ``path`` in place, removing local filesystem locations.

    Returns the replacements applied. Values are substituted as text rather than
    by re-serialising each record, so every other byte of the log — field order,
    spacing, float formatting — is left exactly as ``llama-bench`` wrote it.
    """
    path = Path(path)
    text = path.read_text()
    originals = {
        record["model_filename"]
        for record in (json.loads(line) for line in text.splitlines() if line.strip())
        if "model_filename" in record
    }
    replacements = {
        original: scrub_path(original)
        for original in originals
        if scrub_path(original) != original
    }
    for original, scrubbed in replacements.items():
        text = text.replace(json.dumps(original)[1:-1], json.dumps(scrubbed)[1:-1])
    path.write_text(text)
    return replacements


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
    fields already present in the record: ``test_type`` (see :func:`_test_type`),
    ``t_s`` (alias of ``avg_ts``), ``n_reps`` and ``rel_stddev`` (replication),
    ``family`` (architecture token of ``model_type``), ``kv_class``, ``quant``
    and ``dynamic_quant`` (whether the file is an Unsloth Dynamic quantisation).
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
    runs["test_type"] = [_test_type(record) for record in records]
    runs["t_s"] = runs["avg_ts"]
    runs["n_reps"] = runs["samples_ts"].map(len)
    runs["rel_stddev"] = runs["stddev_ts"] / runs["avg_ts"]
    runs["family"] = runs["model_type"].str.split().str[0]
    runs["kv_class"] = runs["bench_kv"].map(KV_CLASSES)
    runs["quant"] = runs["model_filename"].map(_quant_label)
    runs["dynamic_quant"] = runs["quant"].str.startswith("UD-")
    return runs


def kv_order(frame: pd.DataFrame, column: str = "bench_kv") -> list[str]:
    """KV pairs present in ``frame``, in interpretable order (see :data:`KV_PAIRS`).

    Any pair not covered by :data:`KV_PAIRS` is appended alphabetically, so a
    newly benchmarked configuration still plots — just without a curated position.
    """
    present = set(frame[column].unique())
    known = [kv for kv in KV_PAIRS if kv in present]
    return known + sorted(present - set(known))


def to_wide(runs: pd.DataFrame) -> pd.DataFrame:
    """Pivot tests into one row per configuration, one column per metric.

    The prompt-processing and token-generation records for the same configuration
    become a single row, so a configuration can be read as a point in
    (prefill, decode) space. The row key is :func:`design_factors`, which includes
    any swept runtime knob — averaging over a swept knob here would produce a
    number belonging to no configuration that was actually run.
    """
    factors = design_factors(runs)
    wide = (
        runs.pivot_table(
            index=factors,
            columns="test_type",
            values="t_s",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    ranks = {kv: i for i, kv in enumerate(kv_order(runs))}
    wide["_kv_rank"] = wide["bench_kv"].map(ranks)
    sort_by = ["bench_model", "bench_backend", "bench_depth", "_kv_rank"]
    wide = (
        wide.sort_values([*sort_by, *factors[len(BENCH_COLS) :]])
        .drop(columns="_kv_rank")
        .reset_index(drop=True)
    )
    wide["kv_class"] = wide["bench_kv"].map(KV_CLASSES)
    return wide


def metric_columns(wide: pd.DataFrame) -> list[str]:
    """Metric columns of a wide frame, prefill first then decode.

    Identified by the naming :func:`_test_type` produces, so the set of factors
    can grow without this needing to know about it.
    """
    metrics = [
        column
        for column in wide.columns
        if isinstance(column, str) and column.startswith(("pp", "tg", "pg"))
    ]
    return sorted(metrics, key=lambda metric: (not metric.startswith("pp"), metric))


def model_catalog(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per benchmarked model, describing exactly what was loaded.

    Raises if a ``bench_model`` label maps to more than one GGUF file, which would
    mean the label is not a reliable identifier.
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


# ---------------------------------------------------------------------------
# Integrity audit
#
# The audit does not clean the data. It states, in one place, what the data can
# and cannot support: which runtime knobs really were held fixed, how much
# replication exists, which comparisons are confounded, and which cells of the
# design were never measured.
# ---------------------------------------------------------------------------

#: A finding that invalidates downstream analysis if left unresolved.
BLOCKER = "blocker"
#: A finding that constrains how results may be interpreted.
CAVEAT = "caveat"
#: A check that passed.
OK = "ok"

_SEVERITY_ORDER = {BLOCKER: 0, CAVEAT: 1, OK: 2}


@dataclass
class Audit:
    """Result of :func:`audit_runs`: an ordered list of findings."""

    findings: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, severity: str, check: str, detail: str) -> None:
        self.findings.append((severity, check, detail))

    @property
    def blockers(self) -> list[tuple[str, str, str]]:
        return [finding for finding in self.findings if finding[0] == BLOCKER]

    def to_frame(self) -> pd.DataFrame:
        """Findings as a DataFrame, most severe first."""
        frame = pd.DataFrame(self.findings, columns=["severity", "check", "detail"])
        return (
            frame.assign(_rank=frame["severity"].map(_SEVERITY_ORDER))
            .sort_values(["_rank", "check"])
            .drop(columns="_rank")
            .reset_index(drop=True)
        )

    def __str__(self) -> str:
        lines = [f"[{sev:<7}] {check}: {detail}" for sev, check, detail in self.findings]
        counts = pd.Series([f[0] for f in self.findings]).value_counts().to_dict()
        summary = ", ".join(f"{n} {sev}" for sev, n in sorted(counts.items()))
        return "\n".join([*lines, "", f"{len(self.findings)} checks — {summary}"])


def coverage_matrix(runs: pd.DataFrame) -> pd.DataFrame:
    """Tests measured per ``(model, depth)`` x ``(backend, kv)`` cell.

    A fully replicated design would show ``2`` everywhere (one prefill and one
    decode test per cell). ``0`` marks a cell that was never run.
    """
    return pd.crosstab(
        [runs["bench_model"], runs["bench_depth"]],
        [runs["bench_backend"], runs["bench_kv"]],
    )


def missing_cells(runs: pd.DataFrame) -> pd.DataFrame:
    """Cells of the full factorial design that carry no measurement."""
    factors = design_factors(runs)
    levels = {col: sorted(runs[col].unique()) for col in factors}
    measured = set(map(tuple, runs[factors].to_numpy()))
    absent = [combo for combo in product(*levels.values()) if combo not in measured]
    return pd.DataFrame(absent, columns=factors)


def audit_runs(runs: pd.DataFrame) -> Audit:
    """Run every integrity check over a frame from :func:`load_runs`."""
    audit = Audit()
    _check_machine(runs, audit)
    _check_factors(runs, audit)
    _check_replication(runs, audit)
    _check_build_provenance(runs, audit)
    _check_uniqueness(runs, audit)
    _check_test_shapes(runs, audit)
    _check_metric_completeness(runs, audit)
    _check_design_balance(runs, audit)
    return audit


def _check_machine(runs: pd.DataFrame, audit: Audit) -> None:
    """The machine must be the same throughout; anything else is not one study."""
    varying = {
        col: sorted(map(str, runs[col].unique()))
        for col in MACHINE_FIELDS
        if col in runs.columns and runs[col].nunique() > 1
    }
    if varying:
        detail = "; ".join(f"{col}={vals}" for col, vals in varying.items())
        audit.add(BLOCKER, "machine", f"changes across runs: {detail}")
    else:
        audit.add(OK, "machine", f"identical across all {len(runs)} records")


def _check_factors(runs: pd.DataFrame, audit: Audit) -> None:
    """Report which knobs are being swept and which are constant.

    A swept knob is part of the design, not a defect: it is added to the
    configuration key so contrasts hold it fixed rather than averaging over it.
    """
    factors = design_factors(runs)
    swept = factors[len(BENCH_COLS) :]
    fixed = held_fixed(runs)
    levels = ", ".join(f"{col}={runs[col].nunique()}" for col in factors)
    if swept:
        audit.add(
            OK,
            "factors",
            f"{len(factors)} factors ({levels}); "
            f"{', '.join(swept)} varies and is treated as a factor; "
            f"{len(fixed)} other runtime knobs held constant",
        )
    else:
        audit.add(
            OK,
            "factors",
            f"{len(factors)} factors ({levels}); "
            f"all {len(fixed)} other runtime knobs held constant",
        )


def _check_replication(runs: pd.DataFrame, audit: Audit) -> None:
    reps = runs["n_reps"]
    if reps.max() <= 1:
        audit.add(
            CAVEAT,
            "replication",
            f"all {len(runs)} tests are a single repetition, so stddev_ts is 0 by "
            "construction. Run-to-run variance is unmeasured, and differences of a "
            "few percent cannot be distinguished from noise",
        )
        return
    audit.add(
        OK,
        "replication",
        f"{reps.min()}-{reps.max()} repetitions per test; "
        f"worst relative stddev {runs['rel_stddev'].max():.1%}",
    )


def _check_build_provenance(runs: pd.DataFrame, audit: Audit) -> None:
    by_backend = runs.groupby("bench_backend")["build_commit"].unique()
    commits = {backend: sorted(vals) for backend, vals in by_backend.items()}
    distinct = {commit for values in commits.values() for commit in values}
    detail = ", ".join(f"{be}={'/'.join(cs)}" for be, cs in commits.items())
    if len(distinct) == 1:
        audit.add(OK, "build provenance", f"single build {distinct.pop()} for all runs")
        return
    sets = list(commits.values())
    disjoint = all(
        not set(a) & set(b) for i, a in enumerate(sets) for b in sets[i + 1 :]
    )
    if disjoint:
        audit.add(
            CAVEAT,
            "build provenance",
            f"each backend was built from a different llama.cpp commit ({detail}). "
            "Backend differences are therefore confounded with build version and "
            "cannot be attributed to the backend alone",
        )
    else:
        audit.add(CAVEAT, "build provenance", f"multiple builds present ({detail})")


def _check_uniqueness(runs: pd.DataFrame, audit: Audit) -> None:
    keys = [*design_factors(runs), "test_type"]
    duplicated = runs[runs.duplicated(keys, keep=False)]
    if duplicated.empty:
        audit.add(OK, "uniqueness", "exactly one record per (config, test_type)")
    else:
        audit.add(
            BLOCKER,
            "uniqueness",
            f"{len(duplicated)} records share a (config, test_type) key and would be "
            "averaged silently by the pivot",
        )


def _check_test_shapes(runs: pd.DataFrame, audit: Audit) -> None:
    shapes = sorted(runs["test_type"].unique())
    unexpected = [shape for shape in shapes if not shape.startswith(("pp", "tg"))]
    if unexpected:
        audit.add(BLOCKER, "test shapes", f"unrecognised test types: {unexpected}")
    else:
        audit.add(OK, "test shapes", f"metrics measured: {shapes}")


def _check_metric_completeness(runs: pd.DataFrame, audit: Audit) -> None:
    wide = to_wide(runs)
    metrics = metric_columns(wide)
    incomplete = wide[wide[metrics].isna().any(axis=1)]
    if incomplete.empty:
        audit.add(
            OK,
            "metric completeness",
            f"every one of {len(wide)} configs has all of {metrics}",
        )
    else:
        audit.add(
            BLOCKER,
            "metric completeness",
            f"{len(incomplete)} configs are missing at least one metric",
        )


def _check_design_balance(runs: pd.DataFrame, audit: Audit) -> None:
    absent = missing_cells(runs)
    if absent.empty:
        audit.add(OK, "design balance", "design is fully crossed")
        return
    total = 1
    for col in design_factors(runs):
        total *= runs[col].nunique()
    per_model = absent.groupby("bench_model").size().sort_values(ascending=False)
    audit.add(
        CAVEAT,
        "design balance",
        f"{len(absent)} of {total} factorial cells were never run "
        f"(missing per model: {per_model.to_dict()}). An unpaired mean over the whole "
        "frame would mix different sets of conditions, so every comparison in this "
        "study is paired within a configuration",
    )


# ---------------------------------------------------------------------------
# Paired contrasts
#
# Because the design is unbalanced, averaging a metric over a whole factor would
# compare different sets of conditions. Every comparison here is computed *within*
# a configuration, and only where both sides of the pair were actually measured.
# ---------------------------------------------------------------------------


#: The ``llama.cpp`` default for knobs whose default is not their sorted-first
#: value. Used as the contrast baseline, so a swept knob reads as "what changing
#: it away from the default did" rather than the reverse.
KNOB_DEFAULTS: dict[str, object] = {"use_mmap": True, "flash_attn": 1}


def _factor_levels(wide: pd.DataFrame, factor: str) -> list:
    """Levels of ``factor`` in interpretable order, reference level first."""
    if factor == "bench_kv":
        return kv_order(wide)
    present = set(wide[factor].unique())
    if factor == "bench_backend":
        known = [backend for backend in BACKENDS if backend in present]
        return known + sorted(present - set(known))
    levels = sorted(present)
    default = KNOB_DEFAULTS.get(factor)
    if default in present:
        levels = [default, *(level for level in levels if level != default)]
    return levels


def contrast(
    wide: pd.DataFrame,
    factor: str,
    baseline: object | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compare every level of ``factor`` against ``baseline``, pairwise.

    Args:
        wide: Frame from :func:`to_wide`.
        factor: The column to vary; one of :func:`design_factors`.
        baseline: Reference level. Defaults to the first interpretable level.
        metrics: Metric columns to contrast. Defaults to all of them.

    Returns:
        One row per (held-fixed configuration, level, metric) with columns
        ``baseline`` / ``level`` (the two sides of the pair), ``from`` / ``to``
        (their throughputs), ``ratio`` (``to / from``), ``pct`` (percent change)
        and ``log2_ratio`` (symmetric around 0, for plotting).

    Pairs where either side is missing are dropped rather than imputed, so the row
    count reflects what was genuinely measured.
    """
    factors = design_factors(wide)
    if factor not in factors:
        raise ValueError(f"{factor!r} is not a design factor; expected {factors}")

    levels = _factor_levels(wide, factor)
    if baseline is None:
        baseline = levels[0]
    if baseline not in levels:
        raise ValueError(f"baseline {baseline!r} not present; levels are {levels}")

    metrics = metrics or metric_columns(wide)
    hold = [col for col in factors if col != factor]

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

    The median is used rather than the mean because the measured ratios span
    roughly 0.01x to 90x, and a mean over that range is dominated by its
    largest members.
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


# ---------------------------------------------------------------------------
# Figures
#
# Two conventions apply everywhere, because the alternative hid the main effect in
# earlier drafts of this analysis:
#
# * Throughput axes are logarithmic. Measured values span 6 to 753 t/s; on a
#   linear axis pinned to the maximum, the smallest values flatten against the
#   bottom of the plot and read as "no data".
# * Ratios are plotted as log2 on a diverging colour scale, so a 2x speed-up and a
#   2x slow-down are the same visual distance from parity.
# ---------------------------------------------------------------------------

#: Fixed colour per backend so it means the same thing in every figure.
BACKEND_COLORS = {"rocm": "#c1443c", "vulkan": "#2f6f9f"}

#: Colour saturates beyond a 2x change in either direction. Ratios in this study
#: span 0.44x to 90x; letting the extremes set the scale renders every ordinary
#: difference as indistinguishable white. The exact ratio is annotated in each
#: cell, so saturation costs no information.
RATIO_CLIP_LOG2 = 1.0

_CBAR_LABEL = "log2(ratio), clipped at ±1 (2x) — red = slower, blue = faster"
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


def savefig(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def _log_throughput_axis(ax: plt.Axes) -> None:
    """Log y-axis labelled at decades plus the 2x and 5x steps between them.

    Decade-only labels leave too much unlabelled space to read a value off the
    plot when the data spans 6 to 750 t/s.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1))
    )
    ax.yaxis.set_major_formatter(_TS_FORMATTER)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def plot_kv_sensitivity(wide: pd.DataFrame) -> plt.Figure:
    """Throughput against KV cache type, one row per metric, one column per backend.

    Reading across a row shows how differently the two backends respond to the
    same KV configuration.
    """
    metrics = metric_columns(wide)
    backends = sorted(wide["bench_backend"].unique())
    order = kv_order(wide)
    depths = sorted(wide["bench_depth"].unique())
    models = sorted(wide["bench_model"].unique())

    # Models within a family can trace nearly identical curves. Colour alone would
    # hide one under the other, so each model also gets its own marker and a
    # slightly narrower line than the one before it, leaving the wider line
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
            for (model, depth), group in sub.groupby(["bench_model", "bench_depth"]):
                group = group.set_index("bench_kv").reindex(order).reset_index()
                ax.plot(
                    group["bench_kv"],
                    group[metric],
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
        "Throughput by KV cache type — log scale, so small values stay readable",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


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

    Called after ``tight_layout``: an axes added beforehand is not part of the grid
    and makes the layout engine emit incorrect geometry.
    """
    cbar_ax = fig.add_axes((1.02, 0.15, 0.015, 0.7))
    fig.colorbar(ax.collections[0], cax=cbar_ax, label=_CBAR_LABEL)


def plot_backend_ratio(backend_contrasts: pd.DataFrame) -> plt.Figure:
    """Per-configuration ratio between the two backend builds, one panel per metric.

    Cells are annotated with the plain ratio; colour encodes ``log2(ratio)``. The
    two builds are different commits, so a cell is a reading from two builds and
    not an attribution to the backend.
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
        f"{level} build read against {baseline} build — different commits, "
        "not a backend comparison",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _attach_ratio_colorbar(fig, axes[0][0])
    return fig


def plot_kv_penalty(kv_contrasts: pd.DataFrame) -> plt.Figure:
    """Cost of quantising the KV cache, relative to ``f16/f16``, per backend."""
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

    # Order rows by model, then by KV type in interpretable order, so the baseline
    # sits at the top of each model's block.
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


# ---------------------------------------------------------------------------
# Interactive figures
#
# The static PNGs above are what the README embeds, since GitHub renders no
# JavaScript. The published site uses these instead: with 4 models x 2 depths x
# 2 backends on one axis, being able to isolate a series by clicking the legend
# and read exact values on hover is the difference between a readable chart and
# a thicket.
# ---------------------------------------------------------------------------


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """One row per (configuration, metric) — the shape plotting libraries want."""
    factors = [*design_factors(wide), "kv_class"]
    return wide.melt(
        id_vars=[col for col in factors if col in wide.columns],
        value_vars=metric_columns(wide),
        var_name="metric",
        value_name="t_s",
    ).dropna(subset=["t_s"])


def plot_throughput_interactive(wide: pd.DataFrame, height: int = 760) -> go.Figure:
    """Throughput against KV cache type: one row per metric, one column per backend.

    Colour separates models and dash style separates context depths, matching the
    static figure. Hover reports every factor of the configuration, so a point can
    be identified without cross-referencing the legend.
    """
    import plotly.express as px

    long = to_long(wide)
    factors = design_factors(wide)
    hover = [col for col in factors if col not in {"bench_kv", "bench_model"}]

    fig = px.line(
        long,
        x="bench_kv",
        y="t_s",
        color="bench_model",
        line_dash="bench_depth",
        facet_col="bench_backend",
        facet_row="metric",
        markers=True,
        log_y=True,
        height=height,
        category_orders={
            "bench_kv": kv_order(wide),
            "metric": metric_columns(wide),
            "bench_backend": _factor_levels(wide, "bench_backend"),
        },
        hover_data=hover,
        labels={
            "bench_kv": "KV cache type (K / V)",
            "t_s": "tokens/second",
            "bench_model": "model",
            "bench_depth": "context depth",
            "bench_backend": "backend",
        },
    )
    # Each row is a different metric on a different scale; sharing one y range
    # would flatten decode against the bottom of the prefill row.
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        legend_title_text="model, depth",
        margin={"l": 70, "r": 20, "t": 60, "b": 60},
        hovermode="closest",
        title=(
            "Throughput by KV cache type — log axis; "
            "click a legend entry to isolate a series"
        ),
    )
    return fig


def figure_html(fig: go.Figure) -> str:
    """A plotly figure as a standalone HTML document.

    Returned as a full document so it can be dropped into an ``iframe``, which is
    the one embedding that renders both in the marimo editor and in the exported
    static page. ``plotly.js`` is loaded from a CDN rather than inlined, which
    keeps the built page around 3 MB rather than 40 MB.
    """
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def render_figures(wide: pd.DataFrame, out: str | Path = DEFAULT_FIGURES) -> list[Path]:
    """Write every figure the write-up references, returning the paths written."""
    use_study_style()
    out = Path(out)
    figures = {
        "kv-sensitivity.png": plot_kv_sensitivity(wide),
        "kv-penalty.png": plot_kv_penalty(kv_contrast(wide)),
        "backend-ratio.png": plot_backend_ratio(backend_contrast(wide)),
        "depth-scaling.png": plot_depth_scaling(depth_contrast(wide)),
    }
    return [savefig(fig, out / name) for name, fig in figures.items()]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main() -> None:
    """Refresh the figures and print every number the README quotes.

    Re-reading this output after appending runs shows immediately whether the
    write-up has gone stale.
    """
    parser = argparse.ArgumentParser(description="Refresh figures and summary tables.")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--scrub-paths",
        action="store_true",
        help="rewrite the log to remove local filesystem locations, then exit",
    )
    args = parser.parse_args()

    results = args.results or find_results(__file__)

    if args.scrub_paths:
        replacements = scrub_log(results)
        for original, scrubbed in replacements.items():
            print(f"{original}\n  -> {scrubbed}")
        print(f"\n{len(replacements)} path(s) rewritten in {results}")
        return

    mpl.use("Agg")
    pd.set_option("display.width", 200)

    runs = load_runs(results)
    wide = to_wide(runs)

    span = f"{runs['test_time'].min()} .. {runs['test_time'].max()}"
    print(f"=== snapshot: {len(runs)} tests, {len(wide)} configs, {span} ===\n")
    print(model_catalog(runs).to_string(index=False), "\n")
    print(audit_runs(runs), "\n")

    summaries = {
        "The two builds read against each other (not a backend comparison)": summarize(
            backend_contrast(wide), ["metric", "kv_class"]
        ),
        "Quantised KV vs f16/f16, per backend": summarize(
            kv_contrast(wide), ["bench_backend", "metric", "level"]
        ),
        "Deeper context, per backend": summarize(
            depth_contrast(wide), ["bench_backend", "metric"]
        ),
    }
    for heading, summary in summaries.items():
        print(f"=== {heading} ===")
        print(summary.to_string(index=False), "\n")

    for path in render_figures(wide, args.figures):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
