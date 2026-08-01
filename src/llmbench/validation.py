"""Integrity checks that must be read before any number in this study is quoted.

The audit does not clean the data. It states, in one place, what the data can
and cannot support: which runtime knobs really were held fixed, how much
replication exists, which comparisons are confounded, and which cells of the
design were never measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from llmbench.loading import metric_columns, to_wide
from llmbench.schema import CONFIG_COLS, CONTROLLED_FIELDS

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
    levels = {col: sorted(runs[col].unique()) for col in CONFIG_COLS}
    measured = set(map(tuple, runs[CONFIG_COLS].to_numpy()))
    absent = [combo for combo in product(*levels.values()) if combo not in measured]
    return pd.DataFrame(absent, columns=CONFIG_COLS)


def audit_runs(runs: pd.DataFrame) -> Audit:
    """Run every integrity check over a frame from :func:`llmbench.load_runs`."""
    audit = Audit()
    _check_controlled_fields(runs, audit)
    _check_replication(runs, audit)
    _check_build_provenance(runs, audit)
    _check_uniqueness(runs, audit)
    _check_test_shapes(runs, audit)
    _check_metric_completeness(runs, audit)
    _check_design_balance(runs, audit)
    return audit


def _check_controlled_fields(runs: pd.DataFrame, audit: Audit) -> None:
    varying = {
        col: sorted(map(str, runs[col].unique()))
        for col in CONTROLLED_FIELDS
        if col in runs.columns and runs[col].nunique() > 1
    }
    if varying:
        detail = "; ".join(f"{col}={vals}" for col, vals in varying.items())
        audit.add(BLOCKER, "controlled fields", f"varies across runs: {detail}")
    else:
        audit.add(
            OK,
            "controlled fields",
            f"all {len(CONTROLLED_FIELDS)} held constant across every run",
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
        not set(a) & set(b)
        for i, a in enumerate(sets)
        for b in sets[i + 1 :]
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
    keys = [*CONFIG_COLS, "test_type"]
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
    for col in CONFIG_COLS:
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
