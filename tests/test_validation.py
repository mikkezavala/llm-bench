"""Tests for the integrity audit, including against the real study data."""

from __future__ import annotations

from tests.conftest import make_record, write_jsonl

from llmbench import audit_runs, coverage_matrix, load_runs, missing_cells
from llmbench.validation import BLOCKER, CAVEAT, OK


def _checks(runs) -> dict[str, str]:
    """Map check name -> severity."""
    return {check: severity for severity, check, _ in audit_runs(runs).findings}


def test_single_repetition_is_flagged(synthetic):
    assert _checks(load_runs(synthetic))["replication"] == CAVEAT


def test_replication_passes_when_repetitions_exist(tmp_path):
    path = tmp_path / "replicated.jsonl"
    write_jsonl(path, [make_record(samples_ts=[99.0, 100.0, 101.0], stddev_ts=1.0)])
    assert _checks(load_runs(path))["replication"] == OK


def test_backend_specific_builds_are_flagged_as_confounded(synthetic):
    """The headline comparison must not be quoted without this caveat."""
    audit = audit_runs(load_runs(synthetic))
    build = next(f for f in audit.findings if f[1] == "build provenance")
    assert build[0] == CAVEAT
    assert "confounded" in build[2]


def test_single_build_passes_provenance(tmp_path):
    path = tmp_path / "onebuild.jsonl"
    write_jsonl(
        path,
        [
            make_record(),
            make_record(bench_backend="vulkan", backends="Vulkan"),
        ],
    )
    assert _checks(load_runs(path))["build provenance"] == OK


def test_a_varying_controlled_knob_is_a_blocker(tmp_path):
    """Changing thread count mid-study would confound every comparison."""
    path = tmp_path / "drift.jsonl"
    write_jsonl(path, [make_record(), make_record(n_threads=16, bench_depth=32768)])
    checks = _checks(load_runs(path))
    assert checks["controlled fields"] == BLOCKER


def test_gpu_name_differing_by_driver_is_not_a_blocker(tmp_path):
    """ROCm and the Vulkan RADV driver name the same iGPU differently.

    That is provenance, not a change in the machine under test.
    """
    path = tmp_path / "gpuname.jsonl"
    write_jsonl(
        path,
        [
            make_record(),
            make_record(
                bench_backend="vulkan",
                backends="Vulkan",
                gpu_info="Radeon 8060S Graphics (RADV STRIX_HALO)",
            ),
        ],
    )
    assert _checks(load_runs(path))["controlled fields"] == OK


def test_duplicate_config_is_a_blocker(tmp_path):
    """Two records for one cell would be averaged invisibly by the pivot."""
    path = tmp_path / "dupes.jsonl"
    write_jsonl(path, [make_record(), make_record(avg_ts=50.0, samples_ts=[50.0])])
    assert _checks(load_runs(path))["uniqueness"] == BLOCKER


def test_missing_cells_are_reported(tmp_path):
    path = tmp_path / "sparse.jsonl"
    write_jsonl(
        path,
        [
            make_record(),
            make_record(n_prompt=0, n_gen=128),
            make_record(bench_depth=32768, n_depth=32768),
            make_record(bench_depth=32768, n_depth=32768, n_prompt=0, n_gen=128),
            make_record(bench_backend="vulkan", backends="Vulkan"),
            make_record(bench_backend="vulkan", backends="Vulkan", n_prompt=0, n_gen=128),
        ],
    )
    runs = load_runs(path)
    absent = missing_cells(runs)
    assert len(absent) == 1
    assert absent.iloc[0].to_dict() == {
        "bench_model": "test",
        "bench_backend": "vulkan",
        "bench_depth": 32768,
        "bench_kv": "f16/f16",
    }
    assert _checks(runs)["design balance"] == CAVEAT


def test_coverage_matrix_counts_two_tests_per_measured_cell(synthetic):
    matrix = coverage_matrix(load_runs(synthetic))
    assert set(matrix.to_numpy().ravel()) == {2}


def test_study_data_has_no_blockers(study_runs):
    """The real dataset must be free of blocking defects for the write-up to hold."""
    audit = audit_runs(study_runs)
    assert audit.blockers == [], audit


def test_study_data_carries_the_expected_caveats(study_runs):
    checks = _checks(study_runs)
    assert checks["replication"] == CAVEAT
    assert checks["build provenance"] == CAVEAT
    assert checks["design balance"] == CAVEAT
    assert checks["metric completeness"] == OK
    assert checks["uniqueness"] == OK
