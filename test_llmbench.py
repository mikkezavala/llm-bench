"""Tests for the analysis helpers.

These cover the failure modes that made the first draft of this analysis
unreliable: comparisons against a neighbouring level instead of a baseline, pairs
formed across configurations that were never measured together, records silently
dropped instead of raising, and results that depended on hardcoded metric names.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import llmbench as lb

# --- Fixtures ---------------------------------------------------------------

#: Every field the loader or audit touches, with study-realistic defaults.
BASE_RECORD = {
    "build_commit": "aaaaaaa",
    "build_number": 1,
    "cpu_info": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
    "gpu_info": "Radeon 8060S Graphics",
    "backends": "ROCm",
    "model_filename": "/models/Test-30B-A3B-Q4_K_M.gguf",
    "model_type": "testarch 30B.A3B Q4_K - Medium",
    "model_size": 18_000_000_000,
    "model_n_params": 30_000_000_000,
    "n_batch": 1024,
    "n_ubatch": 512,
    "n_threads": 8,
    "n_gpu_layers": -1,
    "n_cpu_moe": 0,
    "split_mode": "layer",
    "flash_attn": 1,
    "no_kv_offload": False,
    "use_mmap": True,
    "poll": 50,
    "embeddings": False,
    "type_k": "f16",
    "type_v": "f16",
    "n_prompt": 2048,
    "n_gen": 0,
    "n_depth": 16384,
    "test_time": "2026-07-31T21:00:00Z",
    "avg_ts": 100.0,
    "stddev_ts": 0.0,
    "samples_ts": [100.0],
    "bench_model": "test",
    "bench_backend": "rocm",
    "bench_depth": 16384,
    "bench_kv": "f16/f16",
}


def make_record(**overrides) -> dict:
    """A benchmark record with study defaults, overridden field by field."""
    return {**BASE_RECORD, **overrides}


def write_jsonl(path, records) -> str:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return str(path)


@pytest.fixture
def synthetic(tmp_path):
    """A tiny fully-crossed design: 2 backends x 2 KV types, prefill and decode.

    Throughputs are chosen so every expected contrast is an exact round number:
    ROCm f16/f16 = 100, and each factor multiplies it by a known amount.
    """
    prefill = {
        ("rocm", "f16/f16"): 100.0,
        ("rocm", "f16/q4_0"): 10.0,  # 0.10x — a collapse
        ("vulkan", "f16/f16"): 120.0,  # 1.20x vs rocm
        ("vulkan", "f16/q4_0"): 60.0,  # 0.50x vs vulkan baseline
    }
    records = []
    for (backend, kv), value in prefill.items():
        type_k, type_v = kv.split("/")
        shared = {
            "bench_backend": backend,
            "backends": "ROCm" if backend == "rocm" else "Vulkan",
            "build_commit": "aaaaaaa" if backend == "rocm" else "bbbbbbb",
            "bench_kv": kv,
            "type_k": type_k,
            "type_v": type_v,
        }
        records.append(
            make_record(
                **shared, n_prompt=2048, n_gen=0, avg_ts=value, samples_ts=[value]
            )
        )
        records.append(
            make_record(
                **shared,
                n_prompt=0,
                n_gen=128,
                avg_ts=value / 10,
                samples_ts=[value / 10],
            )
        )
    return write_jsonl(tmp_path / "synthetic.jsonl", records)


@pytest.fixture
def wide(synthetic):
    return lb.to_wide(lb.load_runs(synthetic))


@pytest.fixture(scope="session")
def study_runs():
    """The real ``data/results.jsonl``, as loaded for the write-up."""
    return lb.load_runs(lb.find_results(__file__))


@pytest.fixture(scope="session")
def study_wide(study_runs):
    return lb.to_wide(study_runs)


# --- Loading ----------------------------------------------------------------


def test_test_type_derived_from_prompt_and_gen_counts(synthetic):
    assert set(lb.load_runs(synthetic)["test_type"]) == {"pp2048", "tg128"}


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "gappy.jsonl"
    write_jsonl(path, [make_record()])
    path.write_text(path.read_text() + "\n\n")
    assert len(lb.load_runs(path)) == 1


def test_malformed_json_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": 1}\nnot json\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        lb.load_runs(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no records"):
        lb.load_runs(path)


def test_record_with_no_test_is_rejected_not_dropped(tmp_path):
    """A record llama-bench should never emit must fail loudly.

    Silently skipping these means a truncated log produces a smaller table with no
    indication that anything is missing.
    """
    path = tmp_path / "notest.jsonl"
    write_jsonl(path, [make_record(n_prompt=0, n_gen=0)])
    with pytest.raises(ValueError, match="neither n_prompt nor n_gen"):
        lb.load_runs(path)


def test_combined_prompt_and_gen_test_is_labelled_not_dropped(tmp_path):
    path = tmp_path / "pg.jsonl"
    write_jsonl(path, [make_record(n_prompt=512, n_gen=128)])
    assert lb.load_runs(path)["test_type"].tolist() == ["pg512+128"]


def test_pivot_puts_prefill_and_decode_on_one_row(wide):
    assert len(wide) == 4  # 2 backends x 2 KV types
    assert lb.metric_columns(wide) == ["pp2048", "tg128"]
    assert not wide[["pp2048", "tg128"]].isna().to_numpy().any()


def test_kv_order_is_interpretable_not_alphabetical(study_runs):
    """Baseline first, then asymmetric, then symmetric.

    Alphabetical order would put ``q4_0/q4_0`` before ``q8_0/q8_0`` and place
    ``f16/q8_0`` adjacent to ``q4_0/q4_0``, a step that changes both caches at once
    and cannot be read as a single transition.
    """
    assert lb.kv_order(study_runs) == [
        "f16/f16",
        "f16/q8_0",
        "f16/q4_0",
        "q8_0/q8_0",
        "q4_0/q4_0",
    ]


def test_unknown_kv_pair_is_appended_rather_than_dropped(tmp_path):
    path = tmp_path / "newkv.jsonl"
    write_jsonl(
        path,
        [make_record(), make_record(bench_kv="q5_1/q5_1", type_k="q5_1", type_v="q5_1")],
    )
    assert lb.kv_order(lb.load_runs(path)) == ["f16/f16", "q5_1/q5_1"]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("GLM-4.7-Flash-Q4_K_M.gguf", "Q4_K_M"),
        ("GLM-4.7-Flash-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("Qwen3.6-35B-A3B-Q4_K_M.gguf", "Q4_K_M"),
    ],
)
def test_quant_label_parsed_from_filename(filename, expected):
    assert lb._quant_label(f"/models/{filename}") == expected


def test_model_catalog_rejects_an_ambiguous_label(tmp_path):
    """One ``bench_model`` label must mean one GGUF file.

    If a label is reused across files, every per-model comparison in the study
    silently mixes two models.
    """
    path = tmp_path / "ambiguous.jsonl"
    write_jsonl(
        path,
        [
            make_record(model_filename="/models/A-Q4_K_M.gguf"),
            make_record(
                model_filename="/models/B-Q4_K_M.gguf",
                bench_kv="q8_0/q8_0",
                type_k="q8_0",
                type_v="q8_0",
            ),
        ],
    )
    with pytest.raises(ValueError, match="multiple model_filename"):
        lb.model_catalog(lb.load_runs(path))


def test_find_results_walks_up_from_a_nested_path():
    assert lb.find_results(__file__).name == "results.jsonl"


def test_find_results_reports_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="at or above"):
        lb.find_results(tmp_path)


# --- Contrasts --------------------------------------------------------------


def test_backend_contrast_ratio_is_to_over_from(wide):
    row = lb.backend_contrast(wide).query(
        "bench_kv == 'f16/f16' and metric == 'pp2048'"
    )
    assert row["from"].item() == pytest.approx(100.0)
    assert row["to"].item() == pytest.approx(120.0)
    assert row["ratio"].item() == pytest.approx(1.2)
    assert row["pct"].item() == pytest.approx(20.0)


def test_log2_ratio_is_symmetric_about_parity(wide):
    contrasts = lb.backend_contrast(wide)
    doubled = contrasts.assign(ratio=2.0)
    halved = contrasts.assign(ratio=0.5)
    assert np.log2(doubled["ratio"].iloc[0]) == -np.log2(halved["ratio"].iloc[0])


def test_every_kv_level_is_compared_to_the_baseline_not_its_neighbour(wide):
    """A KV contrast must be measured against ``f16/f16`` for every level.

    Walking consecutive KV pairs instead produces transitions such as
    ``f16/q8_0 -> q4_0/q4_0`` that change the key and value cache simultaneously
    and cannot be attributed to either.
    """
    contrasts = lb.kv_contrast(wide)
    assert set(contrasts["baseline"]) == {"f16/f16"}
    assert "f16/f16" not in set(contrasts["level"])


def test_kv_contrast_is_computed_within_a_backend(wide):
    contrasts = lb.kv_contrast(wide)
    rocm = contrasts.query("bench_backend == 'rocm' and metric == 'pp2048'")
    vulkan = contrasts.query("bench_backend == 'vulkan' and metric == 'pp2048'")
    assert rocm["ratio"].item() == pytest.approx(0.10)  # 100 -> 10
    assert vulkan["ratio"].item() == pytest.approx(0.50)  # 120 -> 60


def test_pairs_without_both_sides_are_dropped_not_imputed(wide):
    """An unmeasured half of a pair must produce no row at all."""
    incomplete = wide[
        ~((wide["bench_backend"] == "vulkan") & (wide["bench_kv"] == "f16/q4_0"))
    ]
    assert set(lb.backend_contrast(incomplete)["bench_kv"]) == {"f16/f16"}


def test_baseline_missing_from_the_whole_frame_raises(wide):
    """Silently returning nothing would look like "no effect found"."""
    with pytest.raises(ValueError, match="not present"):
        lb.kv_contrast(wide[wide["bench_kv"] != "f16/f16"])


def test_all_metrics_are_contrasted_without_being_named(wide):
    """Metrics are discovered from the frame, never hardcoded.

    The first draft referenced ``pp2048_to`` directly, so changing the prompt
    length would have broken it silently.
    """
    assert set(lb.backend_contrast(wide)["metric"]) == {"pp2048", "tg128"}


def test_unknown_factor_is_rejected(wide):
    with pytest.raises(ValueError, match="not a design factor"):
        lb.contrast(wide, "bench_nonsense")


def test_unknown_baseline_is_rejected(wide):
    with pytest.raises(ValueError, match="not present"):
        lb.kv_contrast(wide, baseline="q2_k/q2_k")


def test_summarize_reports_the_spread_not_just_a_midpoint(wide):
    summary = lb.summarize(lb.kv_contrast(wide), ["bench_backend", "metric"])
    assert {"n", "median_ratio", "min_ratio", "max_ratio"} <= set(summary.columns)
    assert (summary["min_ratio"] <= summary["median_ratio"]).all()
    assert (summary["median_ratio"] <= summary["max_ratio"]).all()


def test_as_table_limits_precision_to_what_the_data_supports(wide):
    """One repetition per test does not justify two-decimal percentages."""
    table = lb.as_table(lb.backend_contrast(wide))
    assert len(table["pct"].iloc[0].split(".")[1].rstrip("%")) == 1
    assert table["ratio"].iloc[0].endswith("x")


def test_model_contrast_isolates_the_quantisation_variant(study_wide):
    """glm vs glm-xl is the only pair in the study that varies quantisation alone.

    Both are the same base model at the same parameter count, so a paired contrast
    between them holds architecture fixed.
    """
    pair = study_wide[study_wide["bench_model"].isin(["glm", "glm-xl"])]
    contrasts = lb.contrast(pair, "bench_model", baseline="glm")
    assert set(contrasts["level"]) == {"glm-xl"}
    assert contrasts["ratio"].between(0.95, 1.05).all()


# --- Audit ------------------------------------------------------------------


def severities(runs) -> dict[str, str]:
    """Map check name -> severity."""
    return {check: severity for severity, check, _ in lb.audit_runs(runs).findings}


def test_single_repetition_is_flagged(synthetic):
    assert severities(lb.load_runs(synthetic))["replication"] == lb.CAVEAT


def test_replication_passes_when_repetitions_exist(tmp_path):
    path = tmp_path / "replicated.jsonl"
    write_jsonl(path, [make_record(samples_ts=[99.0, 100.0, 101.0], stddev_ts=1.0)])
    assert severities(lb.load_runs(path))["replication"] == lb.OK


def test_backend_specific_builds_are_flagged_as_confounded(synthetic):
    """The headline comparison must not be quoted without this caveat."""
    audit = lb.audit_runs(lb.load_runs(synthetic))
    build = next(f for f in audit.findings if f[1] == "build provenance")
    assert build[0] == lb.CAVEAT
    assert "confounded" in build[2]


def test_single_build_passes_provenance(tmp_path):
    path = tmp_path / "onebuild.jsonl"
    write_jsonl(
        path, [make_record(), make_record(bench_backend="vulkan", backends="Vulkan")]
    )
    assert severities(lb.load_runs(path))["build provenance"] == lb.OK


def test_a_varying_controlled_knob_is_a_blocker(tmp_path):
    """Changing thread count mid-study would confound every comparison."""
    path = tmp_path / "drift.jsonl"
    write_jsonl(path, [make_record(), make_record(n_threads=16, bench_depth=32768)])
    assert severities(lb.load_runs(path))["controlled fields"] == lb.BLOCKER


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
    assert severities(lb.load_runs(path))["controlled fields"] == lb.OK


def test_duplicate_config_is_a_blocker(tmp_path):
    """Two records for one cell would be averaged invisibly by the pivot."""
    path = tmp_path / "dupes.jsonl"
    write_jsonl(path, [make_record(), make_record(avg_ts=50.0, samples_ts=[50.0])])
    assert severities(lb.load_runs(path))["uniqueness"] == lb.BLOCKER


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
            make_record(
                bench_backend="vulkan", backends="Vulkan", n_prompt=0, n_gen=128
            ),
        ],
    )
    runs = lb.load_runs(path)
    absent = lb.missing_cells(runs)
    assert len(absent) == 1
    assert absent.iloc[0].to_dict() == {
        "bench_model": "test",
        "bench_backend": "vulkan",
        "bench_depth": 32768,
        "bench_kv": "f16/f16",
    }
    assert severities(runs)["design balance"] == lb.CAVEAT


def test_coverage_matrix_counts_two_tests_per_measured_cell(synthetic):
    matrix = lb.coverage_matrix(lb.load_runs(synthetic))
    assert set(matrix.to_numpy().ravel()) == {2}


def test_study_data_has_no_blockers(study_runs):
    """The real dataset must be free of blocking defects for the write-up to hold.

    This is also the gate on the deploy: a dataset with a structural defect fails
    the build instead of being published.
    """
    audit = lb.audit_runs(study_runs)
    assert audit.blockers == [], audit


def test_study_data_carries_the_expected_caveats(study_runs):
    checks = severities(study_runs)
    assert checks["replication"] == lb.CAVEAT
    assert checks["build provenance"] == lb.CAVEAT
    assert checks["design balance"] == lb.CAVEAT
    assert checks["metric completeness"] == lb.OK
    assert checks["uniqueness"] == lb.OK
