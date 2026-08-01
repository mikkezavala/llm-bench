"""Tests for JSONL loading, metric derivation and pivoting."""

from __future__ import annotations

import pytest
from tests.conftest import make_record, write_jsonl

from llmbench import load_runs, metric_columns, model_catalog, to_wide
from llmbench.loading import _quant_label, find_results, kv_order


def test_test_type_derived_from_prompt_and_gen_counts(synthetic):
    runs = load_runs(synthetic)
    assert set(runs["test_type"]) == {"pp2048", "tg128"}


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "gappy.jsonl"
    write_jsonl(path, [make_record()])
    path.write_text(path.read_text() + "\n\n")
    assert len(load_runs(path)) == 1


def test_malformed_json_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": 1}\nnot json\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_runs(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no records"):
        load_runs(path)


def test_record_with_no_test_is_rejected_not_dropped(tmp_path):
    """A record llama-bench should never emit must fail loudly.

    The earlier notebook silently skipped these, so a truncated log would have
    produced a smaller table with no indication anything was missing.
    """
    path = tmp_path / "notest.jsonl"
    write_jsonl(path, [make_record(n_prompt=0, n_gen=0)])
    with pytest.raises(ValueError, match="neither n_prompt nor n_gen"):
        load_runs(path)


def test_combined_prompt_and_gen_test_is_labelled_not_dropped(tmp_path):
    path = tmp_path / "pg.jsonl"
    write_jsonl(path, [make_record(n_prompt=512, n_gen=128)])
    assert load_runs(path)["test_type"].tolist() == ["pg512+128"]


def test_pivot_puts_prefill_and_decode_on_one_row(synthetic):
    wide = to_wide(load_runs(synthetic))
    assert len(wide) == 4  # 2 backends x 2 KV types
    assert metric_columns(wide) == ["pp2048", "tg128"]
    assert not wide[["pp2048", "tg128"]].isna().to_numpy().any()


def test_kv_order_is_interpretable_not_alphabetical(study_runs):
    """Baseline first, then asymmetric, then symmetric.

    Alphabetical order would put ``q4_0/q4_0`` before ``q8_0/q8_0`` and place
    ``f16/q8_0`` adjacent to ``q4_0/q4_0``, a step that changes both caches at
    once and cannot be read as a single transition.
    """
    assert kv_order(study_runs) == [
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
    assert kv_order(load_runs(path)) == ["f16/f16", "q5_1/q5_1"]


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
    assert _quant_label(f"/models/{filename}") == expected


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
                model_filename="/models/B-Q4_K_M.gguf", bench_kv="q8_0/q8_0",
                type_k="q8_0", type_v="q8_0",
            ),
        ],
    )
    with pytest.raises(ValueError, match="multiple model_filename"):
        model_catalog(load_runs(path))


def test_find_results_walks_up_from_a_nested_path():
    assert find_results(__file__).name == "results.jsonl"


def test_find_results_reports_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="at or above"):
        find_results(tmp_path)
