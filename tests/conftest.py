"""Shared fixtures: a small synthetic log, plus the real study data."""

from __future__ import annotations

import json

import pytest

from llmbench import find_results, load_runs, to_wide

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
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
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


@pytest.fixture(scope="session")
def study_runs():
    """The real ``data/results.jsonl``, as loaded for the write-up."""
    return load_runs(find_results(__file__))


@pytest.fixture(scope="session")
def study_wide(study_runs):
    return to_wide(study_runs)
