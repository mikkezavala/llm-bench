"""ROCm vs Vulkan on gfx1151 — the study notebook.

Run interactively::

    marimo edit notebooks/analysis.py

Or execute the whole thing headlessly, which is what CI does::

    python notebooks/analysis.py
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from llmbench import (
        as_table,
        audit_runs,
        backend_contrast,
        contrast,
        coverage_matrix,
        depth_contrast,
        find_results,
        kv_contrast,
        load_runs,
        model_catalog,
        summarize,
        to_wide,
    )
    from llmbench.plots import (
        plot_backend_ratio,
        plot_depth_scaling,
        plot_kv_penalty,
        plot_kv_sensitivity,
        use_study_style,
    )

    use_study_style()
    pd.set_option("display.width", 200)
    return (
        as_table,
        audit_runs,
        backend_contrast,
        contrast,
        coverage_matrix,
        depth_contrast,
        find_results,
        kv_contrast,
        load_runs,
        mo,
        model_catalog,
        plot_backend_ratio,
        plot_depth_scaling,
        plot_kv_penalty,
        plot_kv_sensitivity,
        summarize,
        to_wide,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # KV cache types, ROCm and Vulkan on a gfx1151 iGPU

    A study of `llama.cpp` throughput on an **AMD Ryzen AI MAX+ 395 w/ Radeon
    8060S** (gfx1151, 128 GB unified memory), comparing the **ROCm** and
    **Vulkan** backends across KV cache quantisation types, two context
    depths, and four MoE models — including two Unsloth Dynamic (`UD`)
    quantisations.

    The question I started with was "which backend is faster". The data
    answered a more useful question instead: **which KV cache configurations
    each backend handles well**, because that choice moves throughput by far
    more than the backend does.

    Read the audit section before quoting any number from here.
    """)
    return


@app.cell
def _(find_results, load_runs):
    runs = load_runs(find_results(__file__))
    return (runs,)


@app.cell
def _(mo, runs):
    mo.md(f"""
    ## What was measured

    `data/results.jsonl` is the raw `llama-bench --output jsonl` log, appended
    to as runs complete and never edited. This snapshot holds
    **{len(runs)} tests** recorded between `{runs["test_time"].min()}` and
    `{runs["test_time"].max()}`.

    Each record is one test. Two metrics are measured at each context depth:

    | metric | meaning |
    | --- | --- |
    | `pp2048` | prefill: process a 2048-token prompt at cache depth `d` |
    | `tg128` | decode: generate 128 tokens from the same depth |

    The four factors that vary are the harness-injected fields
    `bench_model`, `bench_backend`, `bench_depth` and `bench_kv`. Everything
    else — batch size, thread count, flash attention, layer offload — is held
    fixed, and the audit below verifies that claim rather than trusting it.
    """)
    return


@app.cell
def _(model_catalog, runs):
    model_catalog(runs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Audit

    This is the part that decides what the rest of the notebook is allowed to
    say. Each check either passes, raises a **caveat** that constrains
    interpretation, or raises a **blocker** that would invalidate the
    analysis.
    """)
    return


@app.cell
def _(audit_runs, runs):
    audit = audit_runs(runs)
    audit.to_frame()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Three caveats carry through everything below, and they are the reason this
    notebook reports ratios with ranges instead of single headline numbers:

    1. **One repetition per test.** `stddev_ts` is zero by construction, so
       run-to-run variance is unknown. Differences of a few percent are not
       distinguishable from noise, and no percentage here should be read to
       two decimal places.
    2. **Backend is confounded with build.** The ROCm runs come from one
       `llama.cpp` commit and the Vulkan runs from another. Any ROCm/Vulkan
       difference is a difference between *these two builds on this machine*,
       not a property of the backends in general.
    3. **The design is not fully crossed.** Some model/depth/KV cells were
       never run. An unpaired mean over the whole table would silently compare
       different sets of conditions, so every comparison below is computed
       pairwise *within* a configuration, and pairs with a missing side are
       dropped rather than filled in.
    """)
    return


@app.cell
def _(coverage_matrix, runs):
    coverage_matrix(runs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Each cell above counts tests: `2` means both prefill and decode were
    measured, `0` means the cell was never run. The gaps are concentrated in
    the GLM models at 32k depth.

    ## The measurements

    One row per configuration, both metrics side by side.
    """)
    return


@app.cell
def _(runs, to_wide):
    wide = to_wide(runs)
    wide
    return (wide,)


@app.cell
def _(mo, wide):
    model_filter = mo.ui.multiselect(
        options=sorted(wide["bench_model"].unique()),
        value=sorted(wide["bench_model"].unique()),
        label="models",
    )
    kv_filter = mo.ui.multiselect(
        options=list(dict.fromkeys(wide["bench_kv"])),
        value=list(dict.fromkeys(wide["bench_kv"])),
        label="KV cache types",
    )
    mo.hstack([model_filter, kv_filter], justify="start", gap=2)
    return kv_filter, model_filter


@app.cell
def _(kv_filter, model_filter, wide):
    view = wide[
        wide["bench_model"].isin(model_filter.value)
        & wide["bench_kv"].isin(kv_filter.value)
    ]
    view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Observation 1 — each backend has its own KV cache cliff

    The KV cache type is the single largest lever in this dataset, and the two
    backends fail on *opposite* configurations. Note the log y-axis: on a
    linear axis the ROCm collapse flattens against the bottom of the plot and
    reads as missing data.
    """)
    return


@app.cell
def _(plot_kv_sensitivity, wide):
    plot_kv_sensitivity(wide)
    return


@app.cell
def _(kv_contrast, wide):
    kv = kv_contrast(wide)
    return (kv,)


@app.cell
def _(kv, plot_kv_penalty):
    plot_kv_penalty(kv)
    return


@app.cell
def _(kv, summarize):
    summarize(kv, ["bench_backend", "metric", "level"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Reading those two views together:

    * **ROCm, asymmetric KV (`f16` keys with a quantised value cache).** The
      `qwen35moe` models lose roughly **98 % of prefill throughput** and about
      **80 % of decode** — around 690 t/s down to 14 t/s at 16k. This is not a
      gradual cost, it is a cliff, and its size is the signature of falling
      off an optimised attention path rather than of arithmetic on smaller
      types. The `deepseek2` models are untouched by it.
    * **ROCm, symmetric KV (both caches quantised).** Essentially free for
      prefill (within 1 % of `f16/f16`) and a modest 6–17 % cost for decode.
    * **Vulkan, asymmetric KV.** Also close to free — a few percent at most.
    * **Vulkan, symmetric KV.** Fine for `qwen35moe`, but the `deepseek2`
      models lose about **half their prefill** and a third of decode.

    So "should I quantise the KV cache" has no backend-independent answer
    here. On this machine the safe configuration is `f16/f16` for both
    backends; if the KV cache must shrink, ROCm tolerates symmetric
    quantisation and Vulkan tolerates asymmetric — exactly the reverse of one
    another.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Observation 2 — the backend gap is small next to the cliffs

    With the KV cache held at a configuration both backends handle, Vulkan is
    modestly ahead. Where a cliff is in play, the "ratio" is really measuring
    the cliff, not the backend — which is why the extreme cells are called out
    separately rather than folded into an average.
    """)
    return


@app.cell
def _(backend_contrast, wide):
    backend = backend_contrast(wide)
    return (backend,)


@app.cell
def _(backend, plot_backend_ratio):
    plot_backend_ratio(backend)
    return


@app.cell
def _(backend, summarize):
    summarize(backend, ["metric", "kv_class"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    On the `f16/f16` baseline, Vulkan leads by roughly **9–19 %** on both
    prefill and decode across all four models. That is a real but ordinary
    gap, and it is confounded with the build difference noted in the audit.

    The `51x`–`90x` cells are the ROCm asymmetric-KV collapse seen from the
    other side, and the `0.56x` cells are the Vulkan symmetric-KV regression
    on `deepseek2`. Averaging those into a single "Vulkan is N times faster"
    figure would be meaningless: the median prefill ratio over the whole table
    is inflated by a bug in one backend and depressed by a different bug in
    the other.

    ## Observation 3 — doubling context costs prefill more than decode

    Every pair below is the same model, backend and KV type measured at 16k
    and then at 32k.
    """)
    return


@app.cell
def _(depth_contrast, wide):
    depth = depth_contrast(wide)
    return (depth,)


@app.cell
def _(depth, plot_depth_scaling):
    plot_depth_scaling(depth)
    return


@app.cell
def _(depth, summarize):
    summarize(depth, ["bench_backend", "metric"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Going from 16k to 32k depth costs about **18–24 % of prefill** and only
    **4–9 % of decode** in the healthy configurations — consistent with
    attention over a longer cache dominating a batched prefill more than it
    dominates single-token decode.

    Two details stand out. The ROCm medians look far worse (about −43 %
    prefill) because the ROCm column includes the collapsed asymmetric-KV
    configurations, which degrade *further* with depth: the cliff gets steeper
    as the cache grows, roughly −55 % on top of an already 50x loss. And
    `glm` on ROCm drops about 43 % of prefill from 16k to 32k even on
    `f16/f16`, a much steeper depth curve than the `qwen35moe` models show on
    the same backend.

    ## Observation 4 — the Unsloth Dynamic variants cost nothing measurable

    `glm` (`Q4_K_M`) and `glm-xl` (`UD-Q4_K_XL`) are the same base model at the
    same parameter count, so pairing them isolates the quantisation mix.
    """)
    return


@app.cell
def _(as_table, contrast, wide):
    glm_pair = wide[wide["bench_model"].isin(["glm", "glm-xl"])]
    variant = contrast(glm_pair, "bench_model", baseline="glm")
    as_table(variant)
    return (variant,)


@app.cell
def _(summarize, variant):
    summarize(variant, ["metric"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Across all ten paired configurations the `UD-Q4_K_XL` build lands within
    about **1 %** of `Q4_K_M` — and interestingly, on prefill it is faster in
    10 of 10 pairs. A ~1 % gap is well inside what a single repetition can
    resolve, but a consistent *sign* across ten independent pairs is harder to
    dismiss as noise, and it points the same way as the file size: `glm-xl` is
    the smaller model on disk (16.31 GiB vs 17.05 GiB) at identical parameter
    count.

    The cautious reading is that the dynamic quantisation is
    throughput-neutral to marginally favourable here, and that the reason to
    choose it would be memory footprint or output quality — neither of which
    this study measures. Distinguishing a genuine 1 % edge from a systematic
    run-order or thermal offset needs interleaved repetitions.

    The other `UD` model, `agentworld`, cannot be used this way: it is a
    different fine-tune from `qwen`, so a comparison between them mixes the
    model and the quantisation and cannot separate the two.

    ## Limitations

    * **One repetition per test.** The largest effects here are 50x and would
      survive any plausible variance, but the 9–19 % backend gap and the 1–2 %
      variant difference genuinely need replication before they mean anything.
    * **Backend and build are not separated.** Rebuilding both backends from
      one commit is the single change that would most improve this study.
    * **Throughput only.** No perplexity, no task accuracy. A KV cache
      quantisation that is free in tokens/second is not necessarily free in
      output quality, and nothing here speaks to that.
    * **Two depths.** 16k and 32k give a slope but not a curve; the shape of
      depth scaling past 32k is unmeasured.
    * **One machine, one thermal envelope.** Runs were sequential on a single
      laptop-class iGPU; sustained-load throttling is not controlled for.

    ## What I would measure next

    1. Rebuild both backends from the same `llama.cpp` commit and re-run, with
       at least three repetitions per test.
    2. Fill the missing GLM cells at 32k so the design is fully crossed.
    3. Isolate the ROCm asymmetric-KV cliff: does it survive with flash
       attention disabled? That would distinguish an unsupported fused
       attention path from a general dequantisation cost.
    4. Add a depth sweep (4k, 8k, 16k, 32k, 64k) to get the shape of the
       prefill curve rather than two points on it.
    """)
    return


if __name__ == "__main__":
    app.run()
