"""ROCm vs Vulkan on gfx1151 — the study notebook.

Run interactively::

    marimo edit analysis.py

Or execute the whole thing headlessly, which is what CI does::

    python analysis.py
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
        figure_html,
        find_results,
        kv_contrast,
        load_runs,
        missing_cells,
        model_catalog,
        plot_backend_ratio,
        plot_depth_scaling,
        plot_kv_penalty,
        plot_throughput_interactive,
        summarize,
        to_wide,
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
        figure_html,
        find_results,
        kv_contrast,
        load_runs,
        missing_cells,
        mo,
        model_catalog,
        plot_backend_ratio,
        plot_depth_scaling,
        plot_kv_penalty,
        plot_throughput_interactive,
        summarize,
        to_wide,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # KV cache and context depth on a gfx1151 iGPU

    Throughput measurements for four models people actually run locally for
    coding, on an **AMD Ryzen AI MAX+ 395 w/ Radeon 8060S** (gfx1151, 128 GB
    unified memory), across five KV cache type pairs and two context depths,
    recorded under both the ROCm and the Vulkan build of `llama.cpp`.

    **What this is.** A record of how throughput responds to *configuration* on
    one machine: KV cache type, context depth, and which model is loaded.

    **What this is not.** A comparison of ROCm against Vulkan. The two builds
    are from different `llama.cpp` commits made days apart, so a difference
    between them cannot be attributed to the backend rather than to everything
    else that changed between those commits. Results are reported per backend,
    within each backend, and are not ranked against each other.

    This notebook also does not explain *why* any measured difference occurs.
    Where a measurement has more than one possible explanation, the
    alternatives are left open.

    Read the audit section before quoting any number from here.
    """)
    return


@app.cell
def _(find_results, load_runs):
    runs = load_runs(find_results(__file__))
    return (runs,)


@app.cell
def _(runs):
    # Architecture token of `model_type`, per model label. Summaries below group
    # by it because the measurements separate along that line.
    families = dict(zip(runs["bench_model"], runs["family"], strict=True))
    return (families,)


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

    This is the part that bounds what the rest of the notebook can say. Each
    check either passes, raises a **caveat** that constrains interpretation, or
    raises a **blocker** that would invalidate a comparison.
    """)
    return


@app.cell
def _(audit_runs, runs):
    audit = audit_runs(runs)
    audit.to_frame()
    return (audit,)


@app.cell
def _(audit, mo):
    _caveats = len(audit.to_frame().query("severity == 'caveat'"))
    mo.md(f"""
    **{len(audit.blockers)} blockers, {_caveats} caveats.** The caveats above
    carry through everything below, and they are why ratios are reported as
    ranges rather than single figures:

    * **One repetition per test.** `stddev_ts` is zero by construction, so
      run-to-run variance is unmeasured. Differences of a few percent are not
      separable from noise, and no percentage here should be read to two
      decimal places.
    * **The two builds are not a controlled comparison.** ROCm and Vulkan
      records carry different `build_commit` values, from commits made days
      apart, and no record varies one while holding the other fixed. Nothing
      below reads across the two builds as a backend comparison.

    Every comparison below is computed pairwise *within* a configuration: one
    factor varies, all other factors are held fixed, and a pair with a missing
    side is dropped rather than filled in. The factor list is derived from the
    log, so sweeping a new runtime knob adds it to the configuration key rather
    than being averaged over.
    """)
    return


@app.cell
def _(coverage_matrix, runs):
    coverage_matrix(runs)
    return


@app.cell
def _(missing_cells, mo, runs):
    _absent = len(missing_cells(runs))
    mo.md(f"""
    Each cell above counts tests: `2` means both metrics were measured, `0`
    means the cell was never run. Unmeasured cells in the factorial design:
    **{_absent}**.

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
    ## Measurement 1 — KV cache type, within each backend

    KV cache type is the factor associated with the largest measured spread in
    this dataset. Note the log y-axis: on a linear axis the smallest values
    flatten against the bottom of the plot and read as missing data.

    Click a legend entry to hide or isolate a series, and hover any point for
    the exact configuration and throughput. Each panel is one backend and one
    metric; compare *within* a panel.

    Models are grouped by the architecture token of `model_type` —
    `qwen35moe` (`qwen`, `agentworld`) and `deepseek2` (`glm`, `glm-xl`) —
    because the measurements separate along that line. `f16/q8_0` and
    `f16/q4_0` are labelled **asymmetric** (`f16` keys, quantised values);
    `q8_0/q8_0` and `q4_0/q4_0` are **symmetric**. Those are descriptions of
    the K/V pair, not of a mechanism.
    """)
    return


@app.cell
def _(figure_html, mo, plot_throughput_interactive, wide):
    mo.iframe(figure_html(plot_throughput_interactive(wide)), height="1700px")
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
def _(families, kv, summarize):
    summarize(
        kv.assign(arch=kv["bench_model"].map(families)),
        ["bench_backend", "arch", "kv_class", "metric"],
    )
    return


@app.cell
def _(kv, mo):
    _worst = kv.nsmallest(1, "ratio").iloc[0]
    mo.md(f"""
    Each row of the table above is 8 pairs: 2 models × 2 depths × 2 KV pairs of
    that shape, each compared against `f16/f16` at the same model, depth and
    backend.

    The two largest effects in the dataset fall on different backends *and*
    different architectures — ROCm with `qwen35moe` under asymmetric KV, and
    Vulkan with `deepseek2` under symmetric KV. In each case the same models and
    KV pairs measured on the other backend land close to their own `f16/f16`
    baseline. The single largest is `{_worst["bench_model"]}` on
    `{_worst["bench_backend"]}` at depth {_worst["bench_depth"]}, `{_worst["level"]}`:
    {_worst["metric"]} of {_worst["to"]:.2f} t/s against {_worst["from"]:.2f} t/s
    at `f16/f16`, a ratio of {_worst["ratio"]:.4f}x.

    **Not established by this data.** Whether the ROCm asymmetric result
    reflects a different code path, a fallback, a build-specific behaviour, an
    interaction with `flash_attn=1`, or something else. `flash_attn` is 1 in
    every record, so this log contains no contrast that separates those.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Measurement 2 — the same configuration under each build

    The same configuration, read once under the ROCm build and once under the
    Vulkan build, shown as a per-cell ratio.

    **This is not a backend comparison.** The two builds are different
    `llama.cpp` commits made days apart, so these ratios contain the backend
    *and* every other change between the commits, with no way to separate them.
    It is included because which build handles which KV configuration is a
    practical fact for anyone running this hardware — not as a ranking, and no
    single number is quoted across the two.
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
def _(backend, families, summarize):
    summarize(
        backend.assign(arch=backend["bench_model"].map(families)),
        ["arch", "kv_class", "metric"],
    )
    return


@app.cell
def _(backend, mo):
    _base = backend[backend["bench_kv"] == "f16/f16"]
    _extreme = backend[backend["ratio"] > 2]
    _low = backend[backend["ratio"] < 0.75]
    mo.md(f"""
    With KV at `f16/f16` — {len(_base)} pairs, all four models and both
    depths — the two builds read within
    **{_base["ratio"].min():.3f}x – {_base["ratio"].max():.3f}x** of each other.

    Away from `f16/f16`, {len(_extreme)} pairs read above 2.00x (up to
    {_extreme["ratio"].max():.1f}x) and {len(_low)} below 0.75x (down to
    {_low["ratio"].min():.3f}x). Those are the same cells as measurement 1, seen
    from the other side: they say which build handled which KV configuration,
    not which backend is faster.

    **Not established by this data.** How any of this divides between the
    backend and the rest of what changed between the two commits. Nothing here
    varies one while holding the other fixed.

    ## Measurement 3 — context depth, 16384 → 32768

    Every pair below is the same model, backend and KV type measured at both
    depths.
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
def _(depth, mo):
    def _pct(models, metric):
        rows = depth[
            (depth["bench_kv"] == "f16/f16")
            & depth["bench_model"].isin(models)
            & (depth["metric"] == metric)
        ]
        return f"{rows['pct'].min():+.1f} % to {rows['pct'].max():+.1f} %"

    _qwen = ["qwen", "agentworld"]
    _glm = ["glm", "glm-xl"]
    mo.md(f"""
    The medians above mix architectures and KV types. Restricted to `f16/f16`,
    the two architectures separate:

    | arch | `pp2048` | `tg128` |
    | --- | --- | --- |
    | `qwen35moe` | {_pct(_qwen, "pp2048")} | {_pct(_qwen, "tg128")} |
    | `deepseek2` | {_pct(_glm, "pp2048")} | {_pct(_glm, "tg128")} |

    In both architectures `pp2048` falls by more than `tg128` over the same
    depth change.

    **Not established by this data.** The shape of the curve between and beyond
    these two points. Two depths give one slope per configuration and cannot
    distinguish linear from super-linear growth.

    ## Measurement 4 — two quantisations of the same model

    `glm` (`Q4_K_M`) and `glm-xl` (`UD-Q4_K_XL`) report the same `model_type`
    and the same parameter count, and differ in quantisation and file size.
    They are the only pair in the log where a contrast varies quantisation with
    model, backend, depth and KV type all held fixed.
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
def _(mo, variant):
    _above = int((variant["ratio"] > 1).sum())
    mo.md(f"""
    Across all **{len(variant)} paired configurations**, `glm-xl / glm` falls
    between **{variant["ratio"].min():.3f}x and {variant["ratio"].max():.3f}x**,
    with medians of
    {variant[variant["metric"] == "pp2048"]["ratio"].median():.3f}x (`pp2048`)
    and {variant[variant["metric"] == "tg128"]["ratio"].median():.3f}x
    (`tg128`). {_above} of {len(variant)} pairs are above 1.000x.

    **Not established by this data.** Whether that spread or that direction is
    distinguishable from run-to-run variance, which is unmeasured at one
    repetition per test, or from a systematic run-order or thermal offset —
    runs were sequential and run order is not randomised. Nothing here speaks
    to output quality, the other axis on which these two files differ.

    `agentworld` is also a `UD` build, but it is a different fine-tune from
    `qwen`, so a contrast between them varies the model and the quantisation
    together.

    ## Limitations

    * **One repetition per test.** `stddev_ts` is 0 by construction.
      Differences of a few percent — the `f16/f16` reading between the two
      builds, the quantisation-variant spread — are not separable from noise.
    * **The two builds are not a controlled comparison.** No record varies
      backend and `build_commit` independently.
    * **`flash_attn` is 1 everywhere.** Nothing here shows how the
      measurements change without it.
    * **`use_mmap` and the other runtime knobs are constant.** Not yet swept.
    * **Throughput only.** No perplexity or task accuracy.
    * **Two depths.** 16384 and 32768 only.
    * **5 of 9 K/V type pairs.**
    * **One machine, sequential runs.** Run order is not randomised and
      thermal state is not recorded.
    * **Two architectures, two models each.**

    ## Open questions

    Unresolved by the current data, with the contrast each would need:

    1. Does the ROCm `qwen35moe` asymmetric-KV result persist with
       `flash_attn=0`? The factor is constant at 1 in every record.
    2. What do the two backends read when both are built from the same commit?
       That is the contrast this log cannot supply.
    3. What does `use_mmap` do here, on unified memory with models in the
       16–21 GiB range?
    4. Is the `glm-xl` / `glm` difference reproducible under repeated,
       interleaved runs?
    5. What is the shape of the depth curve? More depths would distinguish a
       slope from a curve.
    6. Does the `deepseek2` / `qwen35moe` split hold with more models per
       architecture, or is it a property of these four files?
    7. Do the KV cache quantisations that cost little throughput cost output
       quality? That needs a metric this study does not collect.
    """)
    return


if __name__ == "__main__":
    app.run()
