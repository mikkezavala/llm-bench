"""Build the static site published to GitHub Pages.

    python build_site.py --out site

Produces:

``index.html``
    The README, rendered, with a banner describing the snapshot it was built from.
``notebook.html``
    The marimo notebook, executed against the current data, with its code
    available behind a disclosure toggle next to each result.
``figures/``, ``data/``
    The figures the write-up references, and the raw log itself so anyone reading
    the page can re-derive every number on it.

Figures are re-rendered from ``data/results.jsonl`` on every build, so a page can
never show a figure that disagrees with the data it was built from.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import markdown
import matplotlib

from llmbench import design_factors, find_results, load_runs, render_figures, to_wide

REPO_URL = "https://github.com/mikkezavala/llm-bench"

STYLE = """
:root {
  --bg: #ffffff;
  --fg: #1c1e21;
  --muted: #5c6370;
  --border: #e3e6ea;
  --accent: #2f6f9f;
  --code-bg: #f5f7f9;
  --note-bg: #fff8e6;
  --note-border: #e0b34d;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171a;
    --fg: #e6e8ea;
    --muted: #9aa3ad;
    --border: #2a2f35;
    --accent: #6fb3e0;
    --code-bg: #1c2126;
    --note-bg: #2a2415;
    --note-border: #8a6d23;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1320px; margin: 0 auto; padding: 0 28px 96px; }
nav {
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
  padding: 16px 0;
  position: sticky;
  top: 0;
  background: var(--bg);
  z-index: 10;
}
nav .inner {
  max-width: 1320px;
  margin: 0 auto;
  padding: 0 28px;
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  align-items: baseline;
}
nav strong { font-size: 15px; }
nav a { color: var(--accent); text-decoration: none; font-size: 14px; }
nav a.home { color: inherit; font-size: inherit; }
nav a:hover { text-decoration: underline; }
nav .spacer { flex: 1; }
h1 { font-size: 2em; line-height: 1.25; margin: 0.4em 0 0.6em; }
h2 {
  font-size: 1.4em;
  margin-top: 2.2em;
  padding-bottom: 0.3em;
  border-bottom: 1px solid var(--border);
}
h3 { font-size: 1.12em; margin-top: 1.8em; }
a { color: var(--accent); }
p, li { overflow-wrap: break-word; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1.5em auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1.4em 0;
  font-size: 14px;
  display: block;
  overflow-x: auto;
}
th, td { border: 1px solid var(--border); padding: 7px 11px; text-align: left; }
th { background: var(--code-bg); font-weight: 600; }
code {
  background: var(--code-bg);
  padding: 0.15em 0.4em;
  border-radius: 3px;
  font-size: 0.88em;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
  overflow-x: auto;
}
pre code { background: none; padding: 0; font-size: 13px; }
blockquote {
  margin: 1.6em 0;
  padding: 12px 18px;
  background: var(--note-bg);
  border-left: 4px solid var(--note-border);
  border-radius: 0 4px 4px 0;
}
blockquote p { margin: 0.4em 0; }
.snapshot {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 18px;
  margin: 0 0 32px;
  font-size: 14px;
  color: var(--muted);
  background: var(--code-bg);
}
.snapshot dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; margin: 0; }
.snapshot dt { font-weight: 600; color: var(--fg); }
.snapshot dd { margin: 0; }
footer {
  border-top: 1px solid var(--border);
  margin-top: 56px;
  padding-top: 18px;
  font-size: 13px;
  color: var(--muted);
}
.cell { margin: 1.4em 0; }
.cell details {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 10px;
  background: var(--code-bg);
}
.cell details > summary {
  cursor: pointer;
  padding: 7px 12px;
  font-size: 12px;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  user-select: none;
}
.cell details[open] > summary { border-bottom: 1px solid var(--border); }
.cell details pre { margin: 0; border: none; border-radius: 0 0 6px 6px; }
.output { overflow-x: auto; }
.output table { font-size: 12.5px; }
.output table th { position: sticky; top: 0; }
.output iframe {
  /* Tall enough for the shallow-protocol Plotly facet (3 metric rows). Marimo's
     static export drops the iframe height attribute, so CSS has to carry it. */
  width: 100%;
  min-height: 1400px;
  height: 1400px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
  display: block;
  margin: 0.6em 0 1.4em;
}
.output img {
  margin: 1.2em auto;
}
.stderr { color: #b3402f; }
"""

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<meta name="description" content="{description}">
<style>{style}</style>
</head>
<body>
<nav><div class="inner">
  <strong><a href="index.html" class="home">{title}</a></strong>
  <span class="spacer"></span>
  <a href="index.html">Write-up</a>
  <a href="notebook.html">Notebook</a>
  <a href="data/results.jsonl">Raw data</a>
  <a href="{repo}">Source</a>
</div></nav>
<div class="wrap">
<div class="snapshot"><dl>
  <dt>Status</dt><dd>Ongoing — data collection in progress</dd>
  <dt>Snapshot</dt><dd>{n_tests} tests across {n_configs} configurations</dd>
  <dt>Recorded</dt><dd>{first} to {last}</dd>
  <dt>Built</dt><dd>{built} from <code>{commit}</code></dd>
</dl></div>
{body}
<footer>
Generated by <code>build_site.py</code> from <code>data/results.jsonl</code>.
Every figure and table on this site is re-derived from the raw log on each build.
</footer>
</div>
</body>
</html>
"""


def git_commit() -> str:
    """Short commit of the working tree, or a placeholder outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def _markdown() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )


def _plain_text(markdown_text: str) -> str:
    """Strip inline markdown so a line can be used as a meta description."""
    stripped = re.sub(r"[*_`]+", "", markdown_text)
    stripped = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", stripped)
    return " ".join(stripped.split())


def render_readme(readme: Path) -> tuple[str, str, str]:
    """Convert the README to HTML, returning (title, description, body)."""
    text = readme.read_text()
    body = _markdown().convert(text)

    lines = [line.strip() for line in text.splitlines()]
    title = next(
        (line.lstrip("# ").strip() for line in lines if line.startswith("# ")), "Study"
    )
    description = next(
        (line for line in lines if line and not line.startswith(("#", ">", "|", "!"))),
        title,
    )
    return title, _plain_text(description), body


def execute_notebook(notebook: Path) -> dict:
    """Run the notebook and return it as a Jupyter document including outputs.

    Two other export paths were rejected. ``export html-wasm`` runs the notebook
    in Pyodide, which cannot import the local ``llmbench`` package or read the
    local JSONL log. ``export html`` only embeds results from a previously saved
    editor session, so from a clean checkout — which is what CI has — it emits
    code with no computed output at all. ``export ipynb --include-outputs``
    genuinely executes the notebook, so the published page is guaranteed to show
    results derived from the data in the same commit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "executed.ipynb"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                str(notebook),
                "--include-outputs",
                "--sort",
                "topological",
                "-o",
                str(destination),
                "--force",
            ],
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        return json.loads(destination.read_text())


def _render_output(output: dict) -> str:
    """Render one Jupyter output as HTML, richest representation first."""
    if output.get("output_type") == "error":
        trace = html.escape("\n".join(output.get("traceback", [])))
        return f'<pre class="stderr">{trace}</pre>'
    if output.get("output_type") == "stream":
        text = html.escape("".join(output.get("text", [])))
        css = "stderr" if output.get("name") == "stderr" else ""
        return f'<pre class="{css}">{text}</pre>'

    data = output.get("data") or {}

    def value(key: str) -> str:
        raw = data[key]
        return raw if isinstance(raw, str) else "".join(raw)

    if "image/png" in data:
        encoded = value("image/png").replace("\n", "")
        return f'<img src="data:image/png;base64,{encoded}" alt="figure">'
    if "text/html" in data:
        return value("text/html")
    if "text/markdown" in data:
        return _markdown().convert(value("text/markdown"))
    if "text/plain" in data:
        return f"<pre>{html.escape(value('text/plain'))}</pre>"
    return ""


def render_notebook(document: dict) -> str:
    """Render an executed notebook as prose with results, code behind a toggle.

    Cells whose only job is to emit markdown are rendered as prose alone: showing
    the ``mo.md(...)`` wrapper around text that is already displayed would be
    noise.
    """
    sections: list[str] = []
    for cell in document.get("cells", []):
        source = "".join(cell.get("source", [])).strip()
        outputs = "".join(_render_output(o) for o in cell.get("outputs", []))

        if cell.get("cell_type") == "markdown":
            sections.append(f'<div class="cell">{_markdown().convert(source)}</div>')
            continue
        if not source:
            continue

        prose_only = source.startswith(("mo.md(", "mo.md (")) and "text/markdown" in {
            key for o in cell.get("outputs", []) for key in (o.get("data") or {})
        }
        parts = []
        if not prose_only:
            code = html.escape(source)
            parts.append(
                # Open by default so a plot cell is not mistaken for empty —
                # the figure sits below, but a closed toggle reads as "nothing
                # here" when skimming the page.
                "<details open><summary>code</summary>"
                f'<pre><code class="language-python">{code}</code></pre></details>'
            )
        if outputs:
            parts.append(f'<div class="output">{outputs}</div>')
        if parts:
            sections.append(f'<div class="cell">{"".join(parts)}</div>')
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("site"))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    try:
        results = find_results(__file__)
    except FileNotFoundError:
        # Every page is derived from the log, so there is nothing to publish
        # without it. Said plainly here, because the alternative is a traceback
        # in a CI log that looks like a code fault rather than a missing input.
        sys.exit(
            "build_site: data/results.jsonl not found.\n"
            "Every figure and table on the site is derived from it, so there is "
            "nothing to build without it.\n"
            "It is currently excluded from git by .git/info/exclude; commit it, "
            "or build the site locally instead of in CI."
        )
    root = results.parent.parent
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    runs = load_runs(results)
    if not args.skip_figures:
        matplotlib.use("Agg")
        render_figures(to_wide(runs), root / "figures")

    shutil.copytree(root / "figures", out / "figures")
    (out / "data").mkdir()
    shutil.copy2(results, out / "data" / results.name)
    # Pages serves the artifact verbatim, but this keeps any future
    # underscore-prefixed asset directory from being treated as Jekyll input.
    (out / ".nojekyll").write_text("")

    title, description, readme_body = render_readme(root / "README.md")
    notebook_body = render_notebook(execute_notebook(root / "analysis.py"))

    common = {
        "title": title,
        "style": STYLE,
        "repo": REPO_URL,
        "n_tests": len(runs),
        "n_configs": runs.groupby(design_factors(runs)).ngroups,
        "first": runs["test_time"].min(),
        "last": runs["test_time"].max(),
        "built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": git_commit(),
    }
    pages = {
        "index.html": (title, description, readme_body),
        "notebook.html": (
            f"Notebook — {title}",
            "The executed analysis notebook, with every table and figure "
            "regenerated from the raw benchmark log.",
            notebook_body,
        ),
    }
    for name, (page_title, page_description, page_body) in pages.items():
        (out / name).write_text(
            PAGE.format(
                page_title=page_title,
                description=html.escape(page_description, quote=True),
                body=page_body,
                **common,
            )
        )

    print(f"site built in {out}/")
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out)}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
