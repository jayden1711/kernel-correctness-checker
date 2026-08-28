"""
Re-render results.md and results.json from an existing results_raw.json.

WHY THIS EXISTS
---------------
Changing how a report is *formatted* should not cost a GPU session. Before this
script, the only way to see a reporting.py change take effect was to re-run
run_benchmark.py -- which needs a live corpus, a real GPU, and ~10 minutes on
Colab -- even though every number in the report was already sitting in
results_raw.json from the previous run.

That is what happened on 2026-08-20: harness.summarize() gained p50/p90/p99
latency percentiles and reporting.py gained the columns to show them, but the
generated results.md still carried the old mean-only table, because regenerating
it appeared to require another benchmark run. It did not.

HOW IT AVOIDS NEEDING A CORPUS
------------------------------
build_markdown(summary, corpus, ...) takes the corpus but uses it for exactly
two things (reporting.py:10-12): `len(corpus)`, and the sorted set of
`e["op"]`. Both are reconstructible from any system's `mutant_results` in
results_raw.json -- one record per mutant, each carrying its "op". So this
script rebuilds a minimal corpus-shaped list of `{"op": ...}` dicts and passes
that. Verified on the 2026-08-20 run: reconstructed 40 mutants across 29
operators, matching the original header exactly.

harness.py and baselines.py import numpy at module scope, and reporting.py uses
np.isnan, but summarize() itself needs only np.mean -- the percentiles are pure
Python (harness._percentile). A minimal numpy stub therefore suffices, which is
what lets this run without the venv (see SESSION_HANDOFF.md §0 on why importing
the real torch/numpy stack off Google Drive File Stream is impractical here).
The stub is deliberately tiny and exact: `mean` is the arithmetic mean and
`isnan` is `x != x`, which is exact for floats rather than an approximation.

WHAT IT IS NOT FOR
------------------
This re-renders existing measurements. It does **not** re-measure anything --
every catch rate, false-positive rate and latency it writes comes verbatim from
results_raw.json. If you changed the checker, a check, or the corpus, you need a
real run; this script would faithfully re-print the *old* results. Use it only
when the thing that changed is how results are summarised or displayed.

SAFETY PROPERTY
---------------
The 2026-08-20 promotion was verified additive: regenerating produced a
byte-identical results.md apart from the new latency columns and their
explanatory note, and a results.json with 4 keys added, 0 removed, and 0
pre-existing values changed across all 11 systems. If a future change to
summarize() or build_markdown() is meant to be additive, diffing before/after
with this script is the cheap way to prove it.

USAGE
-----
    python3 benchmarks/regenerate_report.py [path/to/results_raw.json]

Defaults to benchmarks/autokernel/files/results_raw.json and writes results.md
and results.json next to it. Plain python3 -- no venv, no torch, no GPU.
"""
import importlib.util
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(HERE, "autokernel", "files")
DEFAULT_RAW = os.path.join(FILES_DIR, "results_raw.json")
TITLE = "Checker Comparison -- Real Corpus"


def _install_numpy_stub():
    """Minimal numpy so harness/reporting import without the real stack.

    summarize() uses np.mean; reporting.fmt_pct uses np.isnan; baselines.py
    annotates with np.ndarray at import time. Nothing else is touched, and the
    percentile maths lives in harness._percentile (pure Python) precisely so
    this stub never has to approximate a numpy algorithm.
    """
    if "numpy" in sys.modules:
        return
    np = types.ModuleType("numpy")
    np.mean = lambda v: sum(v) / len(v)
    np.isnan = lambda x: x != x          # exact for floats, not an approximation
    np.ndarray = type("ndarray", (), {})
    sys.modules["numpy"] = np


def _load(module_name):
    path = os.path.join(FILES_DIR, module_name + ".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_corpus(raw):
    """Rebuild the minimal corpus shape build_markdown needs.

    One entry per mutant carrying only "op" -- see module docstring for why
    that is sufficient. Raises if the file has no systems, rather than silently
    writing a report describing an empty corpus.
    """
    if not raw:
        raise ValueError("results_raw.json contains no systems")
    any_system = next(iter(raw.values()))
    return [{"op": rec["op"]} for rec in any_system["mutant_results"]]


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW
    out_dir = os.path.dirname(os.path.abspath(raw_path))

    with open(raw_path) as f:
        raw = json.load(f)

    _install_numpy_stub()
    sys.path.insert(0, FILES_DIR)
    harness = _load("harness")
    reporting = _load("reporting")

    corpus = reconstruct_corpus(raw)
    n_ops = len({e["op"] for e in corpus})
    print(f"Reconstructed corpus: {len(corpus)} mutants across {n_ops} operators")

    summary = harness.summarize(raw)
    md = reporting.build_markdown(summary, corpus, title=TITLE)

    md_path = os.path.join(out_dir, "results.md")
    json_path = os.path.join(out_dir, "results.json")
    with open(md_path, "w") as f:
        f.write(md)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print("Re-rendered from existing measurements -- nothing was re-measured.")


if __name__ == "__main__":
    main()
