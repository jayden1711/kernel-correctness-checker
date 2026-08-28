"""
Real benchmark run: your corpus + your checker vs. allclose + autokernel_gate.

Setup (do this before running):
  1. cp my_corpus_TEMPLATE.py my_corpus.py         and fill it in
  2. python corpus_contract.py my_corpus.py         to validate it
  3. Fill in checker_adapter_template.py with your real checker calls
  4. Update the two TODO import lines below
  5. python run_benchmark.py

Falls back to the numpy-stand-in demo (demo_operators.py +
demo_placeholder_checker.py) if my_corpus.py doesn't exist yet, so you can
confirm the harness itself runs before wiring in your real code.
"""
import json
import warnings
import importlib.util

import numpy as np
from harness import run, summarize, BASE_SYSTEMS
from reporting import build_markdown

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    have_real_corpus = importlib.util.find_spec("my_corpus") is not None
    have_real_checker = importlib.util.find_spec("checker_adapter") is not None

    systems = dict(BASE_SYSTEMS)  # allclose + autokernel_gate, always included

    if have_real_corpus:
        import my_corpus as corpus_mod
        corpus = corpus_mod.CORPUS
        title = "Checker Comparison -- Real Corpus"

        # gpuemu/Propilot need the torch-native entry fields
        # (family/to_torch/torch_ref_fn/torch_mutant_fn) that only
        # my_corpus.py's real entries carry -- not available on the
        # generic demo corpus, so these are gated on have_real_corpus,
        # unlike allclose/autokernel_gate in BASE_SYSTEMS which work off
        # the abstract 5-key contract alone.
        import sota_baselines as sb
        systems["gpuemu (boundary_shape)"] = sb.gpuemu_boundary_shape_system
        systems["gpuemu (adversarial_value)"] = sb.gpuemu_adversarial_value_system
        systems["propilot"] = sb.propilot_system

        # The published-faithful AutoKernel gate (arXiv 2603.21331), added
        # after auditing baselines.py:autokernel_gate against the paper's
        # actual harness section turned up five deviations -- two of them
        # bugs that manufacture false positives on a correct reference.
        # Same family-awareness requirement as gpuemu/propilot above, so
        # same have_real_corpus gate.
        #
        # BOTH gates run: "autokernel_gate" is the old approximation,
        # "autokernel_gate (faithful)" is the corrected one. Keeping both
        # in one run makes the correction a measured delta on identical
        # inputs rather than a claim, and it is the number the paper
        # should cite.
        import autokernel_faithful as akf
        systems["autokernel_gate (faithful)"] = akf.autokernel_gate_faithful
        # RETIRED 2026-08-25 -- the "autokernel_gate (faithful, rtol=0)" system
        # used to be registered here. It measured the strict-literal reading of
        # the paper's prose, which gives an absolute tolerance and never
        # mentions a relative one. Reading AutoKernel's actual bench.py
        # (github.com/RightNow-AI/autokernel) settled it: every atol is paired
        # with an EQUAL rtol (fp32 1e-4/1e-4). rtol=0 corresponds to no real
        # AutoKernel configuration, so running it as a SYSTEM implied a
        # competing interpretation that does not exist.
        #
        # Do not re-add it. If tolerance sensitivity is wanted, it is a
        # sensitivity sweep to report as a bound, not a baseline to publish in
        # the headline table alongside the real gate.
    else:
        print("my_corpus.py not found -- falling back to the numpy demo "
              "corpus. Copy my_corpus_TEMPLATE.py -> my_corpus.py and fill "
              "it in to benchmark your real kernels.\n")
        import demo_operators as corpus_mod
        corpus = corpus_mod.CORPUS
        title = "Checker Comparison -- DEMO corpus (not your real kernels)"

    if have_real_checker:
        # TODO: once checker_adapter_template.py is filled in and renamed
        # to checker_adapter.py, this picks it up automatically. Adjust the
        # attribute names below to match what you named your functions.
        import checker_adapter as ca
        systems["your_checker (full)"] = ca.my_checker_system
        systems["your_checker (numeric only)"] = ca.my_checker_numeric_only
        systems["your_checker (algebraic only)"] = ca.my_checker_algebraic_only
        systems["your_checker (structural only)"] = ca.my_checker_structural_only
    else:
        print("checker_adapter.py not found -- falling back to the "
              "placeholder demo checker (structural layer is a no-op "
              "stub, do not cite these numbers). Fill in "
              "checker_adapter_template.py -> checker_adapter.py to "
              "benchmark your real checker.\n")
        import demo_placeholder_checker as dpc
        systems["your_checker (full) [PLACEHOLDER]"] = lambda e, m, r: (
            dpc.check(e["op"], e["ref_fn"], e["mutant_fn"] if m else e["ref_fn"],
                      e["input_fn"], r)[0], 0.0,
            dpc.check(e["op"], e["ref_fn"], e["mutant_fn"] if m else e["ref_fn"],
                      e["input_fn"], r)[1])

    results = run(systems, corpus)
    summary = summarize(results)
    md = build_markdown(summary, corpus, title=title)

    with open("results.md", "w") as f:
        f.write(md)
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # results.json is summarize()'s output ONLY -- it keeps rates,
    # missed_mutants and FP samples, and drops the per-mutant `detail` and
    # per-check records entirely (harness.py:summarize). That is why the
    # Layer-2/Layer-3 per-check ablation could not be built from any previous
    # run: the attribution was computed every time and never written down.
    # This dump is the fix. Consumed by benchmarks/analyze_check_ablation.py.
    with open("results_raw.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(md)


if __name__ == "__main__":
    main()
