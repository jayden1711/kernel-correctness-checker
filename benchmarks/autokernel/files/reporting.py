import numpy as np


def fmt_pct(x):
    return f"{100*x:.0f}%" if not np.isnan(x) else "n/a"


def build_markdown(summary, corpus, title="Checker Comparison Results"):
    lines = [f"# {title}\n"]
    ops = sorted(set(e["op"] for e in corpus))
    lines.append(f"Corpus: {len(corpus)} mutants across {len(ops)} operators "
                  f"({', '.join(ops)}).\n")

    lines.append("## Headline comparison\n")
    lines.append("| System | Catch rate | False positive rate | p50 (ms) | p90 (ms) | p99 (ms) | Mean latency (ms/check) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, s in summary.items():
        # .get() so a summary loaded from a pre-2026-08-20 run (which has no
        # percentile keys) still renders instead of raising.
        lines.append(f"| {name} | {fmt_pct(s['catch_rate'])} | "
                      f"{fmt_pct(s['false_positive_rate'])} | "
                      f"{s.get('p50_latency_ms', float('nan')):.2f} | "
                      f"{s.get('p90_latency_ms', float('nan')):.2f} | "
                      f"{s.get('p99_latency_ms', float('nan')):.2f} | "
                      f"{s['mean_latency_ms']:.4f} |")
    lines.append("")
    lines.append("> **Steady-state latency (kernel cache warmed).** harness.run() warms "
                 "each system per corpus entry before timing, so these numbers are "
                 "amortised cost rather than first-call cost. Without warming, 84% of "
                 "`your_checker (full)`'s measured time was Triton JIT compilation, "
                 "charged to whichever system the dict happened to run first. All "
                 "systems now use one timer scope: everything from input generation to "
                 "verdict. **Read p50, not the mean** — percentiles use "
                 "`numpy.percentile`'s default linear interpolation, so p50 equals the "
                 "median; the mean is retained only for comparability with pre-warming "
                 "runs. Cold-cache figures for the same run are in results_raw_cold.json.")
    lines.append("")

    lines.append("## Per-operator catch rate\n")
    header = "| System | " + " | ".join(ops) + " |"
    sep = "|---|" + "---|" * len(ops)
    lines.append(header)
    lines.append(sep)
    for name, s in summary.items():
        row = [fmt_pct(s["per_op_catch_rate"].get(op, float("nan"))) for op in ops]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Mutants each system missed\n")
    for name, s in summary.items():
        lines.append(f"**{name}**: " + (", ".join(s["missed_mutants"]) if s["missed_mutants"] else "none"))
    lines.append("")

    any_fp = any(s.get("per_op_false_positive_rate") for s in summary.values())
    if any_fp:
        lines.append("## Per-operator false-positive rate (reference flagged as wrong; 0% cells omitted)\n")
        for name, s in summary.items():
            per_op_fp = s.get("per_op_false_positive_rate") or {}
            if not per_op_fp:
                continue
            lines.append(f"**{name}**: " + ", ".join(
                f"{op}={fmt_pct(rate)}" for op, rate in sorted(per_op_fp.items())))
        lines.append("")

        lines.append("## False-positive example details (up to 3 per operator)\n")
        for name, s in summary.items():
            samples = s.get("false_positive_detail_samples") or {}
            if not samples:
                continue
            lines.append(f"**{name}**:")
            for op, details in sorted(samples.items()):
                for d in details:
                    lines.append(f"  - {op}: {d}")
        lines.append("")

    return "\n".join(lines)
