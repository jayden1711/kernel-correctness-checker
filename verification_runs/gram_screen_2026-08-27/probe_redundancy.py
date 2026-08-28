"""
Per-check / per-variant timing + redundancy probe for the numeric layer.

Runs ONLY `your_checker (full)` over the corpus and dumps every check record
with its duration_ms, subchecks and input_stats. One arm per invocation,
selected by environment (see driver.sh).

Deliberately NOT run through run_benchmark.py: that would run all 11 systems
and its published latency numbers must never be produced under
KCC_CHECK_TIMING=1, which serialises CUDA (see verification/checker.py).
"""
import json, os, sys, time
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/benchmarks/autokernel/files")

import numpy as np
import my_corpus
import checker_adapter as ca

ARM = os.environ.get("KCC_ARM", "unnamed")
N_REF = int(os.environ.get("KCC_N_REF", "5"))

out = {"arm": ARM,
       "env": {k: v for k, v in os.environ.items() if k.startswith("KCC_")},
       "entries": []}

rng = np.random.default_rng(0)
corpus = my_corpus.CORPUS

for entry in corpus:
    # Warm exactly as harness._warm does, with the same RNG snapshot/restore,
    # so this probe's inputs match the benchmark's and the arms match each
    # other. Without the restore every later entry would see shifted draws.
    state = {"np": rng.bit_generator.state}
    import torch
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        ca.my_checker_system(entry, True, rng)
        ca.my_checker_system(entry, False, rng)
    except Exception:
        pass
    rng.bit_generator.state = state["np"]
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

    rec = {"op": entry["op"], "mutant": entry["mutant_name"]}
    passed, dt, detail, records = ca.my_checker_system(entry, True, rng)
    rec["mutant"] = {"name": entry["mutant_name"], "caught": (not passed),
                     "dt_ms": 1000 * dt, "detail": detail, "records": records}
    refs = []
    for _ in range(N_REF):
        rp, rdt, rdetail, rrecords = ca.my_checker_system(entry, False, rng)
        refs.append({"false_positive": (not rp), "dt_ms": 1000 * rdt,
                     "detail": rdetail, "records": rrecords})
    rec["refs"] = refs
    out["entries"].append(rec)
    print(f"[{ARM}] {entry['op']}/{entry['mutant_name']} caught={not passed} "
          f"{1000*dt:.1f}ms", flush=True)

n_caught = sum(e["mutant"]["caught"] for e in out["entries"])
n_fp = sum(r["false_positive"] for e in out["entries"] for r in e["refs"])
n_ref = sum(len(e["refs"]) for e in out["entries"])
out["summary"] = {"n_mutants": len(out["entries"]), "n_caught": n_caught,
                  "catch_rate": n_caught / len(out["entries"]),
                  "n_ref": n_ref, "n_fp": n_fp, "fp_rate": n_fp / n_ref if n_ref else 0.0}
print(f"[{ARM}] SUMMARY catch {n_caught}/{len(out['entries'])} "
      f"fp {n_fp}/{n_ref}", flush=True)
json.dump(out, open(f"/content/probe/{ARM}.json", "w"), default=str)
