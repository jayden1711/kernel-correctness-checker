"""
Out-of-sample test of the §5.3 prediction in benchmarks/NUMERICAL_THEORY.md.

THE PREDICTION, made from the tiling recurrence before any of this was run:

    `matmul:partial_k_reduct` contracts only k < K/2, so its residual is
    R_ij = sum_{k >= K/2} A_ik B_kj. With an all-positive fill that tail is
    ~50% of every entry and the baseline sees it -- which is why the mutant was
    credited on 0 of 2 recorded proposals. But the residual is identically zero
    whenever A[:, K/2:] = 0, making it CLASS E (masked at any tolerance). So a
    proposal that zeroes the second half of the contraction dimension should
    produce a `partial_k_reduct` hit on the first attempt.

This is the first prediction in this project made BEFORE the data rather than
fitted to it, so it is reported whichever way it comes out.

THREE PROPOSALS, and the third is the one that makes the result mean anything:

  P1  A[256,256] ones with A[:,128:]=0 ; B[256,256] ones
      All four masking conditions hold at once (tile-aligned, identical
      strides, fp16-exact sum, zeroed tail). Predict: ALL FOUR mutants credited.

  P2  A[100,256] ones with A[:,128:]=0 ; B[256,100] ones
      M,N = 100 breaks tile alignment; the shapes differ so the stride pairs
      differ. Predict: `partial_k_reduct` credited, `skip_boundary_tiles` and
      `swapped_strides` NOT credited.

  P3  CONTROL -- identical to P2 but WITHOUT the zeroing.
      Predict: `partial_k_reduct` NOT credited. Without this, a hit on P2 could
      just as well be an artifact of the shape, and the claim that the ZEROING
      is what does the masking would be untested.
"""
import json
import os
import sys

sys.path.insert(0, "/content")
os.environ.setdefault("CHECKER_ROOT", "/content")

from verification.adversarial_search.schemas import InputProposal, TensorDescriptor
from verification.adversarial_search.executor import execute_proposal_batch

ROOT = "/content"
OP = "matmul"
REF = os.path.join(ROOT, "TritonBench/reference/mat_mult.py")
MUTS = {
    "skip_boundary":    "TritonBench/cheating/matmult/skip_boundary_tiles.py",
    "partial_k_reduct": "TritonBench/cheating/matmult/partial_k_reduct.py",
    "swapped_strides":  "TritonBench/cheating/matmult/swapped_strides.py",
    "wrong_dtype":      "TritonBench/cheating/matmult/wrong_dtype.py",
}

ZERO_TAIL = [{"indices": "[:, 128:]", "value": 0.0}]


def desc(shape, patches=None):
    return TensorDescriptor(shape=shape, dtype="float32", fill="ones",
                            patches=patches or [])


def mk(pid, a_shape, b_shape, zero_tail):
    return InputProposal(
        proposal_id=pid, worker_id="w0", iteration=0, operator=OP,
        tensors={"A": desc(a_shape, ZERO_TAIL if zero_tail else None),
                 "B": desc(b_shape)},
        rationale="numerical-theory prediction test",
        predicted_failure_mode="partial_k_reduct",
    )


CASES = [
    ("P1  square, tile-aligned, zeroed tail",
     mk("pred-P1-square-zeroed", [256, 256], [256, 256], True),
     {"skip_boundary", "partial_k_reduct", "swapped_strides", "wrong_dtype"}),
    ("P2  M,N=100 (breaks align+strides), zeroed tail",
     mk("pred-P2-unaligned-zeroed", [100, 256], [256, 100], True),
     {"partial_k_reduct"}),
    ("P3  CONTROL: same as P2, NOT zeroed",
     mk("pred-P3-unaligned-control", [100, 256], [256, 100], False),
     {"skip_boundary", "wrong_dtype"}),
    # P4 was added AFTER P1-P3 came back, to disambiguate a confound they
    # exposed: every one of them has a CONSTANT output (all-ones fills), so an
    # out-of-bounds store writes the same value it clobbers and is invisible
    # regardless of tile alignment. Alignment and constancy were therefore
    # confounded in every proposal ever run for this operator. B varies along j
    # here, so C is non-constant while everything else is held fixed.
    ("P4  DISAMBIGUATOR: non-constant C, unaligned, zeroed tail",
     InputProposal(
         proposal_id="pred-P4-nonconstant", worker_id="w0", iteration=0,
         operator=OP,
         tensors={"A": desc([100, 256], ZERO_TAIL),
                  "B": TensorDescriptor(shape=[256, 100], dtype="float32",
                                        fill="arange", scale=0.001)},
         rationale="disambiguate tile-alignment from output-constancy",
         predicted_failure_mode="skip_boundary should become VISIBLE"),
     {"partial_k_reduct", "wrong_dtype"}),
]


def main():
    kernels = [("reference", REF)] + [(k, os.path.join(ROOT, v))
                                      for k, v in MUTS.items()]
    out = []
    for label, prop, predicted in CASES:
        res = execute_proposal_batch(
            proposal=prop, kernels=kernels, reference_src_path=REF,
            operator=OP, timeout_seconds=120)
        by = {r.kernel_id: r for r in res}
        ref = by["reference"]

        # Mirrors coordinator._evaluate_verdict exactly.
        credited = set()
        rows = []
        for name in MUTS:
            m = by[name]
            caught = not m.passed_checker
            gap = caught and m.passed_naive
            if gap:
                credited.add(name)
            rows.append({"mutant": name, "passed_checker": m.passed_checker,
                         "passed_naive": m.passed_naive, "gap": gap,
                         "error": m.error.error_type if m.error else None})
        is_hit = ref.passed_checker and bool(credited)

        print("=" * 72)
        print(f"  {label}")
        print("=" * 72)
        print(f"  reference passed checker : {ref.passed_checker}"
              + ("" if ref.passed_checker else
                 f"   FAILED: {[c['check_name'] for c in ref.check_results if not c['passed']]}"))
        for r in rows:
            print("    %-17s checker=%-5s naive=%-5s gap=%-5s %s"
                  % (r["mutant"], r["passed_checker"], r["passed_naive"],
                     r["gap"], r["error"] or ""))
        print(f"  HIT: {is_hit}")
        print(f"  predicted credited : {sorted(predicted) or '(none)'}")
        print(f"  actual    credited : {sorted(credited) or '(none)'}")
        print(f"  MATCH: {credited == predicted}", flush=True)
        out.append({"label": label, "proposal_id": prop.proposal_id,
                    "reference_passed": ref.passed_checker, "is_hit": is_hit,
                    "predicted": sorted(predicted), "credited": sorted(credited),
                    "match": credited == predicted, "mutants": rows})

    with open("/content/verification_runs/matmul_prediction_2026-08-21/"
              "prediction_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 72)
    print("  VERDICT: %d of %d cases matched the prediction"
          % (sum(1 for o in out if o["match"]), len(out)))
    print("=" * 72)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
