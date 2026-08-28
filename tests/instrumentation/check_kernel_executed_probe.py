#!/usr/bin/env python3
"""
Guards the check_kernel_executed probe-ladder fix (CHECK_ABLATION_FINDINGS.md
§3.0).

WHAT WAS WRONG. The old check used one probe, `x + randn_like(x)*0.1 + 1.0`,
and thereby asserted "different input => different output". That is false for
any non-injective operator, so CORRECT reference kernels failed a Layer-1
check: 20 of 80 proposals in the 2026-08-20 causal_flash_attention run, plus 30
more across earlier history.

WHAT THIS SCRIPT DOES. It replays the REAL recorded input descriptors -- the
exact tensors those runs used -- through the REAL check_kernel_executed in the
repo, and requires:

    the OLD check FAILS all 25 and PASSES all 51  (control 0: the harness can
                                                   actually exhibit the bug)
    the 25 recorded false positives now PASS
    the 51 recorded passes still PASS          (control 1: nothing else moved)
    a genuine ghost kernel is still CAUGHT     (control 2: no FP-for-FN trade)
    each rung rescues what it claims to        (control 3: measured in
                                                ISOLATION, see below)

CONTROL 0 IS NOT OPTIONAL. "25/25 now pass" is unfalsifiable on its own -- a
harness that could not reproduce the false positive in the first place would
report exactly the same thing. The first draft of this script had no control 0
and its control 3 used leave-one-OUT, which measures nothing in a disjunction
whose rungs overlap: every rung reported delta 0 while genuinely rescuing
between 0 and 20 cases. Rungs are therefore measured by leave-one-IN.

WHAT "PASSES" MEANS FOR THESE 25 CASES, precisely. The search executes the
reference kernel as its own candidate (coordinator.py:283-284 passes
reference_src_path for BOTH), so on this replay candidate IS reference. Two
different mechanisms can clear a case and the script reports which:

  - a ladder rung moves the output  -> the kernel is demonstrably input-
    dependent, and the old check was simply wrong to flag it;
  - rung E finds the reference equally still -> "not evaluable", i.e. the
    input is degenerate for this operator and identical outputs are correct.

Rung E is VACUOUS when candidate is reference, and that is expected: you
cannot detect "the reference ignores its input" by comparing the reference to
itself. It is non-vacuous on the corpus path, where the candidate is a Triton
kernel and the reference is PyTorch -- covered by the full benchmark run, not
here.

REQUIRES TORCH. Unlike the other check_*.py scripts in this directory, this one
cannot use a shape-recording stub: the defect is numerical, not structural --
whether two float32 outputs are bitwise equal is the entire question. Run it on
a machine with a real torch (the Colab VM's CPU is fine; no GPU needed).

CPU STAND-IN KERNELS, AND WHY THAT IS HONEST HERE. The corpus kernels are
Triton and need a GPU, so this script drives the check with PyTorch reference
formulas instead. That is sufficient for what it guards -- the false positive
is a property of the OPERATOR'S MATHEMATICS (shift-invariance, saturation,
weights independent of Q), not of Triton codegen. The real-Triton-kernel run is
a separate GPU step and is the number that gets published; this script exists
so that a logic regression is caught in seconds without one.

Exit 0 = pass. Standalone `python3`, not pytest -- named check_* so pytest.ini's
`python_files = test_*.py` does not collect it. Do not rename.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

from verification.layer1_structural import runtime_guards  # noqa: E402
from verification.layer1_structural.runtime_guards import (  # noqa: E402
    check_kernel_executed,
)
from verification.adversarial_search.materializer import (  # noqa: E402
    _materialize_one,
)
from verification.adversarial_search.schemas import TensorDescriptor  # noqa: E402

# §5 instance 4: a ported test once passed while validating a stale copy under
# /tmp. If this path is ever not the repo's own file, the run is worthless.
print(f"module under test: {runtime_guards.__file__}")
if not runtime_guards.__file__.startswith(_REPO):
    print(f"FATAL: runtime_guards resolved outside the repo ({_REPO})")
    sys.exit(1)

FIXTURE = os.path.join(_HERE, "kernel_executed_fp_cases.json")

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


# --------------------------------------------------------------------------
# Reference formulas -- PyTorch stand-ins for the Triton corpus kernels.
# --------------------------------------------------------------------------

def _attention(Q, K, V, causal):
    D = Q.shape[-1]
    S = (Q.float() @ K.float().transpose(-2, -1)) / (D ** 0.5)
    if causal:
        n = S.shape[-1]
        mask = torch.tril(torch.ones(S.shape[-2], n, device=S.device, dtype=torch.bool))
        S = S.masked_fill(~mask, float("-inf"))
    return (torch.softmax(S, dim=-1) @ V.float()).to(Q.dtype)


REFERENCES = {
    "causal_flash_attention": lambda Q, K, V: _attention(Q, K, V, True),
    "flash_attention":        lambda Q, K, V: _attention(Q, K, V, False),
    "softmax":                lambda x: torch.softmax(x.float(), dim=-1).to(x.dtype),
    "argmax":                 lambda x: torch.argmax(x, dim=-1),
}

TENSOR_KEYS = {
    "causal_flash_attention": ("Q", "K", "V"),
    "flash_attention":        ("Q", "K", "V"),
    "softmax":                ("x",),
    "argmax":                 ("x",),
}


class _Spec:
    """Minimal KernelSpec surface used by check_kernel_executed's rung D.

    Deliberately NOT a real spec import: rung D touches exactly
    run_candidate / run_reference, and modelling only that makes it obvious
    which surface the check actually depends on.
    """

    def __init__(self, name):
        self.name = name
        self.multi = len(TENSOR_KEYS[name]) > 1

    def run_candidate(self, fn, inputs):
        return fn(*inputs) if isinstance(inputs, tuple) else fn(inputs)

    run_reference = run_candidate


def load_cases():
    with open(FIXTURE) as f:
        data = json.load(f)
    print(f"fixture: {os.path.basename(FIXTURE)}  ({data['counts']})")
    return data["cases"]


def build_inputs(case):
    keys = TENSOR_KEYS[case["operator"]]
    tensors = tuple(
        _materialize_one(TensorDescriptor(**case["tensors"][k]), device="cpu")
        for k in keys
    )
    return tensors if len(keys) > 1 else tensors[0]


def run_check(case, inputs, candidate_fn, reference_fn, **kw):
    """Invoke the real check exactly the way verification/checker.py does."""
    op = case["operator"]
    spec = _Spec(op)
    primary = inputs[0] if isinstance(inputs, tuple) else inputs

    def _wrap(fn):
        def _run(x):
            new = (x,) + inputs[1:] if isinstance(inputs, tuple) else x
            return spec.run_candidate(fn, new)
        return _run

    return check_kernel_executed(
        _wrap(candidate_fn), primary, _wrap(reference_fn),
        spec=spec, inputs=inputs,
        raw_candidate_fn=candidate_fn, raw_reference_fn=reference_fn,
        **kw
    )


# --------------------------------------------------------------------------
# Control 0: the OLD check, verbatim, so the fixture is shown to discriminate.
# --------------------------------------------------------------------------

def old_check(candidate_fn, x, reference_fn):
    """The pre-fix probe, reproduced exactly as it was.

    Kept as a literal copy rather than imported, because the point is to
    compare against what shipped BEFORE the fix -- an import would track the
    fix and silently turn this control into a tautology.
    """
    x2 = x + torch.randn_like(x) * 0.1 + 1.0
    try:
        out1 = candidate_fn(x).detach().clone()
        out2 = candidate_fn(x2).detach().clone()
    except Exception as e:
        return False, f"Kernel raised an exception: {e}"
    if torch.equal(out1, out2):
        return False, "Kernel output is identical for two different inputs."
    return True, "ok"


def section_old(cases):
    print("\n[0] CONTROL 0 -- the OLD check must reproduce the recorded outcomes")
    torch.manual_seed(0)
    got = {"FAIL": 0, "PASS": 0}
    tot = {"FAIL": 0, "PASS": 0}
    wrong = []
    for case in cases:
        ref = REFERENCES[case["operator"]]
        inputs = build_inputs(case)
        primary = inputs[0] if isinstance(inputs, tuple) else inputs

        def _wrap(fn, _i=inputs):
            return (lambda t: fn(t, *_i[1:])) if isinstance(_i, tuple) else fn

        passed, _ = old_check(_wrap(ref), primary, _wrap(ref))
        expected = case["recorded_outcome"]
        tot[expected] += 1
        if (not passed and expected == "FAIL") or (passed and expected == "PASS"):
            got[expected] += 1
        else:
            wrong.append(case)
    print(f"  old check FAILS {got['FAIL']}/{tot['FAIL']} recorded false positives")
    print(f"  old check PASSES {got['PASS']}/{tot['PASS']} recorded passes")

    # The FALSE-POSITIVE direction is the one that proves this harness can
    # exhibit the defect, and it is asserted strictly.
    if got["FAIL"] != tot["FAIL"]:
        fail(f"control 0: the replay reproduces only {got['FAIL']}/{tot['FAIL']} "
             "recorded false positives, so it cannot exhibit the defect and "
             "every other result in this script is unfalsifiable")
    else:
        print("    -> the replay reproduces the defect; the fixture discriminates")

    # The recorded-PASS direction is NOT asserted strictly, and that is a
    # measured property of the OLD check rather than a weakness of the replay:
    # its single probe is drawn from randn, so on near-degenerate inputs its
    # verdict depends on the draw. Measured over 8 seeds, exactly one of the 51
    # (K scaled to 0.01, i.e. K nearly constant -- the marginal regime of this
    # very defect) flips, failing on 4 of 8 seeds with nothing else changed.
    # Section [4] shows the NEW check has no such instability.
    n_wrong_pass = tot["PASS"] - got["PASS"]
    if n_wrong_pass > 2:
        fail(f"control 0: old check mis-reports {n_wrong_pass} recorded passes, "
             "more than the 1-2 explained by its own RNG sensitivity -- the "
             "replay may have drifted from the recorded runs")
    elif n_wrong_pass:
        print(f"    ({n_wrong_pass} recorded pass(es) flip under the old check's "
              "own RNG sensitivity -- expected, see [4])")


# --------------------------------------------------------------------------
# 1 + control 1: replay every recorded case through the real check.
# --------------------------------------------------------------------------

def section_replay(cases):
    print("\n[1] Replaying recorded proposals through the real check")
    torch.manual_seed(0)
    by_outcome = {"FAIL": [], "PASS": []}
    for case in cases:
        ref = REFERENCES[case["operator"]]
        inputs = build_inputs(case)
        # The search executes the reference kernel as its own candidate
        # (coordinator.py:283-284 passes reference_src_path for both), so a
        # faithful replay uses the same callable on both sides.
        passed, detail = run_check(case, inputs, ref, ref)
        by_outcome[case["recorded_outcome"]].append((case, passed, detail))

    fps = by_outcome["FAIL"]
    now_pass = [c for c, p, _ in fps if p]
    print(f"  recorded FALSE POSITIVES: {len(now_pass)}/{len(fps)} now pass")
    for case, p, detail in fps:
        if not p:
            fail(f"still false-positives: {case['operator']} {case['id'][:8]} -- {detail}")

    # Attribution: report HOW each was cleared, so a pass that comes entirely
    # from rung E's "not evaluable" path is visible rather than indistinguish-
    # able from a pass earned by the ladder.
    by_mech = {}
    for case, p, detail in fps:
        if not p:
            mech = "STILL FAILING"
        elif "Not evaluable" in detail:
            mech = "rung E (not evaluable)"
        else:
            mech = "ladder: " + detail.split("perturbation: ")[-1].rstrip(").")
        by_mech.setdefault(mech, []).append(case["operator"])
    for mech, ops in sorted(by_mech.items()):
        counts = ", ".join(f"{o}x{ops.count(o)}" for o in sorted(set(ops)))
        print(f"    {len(ops):2d} cleared by {mech}  [{counts}]")

    ctrl = by_outcome["PASS"]
    still = [c for c, p, _ in ctrl if p]
    print(f"  CONTROL 1 -- recorded passes: {len(still)}/{len(ctrl)} still pass")
    for case, p, detail in ctrl:
        if not p:
            fail(f"control 1 regression: {case['operator']} {case['id'][:8]} -- {detail}")
    return by_outcome


# --------------------------------------------------------------------------
# Control 2: a genuine ghost must still be caught.
# --------------------------------------------------------------------------

def section_ghost(cases):
    print("\n[2] CONTROL 2 -- false-negative guard: a real ghost must still be caught")
    torch.manual_seed(0)
    caught = evaluable = 0
    missed = []
    for case in cases:
        ref = REFERENCES[case["operator"]]
        inputs = build_inputs(case)
        frozen = ref(*inputs) if isinstance(inputs, tuple) else ref(inputs)

        def ghost(*args, _f=frozen):
            # Ignores its input entirely -- exactly what the check exists for.
            return _f.clone()

        passed, detail = run_check(case, inputs, ghost, ref)
        # A ghost is only DETECTABLE where the reference itself moves. Where it
        # does not, the check correctly reports "not evaluable" -- and naive
        # allclose cannot see the bug there either, so there is no gap to miss.
        if "Not evaluable" in detail:
            continue
        evaluable += 1
        if passed:
            missed.append((case, detail))
        else:
            caught += 1
    print(f"  ghost caught on {caught}/{evaluable} evaluable inputs "
          f"({len(cases) - evaluable} not evaluable, correctly)")
    for case, detail in missed:
        fail(f"GHOST NOT CAUGHT ({case['operator']} {case['id'][:8]}): {detail}")
    if evaluable == 0:
        fail("control 2 vacuous: no input was evaluable, so nothing was tested")


# --------------------------------------------------------------------------
# Control 3: every rung must be load-bearing.
# --------------------------------------------------------------------------

def section_rungs(cases):
    """Measure each rung IN ISOLATION over the recorded false positives.

    LEAVE-ONE-OUT DOES NOT WORK HERE AND WAS THE FIRST DRAFT'S BUG. The rungs
    form a disjunction with overlapping coverage, so removing any single one
    left the total unchanged (delta 0 for all four) while the rungs actually
    rescue between 0 and 20 cases each. A control reporting "no effect" for a
    rung that single-handedly rescues 20/20 is worse than no control at all.

    So: run each rung as the ONLY rung and count what it rescues. That is a
    direct measurement, and it is what the numbers quoted in
    CHECK_ABLATION_FINDINGS.md §3.0 refer to.
    """
    print("\n[3] CONTROL 3 -- each rung measured in isolation (leave-one-IN)")
    fps = [c for c in cases if c["recorded_outcome"] == "FAIL"]
    original = runtime_guards._PRIMARY_PROBES

    def rescued(probe_name=None, companions=False):
        """Cases where THIS rung alone moves the candidate's output.

        Deliberately does not call check_kernel_executed: that would fold in
        rung E, and rung E is vacuous on this replay (candidate is reference),
        so every count would come back 20/20 and measure nothing.
        """
        torch.manual_seed(0)
        n = 0
        for case in fps:
            ref = REFERENCES[case["operator"]]
            inputs = build_inputs(case)
            tup = inputs if isinstance(inputs, tuple) else (inputs,)
            base = ref(*tup)
            if companions:
                moved = False
                for i in range(1, len(tup)):
                    if not (torch.is_tensor(tup[i]) and tup[i].is_floating_point()):
                        continue
                    alt = tup[:i] + (runtime_guards._probe_multiplicative(tup[i]),) + tup[i + 1:]
                    if not torch.equal(ref(*alt), base):
                        moved = True
                        break
                n += moved
            else:
                probe = dict(original)[probe_name]
                alt = (probe(tup[0]),) + tup[1:]
                n += not torch.equal(ref(*alt), base)
        return n

    results = {name: rescued(probe_name=name) for name, _ in original}
    results["companion"] = rescued(companions=True)
    for name, n in results.items():
        print(f"  rung {name:16s} rescues {n:2d}/{len(fps)} recorded FP cases")

    # The load-bearing claim, asserted rather than assumed. §3.0's originally
    # recommended fix was the multiplicative probe alone; it rescues nothing
    # here, and the companion rung is what actually does the work.
    if results["companion"] <= max(results[n] for n, _ in original):
        fail(f"control 3: the companion rung rescues {results['companion']}, no "
             f"more than the best primary rung -- §3.0's conclusion that it is "
             "the load-bearing prong no longer holds, so the docs are stale")
    else:
        print(f"    -> companion is the load-bearing rung "
              f"({results['companion']}/{len(fps)}), as §3.0 records")

    if results["companion"] == 0 or all(v == 0 for v in results.values()):
        fail("control 3 vacuous: no rung rescued anything, so this measured nothing")

    # Rungs that rescue 0 here are NOT dead: multiplicative and fresh_draw
    # target argmax/argmin tail risk on randn inputs, which this corpus (all
    # saturated/constant fills) does not contain. Reported so the distinction
    # between 'measured ineffective here' and 'useless' stays visible.
    zero = [n for n, v in results.items() if v == 0]
    if zero:
        print(f"    note: {', '.join(zero)} rescue 0 on THIS corpus (all "
              "saturated/constant fills); they target randn-input tail risk "
              "and are retained deliberately, not measured useful here")

    n_not_evaluable = 0
    torch.manual_seed(0)
    for case in fps:
        ref = REFERENCES[case["operator"]]
        inputs = build_inputs(case)
        _, detail = run_check(case, inputs, ref, ref)
        n_not_evaluable += "Not evaluable" in detail
    print(f"  rung E resolved {n_not_evaluable}/{len(fps)} via the not-evaluable "
          "path (0 expected when the companion rung already moved the output)")


def section_seed_stability(cases, n_seeds=8):
    """The new check's verdict must not depend on the RNG draw.

    Every rung draws from randn, so seed-dependence is a live risk, and the
    OLD check demonstrably had it: one of the 51 recorded passes flips on 4 of
    8 seeds (see [0]). A check whose verdict on a correct kernel is a coin flip
    is a false-positive source no matter how good its median behaviour is, so
    this asserts stability directly rather than trusting one lucky seed.
    """
    print(f"\n[4] Seed stability of the NEW check across {n_seeds} seeds")
    verdicts = {}
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        for case in cases:
            ref = REFERENCES[case["operator"]]
            inputs = build_inputs(case)
            passed, _ = run_check(case, inputs, ref, ref)
            verdicts.setdefault(case["id"], set()).add(passed)

    unstable = [k for k, v in verdicts.items() if len(v) > 1]
    always_pass = [k for k, v in verdicts.items() if v == {True}]
    print(f"  {len(always_pass)}/{len(cases)} cases pass on ALL {n_seeds} seeds")
    print(f"  {len(unstable)}/{len(cases)} cases give a seed-dependent verdict")
    for k in unstable:
        case = next(c for c in cases if c["id"] == k)
        fail(f"seed-unstable verdict: {case['operator']} {k[:8]} "
             f"(recorded {case['recorded_outcome']})")
    if len(always_pass) != len(cases):
        for k, v in verdicts.items():
            if v == {False}:
                case = next(c for c in cases if c["id"] == k)
                fail(f"fails on every seed: {case['operator']} {k[:8]} "
                     f"(recorded {case['recorded_outcome']})")
    else:
        print("    -> no RNG sensitivity; the old check had 1 such case")


def main():
    cases = load_cases()
    section_old(cases)
    section_replay(cases)
    section_ghost(cases)
    section_rungs(cases)
    section_seed_stability(cases)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED -- {len(FAILURES)} problem(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"PASSED -- all {len(cases)} recorded cases behave as required.")
    sys.exit(0)


if __name__ == "__main__":
    main()
