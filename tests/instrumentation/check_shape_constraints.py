#!/usr/bin/env python3
"""
Verification for 1b -- SHAPE_CONSTRAINTS enforcement in validate_proposal.

Standalone script, NOT a pytest test (see tests/instrumentation/README.md).
Plain python3: no pytest, no torch, no numpy, no GPU. Exit 0 = pass.

    python3 tests/instrumentation/check_shape_constraints.py

WHAT THIS GUARDS, AND WHY IT IS SHAPED THIS WAY
------------------------------------------------------------------------------
SHAPE_CONSTRAINTS rejects adversarial proposals before they execute. The danger
is NOT that it misses a bad input -- that merely wastes an iteration, which is
the status quo. The danger is that it rejects a GOOD one, silently suppressing
the edge cases the search exists to find. A too-strict table would make the
search quietly worse while every number still looked fine.

So the primary gate is a FALSIFICATION test, not a pass/fail test:

    Replay every historical proposal. Any proposal whose reference kernel
    ACTUALLY PASSED, but which the new table would REJECT, is proof the table
    is wrong. Zero such cases is the requirement.

Historical data can only ever falsify a constraint this way -- it can never
confirm one. Constraints are derived from reference-kernel source; see the
editing rule at the top of SHAPE_CONSTRAINTS. (Concretely: layernorm has ONE
historical passing proposal, matmul two. "All powers of two" in that data says
nothing about the kernel and must not be read as a constraint.)

SESSION_HANDOFF.md §5 records twelve instances of work that appeared verified
and was not, so per that discipline every positive claim here is paired with a
control that must fail. In particular §5 instance 11: an "N/N pass" claim with
no paired "the old behaviour fails these same N" is unfalsifiable -- a harness
that cannot reproduce the defect prints the identical number.
"""

import json
import os
import sqlite3
import sys
import glob

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from verification.adversarial_search import schemas as S  # noqa: E402

print(f"module under test: {S.__file__}")
if not os.path.abspath(S.__file__).startswith(os.path.abspath(REPO)):
    sys.exit("FATAL: importing schemas from outside the repo -- run is worthless. "
             "(§5 instance 4: a ported test once passed while validating a stale "
             "copy under /tmp.)")

failures = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


def mk(operator, tensors):
    """Rebuild an InputProposal from a stored proposal_json's tensors dict."""
    return S.InputProposal(
        proposal_id="replay", worker_id="w", iteration=0, operator=operator,
        tensors={
            k: S.TensorDescriptor(
                shape=list(v.get("shape") or []),
                dtype=v.get("dtype", "float32"),
                fill=v.get("fill", "randn"),
            )
            for k, v in tensors.items()
        },
        rationale="", predicted_failure_mode="",
    )


# ── Load every recorded proposal, with whether the REFERENCE actually ran ──────

def load_corpus():
    """
    Returns [(source, operator, tensors, reference_passed)].

    Two DB generations are read. Runs from 2026-08-21 onward have an
    `executions` table (item A1) carrying per-kernel results; older runs only
    have `verdicts.verdict_json.reference_passed`. Both are used -- more
    evidence is strictly better for a falsification test, and the older runs
    cover 9 operators the newer ones do not.
    """
    out = []
    roots = [
        os.path.join(REPO, "adversarial_results", "search_history.db"),
    ] + sorted(glob.glob(os.path.join(REPO, "adversarial_results", "*", "search_history.db")))

    for db in roots:
        if not os.path.exists(db):
            continue
        # Open read-write FIRST: these are WAL databases, and a read-only open
        # of a WAL db with no -shm sidecar fails with "unable to open database
        # file". Opening rw once materialises the sidecar. We never write.
        try:
            sqlite3.connect(db).close()
        except Exception:
            pass
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        tag = os.path.relpath(db, REPO)

        if "executions" in tables:
            rows = con.execute(
                "SELECT e.passed_checker, e.error_type, p.operator, p.proposal_json "
                "FROM executions e JOIN proposals p ON p.proposal_id = e.proposal_id "
                "WHERE e.kernel_id = 'reference'").fetchall()
            for passed, err, op, pj in rows:
                if err:
                    continue          # infra timeout: says nothing about shapes
                out.append((tag, op, json.loads(pj).get("tensors") or {}, bool(passed)))
        elif "verdicts" in tables:
            rows = con.execute(
                "SELECT v.verdict_json, p.operator, p.proposal_json "
                "FROM verdicts v JOIN proposals p ON p.proposal_id = v.proposal_id"
            ).fetchall()
            for vj, op, pj in rows:
                out.append((tag, op, json.loads(pj).get("tensors") or {},
                            bool(json.loads(vj).get("reference_passed"))))
        con.close()
    return out


def load_hits():
    """Proposals recorded as confirmed hits -- inputs that found a real bug."""
    out = []
    for db in [os.path.join(REPO, "adversarial_results", "search_history.db")] + \
              sorted(glob.glob(os.path.join(REPO, "adversarial_results", "*", "search_history.db"))):
        if not os.path.exists(db):
            continue
        try:
            sqlite3.connect(db).close()
        except Exception:
            pass
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for op, hm, pj in con.execute(
                    "SELECT v.operator, v.hit_mutants, p.proposal_json FROM verdicts v "
                    "JOIN proposals p ON p.proposal_id = v.proposal_id WHERE v.is_hit = 1"):
                out.append((op, json.loads(pj).get("tensors") or {}, json.loads(hm)))
        except Exception:
            pass
        con.close()
    return out


corpus = load_corpus()
hits = load_hits()

if not corpus:
    sys.exit("FATAL: no historical proposals found. This harness cannot "
             "discriminate anything without them -- a green run would be "
             "meaningless. Expected adversarial_results/**/search_history.db.")

print(f"corpus: {len(corpus)} recorded reference executions across "
      f"{len({c[1] for c in corpus})} operators")
print(f"hits:   {len(hits)} confirmed bug-finding proposals\n")


# ── 1. THE PRIMARY GATE: no legitimate input may be rejected ──────────────────

print("── 1. falsification: legitimate inputs must not be rejected ──")
falsifying, caught, allowed = [], 0, 0
for tag, op, tensors, ref_passed in corpus:
    err = S.validate_shape_constraints(mk(op, tensors))
    if ref_passed and err:
        falsifying.append((tag, op, {k: v.get("shape") for k, v in tensors.items()}, err))
    elif ref_passed:
        allowed += 1
    elif err:
        caught += 1

check(not falsifying,
      f"{len(falsifying)} proposal(s) whose reference PASSED would now be "
      f"REJECTED -- the constraint table is WRONG:\n" +
      "\n".join(f"      {t} {o} {s}\n        -> {e}" for t, o, s, e in falsifying[:10]))
print(f"   legitimate inputs allowed : {allowed}")
print(f"   invalid inputs caught     : {caught}")
print(f"   FALSIFYING cases          : {len(falsifying)}   (must be 0)")
for t, o, s, e in falsifying[:10]:
    print(f"      {t} {o} {s}\n        -> {e}")


# ── 2. Confirmed hits must survive ────────────────────────────────────────────

print("\n── 2. every confirmed hit must still be proposable ──")
rejected_hits = [(op, {k: v.get("shape") for k, v in t.items()}, m,
                  S.validate_shape_constraints(mk(op, t)))
                 for op, t, m in hits if S.validate_shape_constraints(mk(op, t))]
check(not rejected_hits,
      f"{len(rejected_hits)} confirmed hit(s) would now be rejected: {rejected_hits}")
print(f"   hits checked  : {len(hits)}")
print(f"   hits rejected : {len(rejected_hits)}   (must be 0)")
for r in rejected_hits:
    print(f"      {r}")


# ── 3. NEGATIVE CONTROLS ──────────────────────────────────────────────────────
# Without these, "0 falsifying cases" is unfalsifiable: a harness that cannot
# detect a bad table reports exactly the same number as one that can.

print("\n── 3. negative controls ──")
_original = dict(S.SHAPE_CONSTRAINTS)

# 3a. An over-tight table MUST be caught. This is the specific failure mode the
#     whole design guards against, so the harness must demonstrably detect it.
try:
    S.SHAPE_CONSTRAINTS.clear()
    S.SHAPE_CONSTRAINTS.update({
        op: {"tensors": {k: {"pow2_dims": (-1,)} for k in keys}}
        for op, keys in S.REQUIRED_TENSOR_KEYS.items()
    })
    bogus_falsified = [(op, {k: v.get("shape") for k, v in t.items()})
                       for tag, op, t, ok in corpus if ok
                       and S.validate_shape_constraints(mk(op, t))]
    bogus_hits = [(op, {k: v.get("shape") for k, v in t.items()})
                  for op, t, _ in hits if S.validate_shape_constraints(mk(op, t))]
finally:
    S.SHAPE_CONSTRAINTS.clear()
    S.SHAPE_CONSTRAINTS.update(_original)

check(bogus_falsified,
      "CONTROL FAILED: a deliberately over-tight table (every last dim must be "
      "a power of two) produced ZERO falsifying cases. The corpus cannot "
      "distinguish a good table from a bad one, so gate 1 proves nothing.")
check(bogus_hits,
      "CONTROL FAILED: the over-tight table rejected none of the confirmed "
      "hits, so gate 2 cannot detect suppression of real bug-finding inputs.")
print(f"   3a over-tight table -> {len(bogus_falsified)} falsifying, "
      f"{len(bogus_hits)} hits lost   (both must be > 0)")
for h in bogus_hits[:6]:
    print(f"        would have lost: {h}")

# 3b. An empty table must catch NOTHING, proving gate 1's `caught` count is
#     produced by the rules rather than by something incidental in the replay.
try:
    S.SHAPE_CONSTRAINTS.clear()
    noop_caught = sum(1 for tag, op, t, ok in corpus
                      if not ok and S.validate_shape_constraints(mk(op, t)))
finally:
    S.SHAPE_CONSTRAINTS.clear()
    S.SHAPE_CONSTRAINTS.update(_original)

# The universal positive-dimension rule still applies with an empty table, so a
# handful of genuinely zero-dim proposals may still be caught; the point is that
# it must be far below what the real table catches.
check(noop_caught < caught,
      f"CONTROL FAILED: an EMPTY constraint table still caught {noop_caught} of "
      f"the {caught} the real table catches. The replay is not actually "
      f"exercising SHAPE_CONSTRAINTS.")
print(f"   3b empty table      -> caught {noop_caught} vs {caught} real   "
      f"(must be strictly fewer)")

# 3c. The startup coverage validator must actually abort on a missing entry --
#     the validate_bug_pattern_hints precedent (§5), where running the control
#     was the only thing that revealed the assertion would have hard-failed the
#     search at startup.
try:
    S.SHAPE_CONSTRAINTS.pop("softmax")
    gaps = S.validate_shape_constraint_coverage()
finally:
    S.SHAPE_CONSTRAINTS.clear()
    S.SHAPE_CONSTRAINTS.update(_original)
check(gaps and any("softmax" in g for g in gaps),
      f"CONTROL FAILED: removing 'softmax' did not trip the coverage validator "
      f"(got {gaps!r}). A missing operator could ship unnoticed.")
print(f"   3c removed an entry -> validator reported {len(gaps)} gap(s)   "
      f"(must be > 0)")

check(S.validate_shape_constraint_coverage() == [],
      "coverage validator reports gaps on the REAL table after restore")
check(set(S.SHAPE_CONSTRAINTS) == set(S.REQUIRED_TENSOR_KEYS),
      "SHAPE_CONSTRAINTS and REQUIRED_TENSOR_KEYS key sets differ")
print(f"   restored, all {len(S.SHAPE_CONSTRAINTS)} operators covered")


# ── 4. Targeted cases: the exact inputs that motivated 1b ─────────────────────

print("\n── 4. the 2026-08-21 out-of-domain classes ──")
cases = [
    ("class A: non-pow2 head dim (fails to COMPILE)",
     "causal_flash_attention", {"Q": [64, 48], "K": [64, 48], "V": [64, 48]}, True),
    ("class A: head dim 33",
     "causal_flash_attention", {"Q": [64, 33], "K": [64, 33], "V": [64, 33]}, True),
    ("class B: rank 3 (N, D = Q.shape unpack error)",
     "causal_flash_attention", {"Q": [2, 32, 64], "K": [2, 32, 64], "V": [2, 32, 64]}, True),
    ("class B: rank 4",
     "causal_flash_attention", {"Q": [1, 1, 65, 64], "K": [1, 1, 65, 64], "V": [1, 1, 65, 64]}, True),
    ("legal: non-pow2 SEQUENCE length is fine (partial tile -- the useful case)",
     "causal_flash_attention", {"Q": [33, 64], "K": [33, 64], "V": [33, 64]}, False),
    ("legal: softmax non-pow2 reduction dim (exposes first_tile)",
     "softmax", {"x": [512, 777]}, False),
    ("legal: gelu is rank-agnostic (view(-1))",
     "gelu", {"x": [2, 3, 5, 7]}, False),
    ("legal: flash_attention non-pow2 N with pow2 D",
     "flash_attention", {"Q": [96, 64], "K": [96, 64], "V": [96, 64]}, False),
    ("matmul inner dims disagree (silent garbage, not a raise)",
     "matmul", {"A": [64, 128], "B": [256, 64]}, True),
    ("legal: matmul with agreeing inner dims",
     "matmul", {"A": [64, 128], "B": [128, 64]}, False),
    ("zero dimension",
     "softmax", {"x": [0, 64]}, True),
]
for label, op, shapes, want_rejected in cases:
    err = S.validate_shape_constraints(
        mk(op, {k: {"shape": v} for k, v in shapes.items()}))
    got = err is not None
    check(got == want_rejected,
          f"{label}: expected {'reject' if want_rejected else 'ACCEPT'}, "
          f"got {'reject' if got else 'accept'} ({err})")
    print(f"   [{'reject' if got else 'accept'}] {label}")
    if got and want_rejected:
        print(f"       reason: {err[:100]}")


# ── 5. Rejection messages must be actionable ──────────────────────────────────
# The reason string is fed to the model verbatim on the retry turn. If it does
# not name the offending tensor and the requirement, the retry cannot succeed.

print("\n── 5. rejection reasons are specific enough to act on ──")
err = S.validate_shape_constraints(
    mk("causal_flash_attention", {k: {"shape": [64, 48]} for k in ("Q", "K", "V")}))
check(err is not None and "Q" in err and "48" in err and "power of two" in err,
      f"pow2 rejection message is not actionable: {err!r}")
err2 = S.validate_shape_constraints(
    mk("causal_flash_attention", {k: {"shape": [2, 32, 64]} for k in ("Q", "K", "V")}))
check(err2 is not None and "rank" in err2.lower(),
      f"rank rejection message does not mention rank: {err2!r}")
print("   pow2 message names tensor, value and requirement: ok")
print("   rank message names rank: ok")

from verification.adversarial_search.prompts.base import format_rejection_turn  # noqa: E402
shape_turn = format_rejection_turn(ValueError(err), "causal_flash_attention")
parse_turn = format_rejection_turn(ValueError("Expecting value: line 1 col 1"), "softmax")
check("do not change the format" in shape_turn,
      "shape rejection prompt does not tell the model its JSON was fine")
check("No markdown" in parse_turn,
      "parse-failure prompt lost its formatting guidance")
check("No markdown" not in shape_turn,
      "shape rejection prompt still gives JSON-formatting advice, which is the "
      "misleading behaviour 1b set out to fix")
print("   shape vs parse retry prompts differ appropriately: ok")


# ── 6. Worker survives a rejection (the throughput prerequisite) ──────────────
# coordinator._worker_loop used to `return` when a worker exhausted its retries,
# forfeiting every remaining iteration. Measured cost: worker w0 died at
# iteration 13 of the 2026-08-21 run, which is why it produced 74 proposals
# rather than 80. Adding constraints raises the rejection rate, so without
# recovery 1b would be a net throughput loss.

print("\n── 6. a rejected proposal must not kill the worker ──")
import inspect  # noqa: E402
# coordinator.py is read as TEXT, never imported: it pulls in executor.py, which
# imports torch at module scope, and this suite must stay runnable with plain
# python3 on the dev machine (see tests/instrumentation/README.md).
src = open(os.path.join(REPO, "verification/adversarial_search/coordinator.py")).read()
check("except ProposalRejected" in src,
      "coordinator does not handle ProposalRejected -- a rejected proposal "
      "still kills the worker and forfeits its remaining budget")
check(src.count("except ProposalRejected") >= 2,
      "coordinator handles ProposalRejected in fewer than both places -- the "
      "cold-start path and the refine path each forfeit the whole worker")
_after_reset = src.split("resetting to a fresh proposal")[-1][:400] \
    if "resetting to a fresh proposal" in src else ""
check("worker.propose()" in _after_reset,
      "the refine-path ProposalRejected handler does not recover with a fresh "
      "proposal; a plain `continue` would re-execute the previous iteration's "
      "input, duplicating work and double-counting statistics")

from verification.adversarial_search.worker import AdversarialWorker, ProposalRejected  # noqa: E402
wsrc = inspect.getsource(AdversarialWorker._call_and_parse)
check("raise ProposalRejected" in wsrc,
      "worker still raises a bare RuntimeError on exhaustion, which the "
      "coordinator cannot distinguish from an LLM outage")
check("_history.pop()" in wsrc,
      "worker does not drop the dangling user turn after a failed call -- the "
      "next attempt would send two user messages in a row and stay poisoned")
print("   coordinator recovers via a fresh proposal: ok")
print("   worker raises ProposalRejected and cleans its history: ok")

# The history-cleanup claim, exercised rather than grepped.
w = AdversarialWorker(worker_id="t", operator="softmax", model="none")
w._history = [{"role": "user", "content": "dangling"}]
before = len(w._history)
try:
    w._llm_call = lambda m: "not json at all"
    w._call_and_parse("second user message")
except ProposalRejected:
    pass
except Exception as e:  # pragma: no cover
    failures.append(f"expected ProposalRejected, got {type(e).__name__}: {e}")
checks += 1
if w._history and w._history[-1].get("role") == "user" and len(w._history) > before:
    failures.append("history still ends with a dangling user turn after a "
                    "failed call -- the next call would send two in a row")
print(f"   history after failed call: {len(w._history)} entr(ies), "
      f"last role={w._history[-1]['role'] if w._history else 'none'}")


# ── Result ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
if failures:
    print(f"FAIL — {len(failures)} of {checks} checks failed\n")
    for f in failures:
        print(f"  ✗ {f}\n")
    sys.exit(1)
print(f"PASS — {checks} checks")
print(f"  {allowed} legitimate inputs allowed, {caught} invalid caught, "
      f"0 falsifying")
print(f"  {len(hits)} confirmed hits all still proposable")
print(f"  controls fired: over-tight table loses {len(bogus_hits)} hits, "
      f"empty table catches {noop_caught} vs {caught}")
