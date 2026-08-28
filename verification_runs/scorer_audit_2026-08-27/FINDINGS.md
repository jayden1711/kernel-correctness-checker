# Scorer audit — the diversity penalty is dormant because its consumer was never built; the error penalties never had an occasion to fire. Nothing shipped; recommendations below.

**Investigated 2026-08-27.** Probes in `probes/`, replay logs in `data/`.
**No scorer, strategy, or coordinator behavior was changed** — per the task's
constraint, and because any scorer change moves `beam[0]`, which reseeds LLM
refinement paths (the same class of hazard as the RNG-stream rules).

Evidence base: the three banked run databases (14 runs, 414 verdicts, 308
instrumented per-kernel executions), replayed chronologically through the real
strategy classes; git history of `verification/adversarial_search/`.

---

## Verdict up front

| question | answer |
|---|---|
| Is the diversity weight zero? | **No. λ = 3.0, non-zero, correctly plumbed** CLI → coordinator → `DiverseBeamStrategy`. Not a wiring bug. |
| Why is it inert, then? | **Three stacked reasons; the decisive one is that the coordinator only ever consumes `_beam[0]`**, which is the unconditional global argmax under every strategy at every λ. §1 |
| Bug or dormant feature? | **A half-implemented design, from day one.** The docstrings describe per-worker beam-slot ownership; `_pick_beam_seed(worker_id)` takes the parameter that design needs and ignores it. Strategy files untouched since the introducing commit (`5277cd1`, 2026-07-21). §2 |
| Did inertness cost anything? | **Zero on every recorded run, measured.** Replaying all 414 selection steps: `beam[0]` identical between beam and diverse at λ=3, λ=100, and λ=10⁹ — **0/414 divergences**; the beam *set* differed at all in only 2/414 steps (never consumed). 9 of 12 original runs actually used `diverse`, including four at beam width 6 where the theory-audit's B≤4 bound doesn't apply — still zero. §3 |
| Error penalties: what happened? | **Aspirational docstrings written against a schema that couldn't express them.** At authoring time `ProposalVerdict` had no per-mutant error or no-gap data, so beam's −2 and greedy's −3/+2 were unimplementable; the code shipped cut-down and the docstrings were never reconciled. Two further latent defects found in the beam scorer while verifying: a dead `+3` branch and a misfiring `+2` bonus. §4 |
| Did the missing error penalties cost anything? | **No occasion to fire, ever:** zero mutant-errored verdicts in all instrumented executions (the 4 recorded TimeoutErrors are reference-side, already handled by −5). Rescoring both instrumented runs under the intended scheme changes the consumed seed on **0/154 steps**. §5 |
| Recommendations | Diversity: **leave behavior as-is, fix the docstrings now; per-worker beam seeding is a fix-later behind a measured A/B.** Error penalties: **fix later, after the in-flight (uncommitted) verdict-split change lands** — the fix is now cheap but must not ride on someone else's diff. Precise diffs in §6. |

---

## 1. Why the penalty is inert — three mechanisms, ranked

**(a) Score magnitudes (the theory-audit result, holds at B=4).** Every valid
proposal scores ≥ 10; the maximum overlap penalty in a width-4 beam is
3·3 = 9. At B=4 no valid proposal can ever be displaced.

**(b) The acceptance rule plus the tail-fill (kicks in at B=6).** Four of the
real runs used `--workers 6`, so beam width 6 and a maximum penalty of 15 —
mechanism (a) no longer protects a 12-point proposal. But `diverse.select`
admits the first `B//2` slots unconditionally, only *skips* on a non-positive
effective score (it never re-ranks), and then back-fills the beam from the
skipped candidates **in rank order**. Measured on the real pools: the selected
*set* differed from plain top-B in 2 of 414 steps (both in the layernorm run);
order never differed otherwise.

**(c) The decisive one: nothing reads slots 1..B−1.** `self._beam` is consumed
at exactly one place, `_pick_beam_seed` (`coordinator.py:575-584`), which
ignores its `worker_id` argument and returns `_beam[0]`. Slot 0 is the
unconditional global argmax under beam (stable top-B sort), diverse ("pick the
highest-scoring candidate unconditionally, best first"), and greedy (single
max) alike. **Therefore no λ — including ∞ — can change any decision the
coordinator makes.** Verified on the recorded pools: 0/414 slot-0 divergences
at λ ∈ {3, 100, 10⁹}.

The replay also validates itself: recomputing the shipped score from every
stored verdict reproduces the stored `beam_score` **414/414**, and the stored
score distribution ({−5, 3, 11, 12, 18, 26, 34}) decomposes exactly as
{−5 ref-fail, −5+8k ref-fail-with-gap-mutants, 10+2, 10+8k} — confirming which
code path produced every recorded number.

## 2. Intentional or a bug? The history says: designed, half-built, never revisited

- The entire `strategy/` package dates to the initial pipeline commit
  (`5277cd1`, 2026-07-21) and is **byte-identical today** — no tuning, no
  follow-up, no A/B.
- `beam.py`'s own docstring describes the missing half: *"A beam of 4 with 4
  workers means each worker owns one beam member and expands it
  independently."* The shipped `_pick_beam_seed(worker_id)` — same commit —
  accepts the parameter slot-ownership needs and returns `_beam[0]`
  unconditionally, so **all workers converge on refining the single global
  best**. That is the exact "collapse onto the highest-scoring region" that
  `diverse.py`'s docstring says the penalty exists to prevent — the collapse
  is implemented in the coordinator, downstream of any selection strategy.
- Conclusion: **dormant feature, not a mistuned knob.** The diversity penalty
  operates on state (`_beam[1:]`) that has no consumer. In the shipped system,
  proposal diversity comes from the round-robin per-worker bug-pattern prompts
  and per-worker feedback, not from beam selection.

## 3. Has the missed coverage been measured? No — and retroactively it cost nothing

No strategy A/B exists anywhere in the repo (no benchmark doc mentions one,
and the run table shows each operator searched once, under one strategy). What
the banked data does support is the replay above: on every pool that ever
actually arose, **turning the penalty into a hard exclusion (λ=∞) would not
have changed a single seed decision.** So the inertness has zero measured
cost.

The open question that remains real is not the penalty but the **consumer**:
whether per-worker beam-slot seeding (true beam expansion) would reduce
proposals-to-hit. That connects to the theory audit's H4 (no completeness
property; coverage is heuristic): today's search explores one lineage plus
prompt-level pattern diversity. Untestable retroactively — the counterfactual
proposals were never generated. It is a legitimate future experiment, not a
retroactive loss.

## 4. The error penalties — what the intended behavior was, and two more defects found while pinning it down

Docstring vs implementation, verified against the code and the stored score
distribution:

| term (docstring) | beam.py code | greedy.py code |
|---|---|---|
| −2 (beam) / −3 (greedy) per errored mutant | **absent** | **absent** |
| +3 (beam) / +2 (greedy) per mutant caught without gap | **dead branch** (see below) | **absent** |
| +2 valid-but-nothing-caught (beam) | **misfires**: granted whenever `hit_mutants` is empty, i.e. also on caught-no-gap and crashed-mutant verdicts | n/a |

- **Dead branch:** `hit_mutants` is only ever populated by the
  caught-with-gap branch of `_evaluate_verdict`, which also sets
  `gap_confirmed=True` — so beam's `+3 if not gap_confirmed` arm is
  unreachable. The stored scores corroborate: no recorded value needs +3 to
  decompose.
- **Provenance:** at the authoring commit, `ProposalVerdict` carried only
  `hit_mutants` / `missed_mutants` (a union that conflates caught-no-gap with
  not-caught) and **no error information at all**. Both promised terms were
  unimplementable against that schema. This is an aspirational docstring the
  code was cut down from, not a stale docstring after a refactor — nothing was
  ever removed.
- **What a crashed mutant scores today:** `_error_result` returns
  `passed_checker=False, passed_naive=False`, so it lands in `caught_no_gap`:
  no credit, no penalty, plus the misfiring +2 — total 12, indistinguishable
  from a clean valid no-catch proposal. It cannot fake a hit (a hit requires
  `passed_naive=True`), so the failure mode is bounded: a crash-heavy proposal
  ties a clean one and can win `beam[0]` only by insertion order.

## 5. Measured impact of the missing terms: zero occasions in recorded history

From the two runs that banked per-kernel executions (154 verdicts, 308
executions — the causal_flash_attention reruns, exactly the operator with the
most out-of-domain adversarial inputs):

- **Mutant-errored verdicts: 0.** The only 4 recorded errors (TimeoutError)
  are on the **reference** kernel, where `reference_passed=False` already
  applies −5 under both shipped and intended scoring.
- Every one of the 154 verdicts has caught-no-gap mutants (the known CFA
  no-gap story), so the intended +3 term shifts scores nearly uniformly —
  and the consumed running-argmax seed differs on **0/154 steps** under
  intended vs shipped scoring.
- The 260 older verdicts (12-run DB) bank no per-mutant error data; impact
  there is unknowable but the same uniform-shift argument applies to their
  all-no-gap operators, and reference-side crashes were already −5.

## 6. Recommendations (nothing shipped; sign-off required for anything behavioral)

**R1 — diversity penalty: leave behavior as-is; correct documentation now.**
Cost/benefit: a behavioral "fix" (making the penalty bite) would change
nothing while `_pick_beam_seed` reads only `[0]`, and changing
`_pick_beam_seed` itself is an exploration-policy change with unmeasured
benefit — exactly the class of change this project doesn't make on zero
evidence (the n_samples precedent). The cheap, zero-risk action is honest
docs: mark `diverse.py` and the `--diversity-weight` help text as having no
effect on exploration under the current coordinator, and fix `beam.py`'s
"each worker owns one beam member" line to describe what `_pick_beam_seed`
does. **Classification: leave as-is + corrected docstring.**

**R2 — per-worker beam-slot seeding (the dormant feature): fix later, behind
a measurement.** If ever pursued: `_pick_beam_seed(worker_id)` maps each
worker to `_beam[worker_index % len(_beam)]` (falling back to `[0]` when the
beam is short), making selection — and only then the diversity penalty —
actually matter. Must be gated on an A/B measuring proposals-to-hit and
non-hit rate against the current behavior, with the RNG/seeding caveats the
project already applies to search changes. Not urgent: the recorded runs show
the current policy still hits on 11 of 12 operators.

**R3 — error penalties and the +2/+3 defects: fix later, specifically after
the in-flight verdict-split change lands.** The data needed to implement the
docstring faithfully (`not_caught` / `caught_no_gap` / `mutant_records` with
per-mutant error outcomes) exists **only in the currently uncommitted working
tree** (someone's in-flight change; `strategy/` itself is clean). Implementing
now would ride on an unlanded diff. Once landed, the faithful fix is small and
self-contained in `beam.py`/`greedy.py`:

- score errored mutants (per `mutant_records` outcome/error) at −2 (beam) /
  −3 (greedy), excluded from any "caught" credit;
- score `caught_no_gap` mutants at +3 (beam) / +2 (greedy), replacing the dead
  branch;
- grant the +2 bonus only when nothing was caught *and* nothing errored.

Rationale for implementing rather than doc-fixing: the −2 term guards a real
(if so-far unobserved) failure mode — a crash-inducing proposal tying a clean
one at 12 and winning the seed by insertion order — and the replay shows the
change is **retroactively verdict-neutral** (0/154 seed changes), which is
also the regression test to ship with it: replay both instrumented DBs and
assert no seed decision changes. If the team prefers zero behavioral risk,
the acceptable alternative is R1-style: rewrite the docstrings to describe
shipped behavior and delete the promises.

**Priority order: R1 (now, docs only) → R3 (after the verdict-split lands) →
R2 (only with an A/B someone actually wants to run).**

---

## 7. Reproduce

```bash
cd verification_runs/scorer_audit_2026-08-27
PY=../../.venv/bin/python
$PY probes/replay_selection.py   # 14 runs, 414 verdicts: score replication + strategy-equivalence replay
$PY probes/replay_intended.py    # 2 instrumented runs: intended-vs-shipped scoring, seed divergence count
```

Key raw queries: `runs` table for strategies/widths actually used;
`executions WHERE error_type IS NOT NULL` for the 4 reference-side timeouts;
`SELECT beam_score, count(*) FROM verdicts GROUP BY beam_score` for the score
decomposition.

## 8. Limits

- The 12-run main DB predates the `executions` table, so mutant-error
  frequency there is unknowable; the zero-occasions claim is exact only for
  the 154 instrumented verdicts.
- The replay reconstructs the coordinator's pool from `verdicts.created_at`
  ordering; with 4–6 concurrent workers, sub-second interleaving could differ
  from the true lock acquisition order. This affects tie-breaking only, and
  every conclusion above is tie-insensitive (slot-0 identity holds pointwise,
  not just in aggregate).
- "Zero cost" is a statement about the recorded runs, not about future
  operators or crash-prone mutants; R3's latent failure mode is real going
  forward even though it has never fired.
- Greedy was never used in a recorded run; its mismatches are documented but
  have no measured-impact question to answer.
