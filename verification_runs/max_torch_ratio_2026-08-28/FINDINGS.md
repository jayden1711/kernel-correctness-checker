# max_torch_ratio = 0.5 — derivation, margins, and scope — 2026-08-28

**Item:** theory-audit flag #4. `check_partial_computation(candidate,
max_torch_ratio=0.5)` (ast_analysis.py:330-358) flags a candidate when
counted PyTorch compute calls exceed half of counted compute calls
(torch calls + Triton launches), per the wrapper function's AST. The 0.5 was
an unexamined magic constant, and CORPUS_EXPANSION_PLAN §Change-3 flags the
check as unsound for fused kernels. This round gives it the l3_margins
treatment: derive the value, measure the nearest miss, scope the unsoundness.

## 1. The constant is derivable, and the derivation says exactly 0.5-strict

The check's contract (its own docstring) is majority rule: fail when the
candidate does NOT do "most of its own work". Under the call-count statistic,
"most" is the > 1/2 criterion — 0.5 with strict inequality is the literal
transcription of the spec, not a tuned number. What makes it *sound* on this
corpus is an empirical fact the constant's author never recorded: **honest
corpus kernels legitimately delegate at most half of their counted compute
calls to the host** — exactly one kernel family attains the bound
(cross_entropy: the spec itself puts the final mean over per-row losses on
the host as `per_sample_loss.mean()`, and `mean` is in `_TORCH_COMPUTE_OPS`,
so its wrapper measures 1 torch / 1 launch = ratio 0.5).

## 2. Measured margins (124 wrapper functions + 12 banked corpus records)

`probes/measure_ratio.py` over every on-disk candidate population
(`data/ratio_measurements.json`); corroborated by the 477 banked
`partial_computation` check records in results_raw.json.

| population | wrappers | ratio distribution |
|---|---:|---|
| reference | 29 | all 0.0 except cross_entropy = **0.500** |
| cheating mutants | 41 | all 0.0 except cross_entropy/missing_max_subtraction = **0.500** |
| near_miss (m+v series) | 50 | all 0.0 |
| experiments/ (torch driver scripts, not candidates) | 4 | 1.0 — would flag, correctly |

Banked corpus agreement: 12 records measure "delegation ratio 50%"
(cross_entropy reference ×10 + its mutant ×2), everything else 0%; zero
fires ever recorded.

**Nearest-miss geometry:**
- **FP side: margin is exactly ZERO.** The shipped reference cross_entropy
  sits *at* the threshold and survives only by the strict `>`. The knife-edge
  is deterministic, not fp-fragile: equality requires n_torch = n_triton, and
  IEEE division then returns exactly 0.5 (representable, correctly rounded),
  so `0.5 > 0.5` is reliably False on every platform. But any tightening —
  `>=`, or any t < 0.5 — flags a shipped reference kernel immediately.
- **Cheat side: margin 2×.** The minimal *visible* cheat (pure delegation via
  a listed op name, e.g. `return torch.softmax(x, -1)`) measures 1.0;
  flagged for any t < 1.
- **Sound interval on this corpus: t ∈ [0.5, 1.0)** — all values give
  bit-identical corpus outcomes (honest max = 0.5 passes via strict >, cheats
  at 1.0 flagged). The shipped value is the infimum: maximum cheat
  sensitivity subject to zero FP. Unlike the l3 probe constants (dead zone
  (0.58, 0.998), slack on both sides), this constant has an *attained*
  boundary on the FP side — the honest distribution is bimodal {0, 0.5} with
  mass ON the threshold.

## 3. Where the check is structurally blind (no constant fixes these)

Verified by running the real `check_partial_computation` on synthesized
candidates (probe's SYNTH panel):

1. **Decoy-launch delegation** — one trivial Triton launch + one listed
   torch op computing the whole answer → ratio 0.5 → **passes**. Catching it
   needs t < 0.5, which flags reference cross_entropy: the honest and cheat
   populations OVERLAP at exactly 0.5, so under the call-count statistic
   **no threshold is simultaneously sound and complete**. The constant is at
   its optimum; the residual gap belongs to the statistic (calls, not FLOPs).
2. **The op-name allowlist misses 14 of 27 corpus operators' canonical
   delegation call**: log_softmax, cross_entropy, group_norm, instance_norm,
   rms_norm, scaled_dot_product_attention (flash/sdpa/causal ×3), and all six
   pools. `F.log_softmax(x)` / `F.cross_entropy(...)` / `F.max_pool1d(...)`
   measure ZERO compute ops → "no compute ops detected" → pass (verified).
   Mitigation as deployed: with no launch present, `check_ghost_optimization`
   fails the candidate (ast_analysis.py:106-121 — verified: no-launch, no
   `.apply()` → FAIL), so launch-free delegation is still caught one check
   earlier. The *combination* — decoy launch + unlisted op name — passes both
   AST checks (verified: "0 PyTorch op(s), ratio 0%"); downstream numeric
   layers then see a numerically CORRECT output, because delegation is a
   benchmark-gaming pattern, not a wrongness pattern. Net: **the checker as
   deployed cannot flag decoy-launch delegation.** No such candidate exists
   in any corpus or banked search proposal (searched: proposal store holds
   adversarial inputs, not kernel sources), so this is a documented limit,
   not a live defect.

## 4. Fused-kernel scoping (the CORPUS_EXPANSION_PLAN §Change-3 unsoundness)

Confirmed empirically: an honest fused candidate that keeps conv in cuDNN and
fuses only the epilogue (`F.conv2d` + 1 launch) measures **0.5 — knife-edge
pass**; adding ANY second legitimate torch compute call (bias/relu/aux
reduction: 2 torch / 1 launch = 0.667) is a **false FAIL** — and its AST
counts are *identical* to a genuine partial-delegation cheat of the same
shape. For a fused-kernel corpus the honest ratio range {0.5, 0.67, …}
overlaps the flag region: under call counting there is no threshold to
retune, the same conclusion as §3.1 but now with real kernels on the wrong
side. Before such kernels can be checked, one of (in rising order of cost):
(a) gate `partial_computation` (and `ghost_optimization`) off for corpora
whose contract permits partial delegation — sound, loses the anti-gaming
signal; (b) make the statistic op-role-aware: flag only when the *named
operator's own* canonical call appears (needs the per-op delegation-name
table from §3.2 anyway); (c) weight by estimated FLOPs instead of call
count, and/or follow submodule sources per the plan's suggestion — the only
route that keeps a quantitative "most of the work" semantics.

## 5. Disposition

Shipped value **kept, unchanged**: 0.5-strict is derived (majority rule) and
optimal (infimum of the sound interval). Two follow-ups recommended, not
done here: a regression test pinning the two boundary facts (reference
cross_entropy passes at exactly 0.5; listed-op pure delegation flags), and
the per-op delegation-name table if the anti-gaming signal is ever wanted
for the 14 uncovered operators. Fused-kernel corpora remain out of scope for
this check until §4(a–c); that is a corpus-contract gate, not a constant to
retune.

## Reproduce

```bash
.venv/bin/python verification_runs/max_torch_ratio_2026-08-28/probes/measure_ratio.py
```
Pure AST, deterministic, no GPU.
