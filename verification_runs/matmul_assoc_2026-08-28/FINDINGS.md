# The last 2 reference-suspect records adjudicated: both matmul scalar_associativity failures are CHECK-DOMAIN false alarms of the pre-loosening tolerance, the (65,65) hypothesis is refuted, and the ledger is now fully cleared — with one latent boundary flagged on the shipped tolerance at K≈4000

**Adjudicated 2026-08-28, CPU fp32/fp64 emulation, 10-seed unanimity**
(randn fills, no recorded seeds — same discipline as the norm and softmax
adjudications). Probe `probes/matmul_assoc_adjudication.py`, log
`data/matmul_assoc_adjudication.log`. Records: `50fb4e31`
((64,256)@(256,32)) and `eab75b02` ((65,65)@(65,65)), both plain randn
scale-1 — the only two entries of the 45-record reference-suspect ledger
that survived the norm (28) and softmax (12) rounds.

## Verdict up front

| | result |
|---|---|
| `50fb4e31` | **CHECK-DOMAIN FALSE ALARM** — faithful kernel emulation fails the run-era check 10/10 (max_err 9.8e-4–1.5e-3, 10–15× its atol), an *independently correct* fp32 matmul (torch.mm) fails 10/10 (up to 59× atol), the algebraic identity holds exactly in fp64 (0/10, ~1e-11), and the SHIPPED loosened check passes 10/10. |
| `eab75b02` (the (65,65) one) | **CHECK-DOMAIN FALSE ALARM** — same pattern (6–9× atol faithful, torch.mm worse, algebra exact, shipped check clean). |
| the "(65,65) boundary-tile" hypothesis | **REFUTED.** The run-era check false-positives **10/10 at every control shape including pure-pow2 (64,64) and (128,128)** with the same margins, and kernel-vs-torch.mm divergence is flat ~1e-5 across shapes with **no spike at 65 or 96** — the kernel's masked boundary tiles are numerically indistinguishable from full tiles. The 65 in the record was incidental; the check failed everywhere. |
| the "pre-loosening tolerance FP" hypothesis | **VERIFIED, from source + git.** The run-era (2026-07-23) check was the committed `atol=1e-4` with `torch.allclose`'s default rtol=1e-5; the loosening to atol=2e-3/rtol=1e-3 with the documented FP note (max_err 0.0012–0.0015 "both CPU and GPU" — squarely inside this round's measured 9.8e-4–1.5e-3) is the uncommitted working-tree fix. These two records are that documented FP, caught in the wild a month before it was diagnosed. |

**The reference-suspect ledger is now closed: 45/45 adjudicated** — 3 real
reference bugs (flash, sdpa, layernorm — all fixed), 42 check-domain false
alarms (27 norm + 12 softmax + 2 matmul + the rest of the norm round's
accounting). Zero remain open.

## The mechanism, sharpened beyond the doc note

Two independent fp32 error sources defeat the run-era atol=1e-4, and the
second was not in the loosening note:

1. **Accumulation-order noise** (the documented one): (100·A)@B and
   100·(A@B) round differently; the difference is a random walk of scale
   E/c ≈ **(15–18)·u·√K** per element (measured constant across
   K = 32…1024; u = 2⁻²⁴). At c=100, K=256 this is 10–15× the run-era
   atol — the check was invalid at **every** corpus shape, and in fact at
   every K ≥ 1 (the constant alone exceeds atol/c·u).
2. **Input-scaling rounding, implementation-independent**: the check
   computes `c*A` in fp32 *before* the kernel sees it, so even an
   infinitely-precise accumulator inherits a `round32(100·a)` random walk
   — measured: **fp64-exact accumulation still fails the run-era check
   1/10 at K=256** (margin 1.24× on a mid-magnitude element). No
   implementable kernel can pass this check reliably at atol=1e-4; the
   tolerance was below the check's own input quantization.

The failing elements are the near-zero-output ones, as the doc note said
(the error is |y|-independent while the old rtol=1e-5 slack is
|y|-proportional) — confirmed per-seed via the worst-element location.

## Counterfactual, closed the honest way

Both proposals recorded confirmed mutant gaps at run time (`50fb4e31`
caught skip_boundary, `eab75b02` caught swapped_strides). Under the
corrected check the reference passes 10/10, so both proposals were valid
and **both would have been hits**. Effect on published numbers:
second-order only (proposals-spent counts), exactly the boundary
CLAIMS_SWEEP drew; no §4-style denominator claim covers matmul.

## One latent boundary FLAGGED on the shipped tolerance (not fixed here)

The shipped atol=2e-3/rtol=1e-3 is clean on every tested corpus-scale
shape (0/10 FP at K = 32…2048). But the worst per-element violation ratio
max_j err_j/(2e-3 + 1e-3·|y_j|) grows as **√K** (measured 0.25 → 0.37 →
0.53 → 0.78 → **1.09** at K = 256/512/1024/2048/**4096**): a correct
kernel starts false-positiving at **K ≈ 4000**. Corpus margin at K=256 is
3.9×. Per fix discipline this is flagged, not fixed — if a future corpus
adds K ≳ 2048 matmuls, the tolerance needs the same treatment as the
distributivity derivation (an atol term scaling with c·u·√K, or comparison
in unscaled units), with its own regression round.

## Limits

- CPU emulation: `tl.dot` on the T4 accumulates within a 32-block in a
  different order than torch's CPU mm; the doc note's own GPU measurement
  (0.0012–0.0015) sits inside this round's faithful-emulation range, and
  every classification margin is ≥ 6× with an independent second
  implementation agreeing, so no verdict depends on ordering details.
- The K≈4000 crossing is measured at (64,K)@(K,64) randn; the √K law is
  clean but the crossing constant shifts with M·N (max over more
  elements) — treat 4000 as order-of-magnitude, derived boundary shape.
- Run-era check reconstructed from git HEAD (last commit touching the
  file predates the 2026-07-23 runs; the loosening exists only in the
  working tree — verified via `git diff HEAD`).

## Reproduce

```bash
.venv/bin/python probes/matmul_assoc_adjudication.py
```
