# Layer-3 validity domains: the eps-family boundaries derived and verified to ~1.2x, the softmax positivity domain derived — and all 12 outstanding softmax reference-suspect records adjudicated CHECK-DOMAIN FALSE ALARMS at 10x margin

**Derived and verified 2026-08-28, CPU fp32/fp64 emulation.** Probes in
`probes/`, logs in `data/`. Style follows the two existing exemplars: the
fp-absorption threshold (`v > 21.5` fp32, NUMERICAL_THEORY §4.2) and the
softmax-blindness condition. This round covers the **eps-vs-scale family**
(l1norm / l2norm / frobenius_norm — the last family members without derived
domains) and the **softmax positivity check's fp-underflow domain**, then
uses the latter to adjudicate the 12 outstanding reference-suspect records
from `../attention_mask_fix_2026-08-27/FINDINGS.md` §4.

## 1. Derived validity domains (eps = 1e-12, atol = 1e-3, fp32 kernels)

Kernel eps placements differ per operator and matter:
`l1norm: x/(S₁+ε)`, `l2norm: x/√(S₂+ε)`, `frobenius: x/(√S₂+ε)`.

| check | exact-math deviation | check INVALID iff | measured/predicted boundary |
|---|---|---|---|
| `unit_l1_norm` | ε/(S₁+ε) | S₁ < ε(1−a)/a ≈ **1.0e-9** | **1.24** |
| `unit_l2_norm` | 1−√(S₂/(S₂+ε)) ≈ ε/2S₂ | S₂ < ε/2a = **5.0e-10** (‖x‖₂ < 2.2e-5) | **1.18** |
| `unit_frobenius_norm` | ε/(√S₂+ε) | ‖x‖_F < ε(1−a)/a ≈ **1.0e-9** | **0.99** |
| `positive_scale_invariance` (c) | per-element out_j·E, E = (ε/S₁)(1−1/c) [l1], (ε/2S₂)(1−1/c²) [l2], (ε/√S₂)(1−1/c) [frob] | E > rtol + atol/max\|out\| — **fill-dependent**: sharper for peaked inputs | peaked: **0.99 / 0.99**; randn: **1.06 (l1) / 2.95 (l2)** |

Verification: `probes/norm_domains.py` sweeps the input scale in faithful
fp32 emulation and locates the largest failing scale. Five of six
boundaries land within 0.99–1.24× of the derivation. The one loose case —
l2 scale-invariance on dispersed randn fills — fails at ~3× the predicted
scale: the point-estimate for max|out| and the per-row S₂ fluctuation
(min-over-rows of a χ² with 128 dof) both push the same direction. The
FORMULA is validated by the peaked fills (0.99×); the dispersed-fill
prefactor is order-1 conservative, stated rather than tuned away.

**fp32 absorption cliff**: computed in fp32, `S + ε == S` exactly once
S ≥ 2²⁴ε = 1.678e-5, so above that the entire eps term vanishes and these
checks measure pure rounding (~1e-7 measured). The eps deviation is
*visible* only in the window below absorption and *binding* only below the
atol boundary — e.g. for unit_l2 at (64,128) randn, the check is invalid
below σ ≈ 2e-6, passes with shrinking margin up to σ ≈ 1e-4, and is
eps-free above σ ≈ 1e-3. On corpus-scale inputs (σ = 1) every one of these
checks is >10⁴ inside its domain — the boundary only matters for
adversarial small-scale fills, which is exactly where the search's
"invalid input" records live.

## 2. The softmax positivity domain

`tile_coverage_softmax_positivity` asserts every output column has some
entry > 0 (exp > 0 in exact math). In floating point, output entry
y_ij = exp(ℓ_ij − m_i)/L_i rounds to zero once

    m_i − ℓ_ij  >  B − ln L_i,   B = 150·ln2 = 103.97 (fp32, subnormals)
                                 B = 126·ln2 =  87.34 (fp32 under FTZ)
                                 B = 1075·ln2 = 745.1 (fp64)

so the check is **valid iff every column has at least one row within B of
that row's max**: max_j min_i (m_i − ℓ_ij) < B. Full-column-height patches
of value P on a fill with max F violate this for every unpatched column as
soon as P − F > B — on ANY correct implementation. The shipped
RANGE_LIMIT = 300 domain gate is a fitted proxy for this; the derived
boundary is ~88–104 (+ln L), i.e. the gate is ~3× loose on the safe side
for these records but would pass a range-150 input that can still legally
zero out columns. *Recommendation (not shipped here, fix-discipline): the
reference-failure classifier could treat a positivity failure as
domain-expected whenever the derived criterion holds on the materialized
input — that is a checkable condition, not a fitted constant.*

## 3. Adjudication of the 12 softmax records — all check-domain

`probes/softmax_positivity_adjudication.py`, NORM_ADJUDICATION standard:
each proposal materialized with the run's own materializer, the reference
kernel's arithmetic emulated in fp32, the SHIPPED check executed, then the
float64 ideal math through the same check. randn fills have no recorded
seed → 10-seed unanimity required (and achieved); zeros fills are exact.

**Verdict: 12/12 CHECK-DOMAIN FALSE ALARMS, 0 reference-implicated, 0
no-repro** (`data/softmax_adjudication.log`, per-record table). Every
record carries a full-column-height patch at +1e3/+1e4 on zeros/randn:
minimum deficit m−ℓ over unwritten columns is **995–10000**, i.e. ≥9.6×
beyond the fp32 boundary and ≥1.3× beyond even the **fp64** boundary — the
ideal math fails the check identically, 10/10 seeds where seeded. The
faithful emulation reproduces the recorded failure in every record.

One record (`e251c3aa`) additionally recorded `kernel_executed` failing:
that is the Layer-1 false positive that was diagnosed and fixed on
2026-08-21 (25/25 recorded FPs cleared on GPU); the record predates the
fix. Its positivity component is adjudicated here like the others.

## 4. The reference-suspect ledger after this round

Of the 45 reference-suspect verdicts surfaced on 2026-08-27: instancenorm
15 + layernorm 11 + rmsnorm 2 adjudicated in NORM_ADJUDICATION (27
check-domain + 1 real bug, since fixed); softmax 12 adjudicated here (all
check-domain); the 3 known July records were the flash/sdpa/layernorm
bugs, all fixed. **Remaining unadjudicated: the 2 matmul
`scalar_associativity` records** (plain randn inputs, one at (65,65)) —
explicitly out of this round's scope and still open; they are the only
survivors of the ledger. *(Update, later on 2026-08-28: both adjudicated
CHECK-DOMAIN — the documented pre-loosening tolerance FP, the (65,65)
hypothesis refuted; the ledger is now fully closed, 45/45. See
`../matmul_assoc_2026-08-28/FINDINGS.md`.)*

## 5. Family status

Derived validity domains now exist for: unit_variance/unit_rms eps-domain
(norm adjudication round), fp-absorption v > 21.5 and the cancellation-at-
shift domain (NUMERICAL_THEORY), rmsnorm precision-check eps placement
(NORM_ADJUDICATION §1), the l1/l2/frobenius family (§1 above), and softmax
positivity (§2). **Not derived** (and not claimed): pool/reduction
tie-breaking domains, matmul scalar_associativity (relevant to the two
open records), attention convex_hull_bound, and the affine/precision
layernorm checks beyond their measured margins — those are Item-5
territory (margin measurement, not domain derivation).

## Limits

- CPU fp32 emulation; the GPU kernels' tree-reduction order differs. All
  adjudication margins here are ≥9.6× (most 100×), far beyond
  reduction-order noise; the FTZ-vs-subnormal ambiguity (87 vs 104) is
  bracketed and does not affect any verdict.
- The scale-invariance dispersed-fill boundary is conservative by up to
  ~3× in scale (§1); the peaked-fill boundary is exact.
- Recorded failure summaries carry no per-column detail, so "reproduces
  the recorded failure" means check-level identity, the same granularity
  the norm adjudication used.

## Reproduce

```bash
.venv/bin/python probes/softmax_positivity_adjudication.py
.venv/bin/python probes/norm_domains.py
```
