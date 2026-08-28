# Taxonomy third exclusion flavour + two deliberate deferrals — 2026-08-28

**Item:** the two worth-doing piggybacks from theory-audit ranked item 5.
Documentation-only round; no measurement.

## 1. cumprod exclusion flavour added to METHOD.md

`verification_runs/method_formalization_2026-08-27/METHOD.md` §1(d) now
distinguishes three mechanically different causes of structural exclusion,
where the taxonomy previously had only one:

- **(i) `J = 0` a.e.** — argmax/argmin (the original class-4 signature);
  remedy: exact-match checks.
- **(ii) non-C¹ kink mass** — the l1norm boundary; excluded above kink
  measure p ≳ 0.59 per the theory-closure bound; remedy: exclusion, never a
  threshold retune.
- **(iii) unbounded conditioning** — cumprod (`J_ij = ∏_{k≤i,k≠j} x_k`,
  CORPUS_EXPANSION_PLAN L1 #90): `J` is smooth and nonzero, but its row
  norms have unbounded input-dependent condition number, so no useful `L`
  exists and the sensitivity estimate has no stable population parameter.
  Excluded because the *tolerance functional* is ill-posed, not the
  derivative; remedy: per-input exact-J evaluation or property checks only.

§7 step 2 was extended to route flavour (iii) at derivation time (and to
record the flavour when filing an exclusion).

## 2. Deliberately deferred — recorded, not dropped

Appended to METHOD.md's tail (dated):

- **RoPE second blind test** — deferred unless the method is written up
  externally. The passed logcumsumexp blind test is the standing
  generalisation evidence; a second blind operator strengthens a publication
  claim but changes no in-repo decision.
- **Scan-family ~−2% signed residual** — deferred unless the scan family is
  re-measured. It is inside single-draw noise for every banked decision and
  only diagnosable with fresh multi-draw data (theory_closure §1's direct
  parent already bounds it at ≤1.9%).
