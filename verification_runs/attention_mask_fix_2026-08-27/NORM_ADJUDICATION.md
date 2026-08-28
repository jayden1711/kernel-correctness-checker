# Norm-family adjudication — 27 of 28 are check-domain false alarms; 1 is a SECOND real padded-lane reference bug (layernorm, n_cols=333), caught live in July and filed as "invalid input"

**Adjudicated 2026-08-27, CPU emulation** at the attention-bug standard: per
record, the proposal's input is materialised exactly, the reference kernel's
own arithmetic and the ideal math are both emulated in fp32, and the SHIPPED
check functions are run through the RUN-ERA wrappers (commit 5277cd1/a4e8aa1
— the working-tree `_wrap_precision` fix postdates the runs). Probe
`probes/norm_adjudication.py`; per-record data `data/norm_adjudication.json`,
log `data/norm_adjudication.log`. **Nothing fixed in this pass.**

Adjudication criterion, per recorded failed check: the failure must reproduce
under the kernel-faithful emulation (validation), and then — if the **ideal
math also fails** the check on that input, the record is a *check-domain
false alarm* (the reference is not implicated); if the ideal math **passes**,
the failure is *reference-bug-caused*.

---

## Verdict up front

| | count |
|---|--:|
| records adjudicated | **28** (instancenorm 15, layernorm 11, rmsnorm 2) |
| confirmed check-domain false alarms (every recorded failed check reproduced, ideal fails too) | **27** |
| **reference-bug-caused** | **1** — `f322abe4`, layernorm (512, 333): a REAL bug in the layernorm reference kernel, §2 |
| BUG_CLASS_THEORY §4 denominators | **instancenorm 3/3 and rmsnorm 2/2 CONFIRMED** (verified, not assumed). **layernorm corrected**: 1 of 1 as recorded, but **1 of ≥4** under corrected tooling — §3 |

## 1. The 27 false alarms, by mechanism (per-record table in the JSON)

| mechanism | records | evidence |
|---|---|---|
| **eps-vs-variance / degenerate input** (the documented `run_random_baseline.py` condition): constant or var ≪ eps input makes `unit_variance`/`unit_rms`/`scale-invariance` fail on the *correct* operator (output var ≈ var/(var+eps) or ≡ 0) | instancenorm all 15 (`ones` fills → dev exactly 1.000; `randn×1e-4..1e-6` → dev 0.999); layernorm `f2fac9f6`, `16c0b6eb`, `a47a431b` (constant rows via patches, dev 1.000); layernorm `3b2a37a6`/`a7867d2c` uv (eps bias: dev 1.19e-3 and 0.107); rmsnorm `0eb28274` (RMS 1e-8: dev ≈ 1) | reproduced 10/10 seeds where random; ideal fails identically |
| **run-era `_wrap_precision` bug** (since fixed in the working tree): the wrapper passed the proposal's γ=2/β=3 into a check whose reference is the bare norm — fails on ANY correct kernel, margin 3000–7600× atol | layernorm `a7e94cf9`, `d8eb4716`, `35205e68`(pc), `3b2a37a6`(pc), `a7867d2c`(pc), `f2fac9f6`(pc); rmsnorm `dc510d61` | deterministic; **both records whose only failure was this (`a7e94cf9`, `d8eb4716`) pass 10/10 under the fixed wrapper** |
| **fp32 cancellation at extreme shift** (100–1000): mean-subtraction error at ulp(shift) scale defeats atols of 1e-3/1e-4 on the correct math | layernorm `2d7f4f3e`, `15b63912`, `35205e68`(ac) | ideal fails alongside faithful |
| **the rmsnorm precision check's own eps placement** (`ref = x/(RMS + 1e-5)` vs the kernel's `x/√(RMS²+1e-5)` — 316× divisor difference in the degenerate regime) | rmsnorm `0eb28274`(pc) | ideal fails identically |

CPU-fidelity caveats, stated rather than hidden: `2d7f4f3e`/`15b63912`
reproduce all their *recorded* failures 10/10 but the CPU emulation
additionally fails `precision_coercion` (which the GPU run recorded as
passing) — a reduction-order difference at shift 1000 in a check *outside*
the recorded set; the recorded-check adjudication carries 100–200× margins
and is unaffected. `35205e68`'s `affine_correctness` margin is only 2.7×
atol (exact failed-set match 5/10 seeds) — lower confidence on that single
check, but ideal fails it too on every seed, so the classification does not
depend on the margin.

## 2. The one real bug — layernorm's padded-lane variance

> **FIXED 2026-08-28** — the mask shipped and passed all four regression
> criteria plus the oob-round no-movement prediction; see
> `../layernorm_mask_fix_2026-08-28/FINDINGS.md`.

`TritonBench/reference/layernorm.py` computes, per row loaded with
`other=0.0` into a `next_power_of_2(n_cols)` block:

```
diff = row - mean                      # NO mask — padded lanes become -mean
variance = tl.sum(diff * diff) / n_cols
```

Every padded lane adds `mean²` to the variance sum. At (512, **333**) the
block is 512, so 179 padded lanes inflate the variance by `179·mean²/333 ≈
0.54·mean²` — for `f322abe4`'s in-contract input (randn, shift 10) that is
~54× the true variance, shrinking the output ~7.4×. Measured (10/10 seeds):
`unit_variance` dev 0.986 and `affine_correctness` max_err 7.85 under the
kernel's arithmetic, while the **ideal math passes both cleanly**. The
contrast that clinches it: **`instancenorm.py` masks this exact term**
(`diff = tl.where(mask, row - mean, 0.0)`) and `rmsnorm.py` is safe by
construction (0² = 0) — the layernorm kernel alone omits the mask.

This is the **second member of the padded-lane bug family** (after
flash/sdpa's softmax-denominator leak), again live only when the reduced
width is not a power of two, again detected by the checker on 2026-07-23 and
filed as "invalid input", and again sitting inside the shipped spec:
layernorm's `valid_shapes` includes **(1000, 333)**, so the cross-shape sweep
exercises the buggy width (reference-vs-itself, so recorded corpus verdicts
are structurally unaffected — the same argument as the attention impact
round, to be *verified* in the dedicated investigation, not assumed).

**FLAGGED, NOT FIXED** — per the same discipline as the attention bug, the
one-line fix (`diff = tl.where(mask, row - mean, 0.0)`) needs its own change
with a blast-radius check first (corpus verdicts at (1000,333); whether any
theory-round layernorm number used a non-pow2 width — the corpus layernorm
shape is (64,128), pow2, so the sandwich/M3/L tables are expected clean;
groupnorm/batchnorm/l1norm/l2norm/frobenius and every other reduction kernel
should be swept for the same unmasked-lane pattern while at it).

## 3. What this does to BUG_CLASS_THEORY §4

- **instancenorm "3 of 3" — CONFIRMED, now verified rather than assumed.**
  All 15 "invalid" records are true check-domain artifacts; no valid
  proposal was hidden.
- **rmsnorm "2 of 2" — CONFIRMED.** Both flagged records adjudicated
  check-domain.
- **layernorm "1 of 1" — CORRECTED.** As a statement about the run-era
  checker's labels it stands, but three of the eleven "invalid" proposals
  were genuinely valid inputs mislabeled by since-identified bugs:
  `f322abe4` by the **reference-kernel bug** above, and `a7e94cf9` +
  `d8eb4716` solely by the **since-fixed precision-wrapper bug** (both pass
  every check 10/10 under the fixed wrapper). The honest count is **hit on
  1 of ≥4 valid proposals**; whether the three mislabeled ones would have
  been hits is **not determined** (no counterfactual mutant/gap analysis was
  run — same boundary the flash counterfactual respected).

§4 of BUG_CLASS_THEORY.md now carries a dated correction box saying exactly
this.

## 4. Reproduce

```bash
PY=.venv/bin/python
$PY verification_runs/attention_mask_fix_2026-08-27/probes/norm_adjudication.py
# the post-fix wrapper check on a7e94cf9/d8eb4716 is appended in
# data/norm_adjudication.log
```

Limits: fp32 CPU emulation vs the T4's `tl.sum` reduction order — margins are
reported per record and every classification rests on ≥2.7× margins (median
far larger), with the two knife-edge/extra-failure cases called out in §1;
randn fills have no recorded seeds (10-seed unanimity required and achieved);
`f322abe4`'s counterfactual search outcome is out of scope.
