# Layernorm padded-lane blast radius — every reported number is bitwise invariant except one ablation statistic (a bug-manufactured cross_shape catch); project-wide sweep: the bug family ends at three kernels

**Investigated 2026-08-27, CPU emulation only** (validated against banked
records; the affected quantities are structurally decidable — see §2).
Probes in `probes/`, logs in `data/`. **Nothing fixed in this pass**, per the
established impact-first discipline. *(Update 2026-08-28: the fix has now
shipped and every §4 regression criterion passed —
`../layernorm_mask_fix_2026-08-28/FINDINGS.md`.)* The bug: `TritonBench/reference/
layernorm.py` computes `diff = row − mean` without masking padded lanes, so
each of the `next_pow2(n) − n` lanes adds `mean²` to the variance sum —
live only at non-power-of-two widths (adjudicated in
`../attention_mask_fix_2026-08-27/NORM_ADJUDICATION.md` §2).

---

## Verdict up front

| question | answer |
|---|---|
| Any reported theory number affected? | **No — bitwise.** Every layernorm width used by any theory/benchmark measurement is a power of two ({128, 512, 1024}); at pow2 widths the pad term is structurally zero and the buggy and corrected kernels are **bitwise identical** (asserted in emulation, `torch.equal`, 4 shapes). |
| Any recorded corpus verdict affected? | **No.** 40/40 catch and 0/200 FP stand: reference trials at (1000,333) are reference-vs-itself; all three layernorm mutants stay caught (two by L2 checks at the pow2 base shape; `wrong_variance_estimate` by its dedicated adversarial trigger). |
| Anything at all affected? | **Yes — one check-level ablation statistic.** The recorded `cross_shape` catch of `layernorm/wrong_variance_estimate` fails **only** at (1000,333) (banked max_err 0.0249) and is **bug-manufactured**: emulated 10/10 fail vs the buggy reference (0.023–0.039) and 10/10 **pass vs the corrected reference at 5–7e-7, 200× inside atol**. Under a corrected reference, `cross_shape`'s catch count drops by one (28/40 → 27/40 in CHECK_ABLATION.md; the "29/40 (72%)" quoted in CHECK_ABLATION_FINDINGS likewise −1), the `cross_shape[shape=(1000,333)]` sub-row loses its one-extra catch, and the pairwise-subsumption rows involving `cross_shape` shift by this mutant. Dated notes added to both docs. Headline claims (100% catch, layer attribution) unaffected — both of this mutant's catchers are numeric-layer. |
| f322abe4 counterfactual | Closed, honestly: **indeterminate.** Under kernel-fix-only, the run-era wrapper bug keeps the reference "invalid" → recorded trajectory unchanged. Under full correction (kernel + wrapper), the reference is valid 10/10 but a hit occurs in only **3/10 seed draws** — via `wrong_variance` being caught by a knife-edge fp-cancellation margin on the affine check while passing naive — and the actual July input draw is not recoverable. |
| Part 2 sweep | **The bug family ends at three kernels.** All 64 operators' kernels audited (29 corpus + 27 Phase-1 + 8 conv): flash_attention and sdpa (fixed 2026-08-27), layernorm (this round). Every other reduction kernel either uses the reduction-appropriate sentinel or masks explicitly — §3. |

## 1. Part 1 — enumeration and per-item disposition

Layernorm width inventory across every banked dataset:

| dataset / claim | width(s) | pow2? | disposition |
|---|---|---|---|
| Theory rounds — sandwich table, closed-form L, M3 (gpu_native, gen_native, replay, matched fits) | 128 (corpus (64,128)) | ✓ | **bitwise invariant** |
| adaptive_tol probes (19-config linearisation, sandwich §3; structural_l regime probe) | 512 ((512,512)) | ✓ | bitwise invariant |
| NUMERICAL_THEORY / BUG_CLASS_THEORY layernorm derivations | 512 | ✓ | bitwise invariant |
| Corpus benchmark runs (CURVE + 6 VALID + scope + timing arms, autokernel results, kernel_executed/delegation raws) — base inputs + L2/L3 batteries | 128 | ✓ | bitwise invariant |
| — their `cross_shape` sweep | 512, 1024, 512, **333**, 128 | ✗ at 333 | refs: self-comparison, invariant. Mutants: see §2 |
| Adversarial search (12 layernorm proposals) | 512, **333**, 64 | ✗ at 333 | adjudicated in NORM_ADJUDICATION; counterfactual closed above |
| KernelBench corpus (40_LayerNorm) | 128 | ✓ | bitwise invariant |

The structural argument doing the heavy lifting, verified rather than
assumed: at `n_cols` a power of two, `BLOCK == n_cols`, there are zero padded
lanes, and the buggy arithmetic is the **same instructions** as the corrected
one — `torch.equal` asserted across the four pow2 spec shapes
(`probes/ln_blast.py`). The only non-pow2 layernorm width anywhere in the
project's recorded history is **333**.

## 2. The one affected recorded outcome, and its reported dependents

`wrong_variance_estimate` computes one-pass `E[x²] − E[x]²` over 0-padded
loads — **it does not share the reference's padded-lane term** (padded lanes
contribute 0 to both its sums), while being algebraically identical to the
correct variance. So at (1000,333):

- vs the **buggy** reference: disagreement 0.023–0.039 (banked 0.0249 —
  emulation validated in-range), > atol 1e-4 → sub fails → `cross_shape`
  records a catch;
- vs the **corrected** reference: disagreement 4.8–7.2e-7 (pure fp
  cancellation of the one-pass form) → **passes with 200× margin**, 10/10
  seeds.

The recorded catch is therefore the reference's bug detecting itself against
the one mutant that doesn't share it. Reported numbers that count it:

| reported number | where | under corrected reference |
|---|---|---|
| `cross_shape` 40-ran / 28-caught (70%) | CHECK_ABLATION.md main table | 27 (67.5%) |
| "`cross_shape` 29/40 (72%)" | CHECK_ABLATION_FINDINGS.md | 28/40 |
| `cross_shape[shape=(1000, 333)]` 10/17 | CHECK_ABLATION.md sub-table | 9/17 (in line with the other four shapes' 9) |
| pairwise subsumption rows involving `cross_shape` (e.g. vs `adversarial_wrong_variance_trigger`, "partial overlap 26\|2\|1") | CHECK_ABLATION.md | shift by this one mutant |

**No paper-facing headline depends on any of these**: the mutant remains
caught (`adversarial_wrong_variance_trigger`), 100%-catch/0%-FP stand, and
layer-level attribution is unchanged (both catchers are numeric-layer).
Dated correction notes added to both ablation docs pointing here.

## 3. Part 2 — the project-wide unmasked-lane sweep

Every kernel audited at source for how out-of-bounds/padded lanes enter its
reductions (`data/` logs carry the extracts):

| kernel(s) | lane handling | verdict |
|---|---|---|
| softmax, log_softmax, cross_entropy | `other=-inf` → `exp(pad − max) = 0` | **safe** |
| sum/mean_reduction, l1norm, l2norm, frobenius_norm | `other=0.0`, summand vanishes at 0 (`x`, `\|x\|`, `x²`) | safe |
| max/min_reduction, argmax, argmin, max_pool1/2/3d | `∓inf` sentinels (+ tie logic) | safe |
| avg_pool1/2/3d | `other=0.0`, divisor = kernel size (documented count_include_pad semantics) | safe |
| matmul, matvec, batched/diagonal/triangular matmul, conv family (8) | zero-padding annihilates in multiply-accumulate | safe |
| batchnorm, gelu, swish, activations (9), rope, swiglu | elementwise / per-element stats | safe |
| **groupnorm, instancenorm** | `diff = tl.where(mask, row − mean, 0.0)` — **explicitly masked** | safe |
| **std/var_reduction, logsumexp, losses (Phase 1)** | `tl.where(m, …, 0.0)` before every sum — explicitly masked | safe |
| scans (cumsum ×4) | `tl.cumsum` over 0-padded tail, stores masked | safe |
| rmsnorm | `x²` at 0-pads = 0, safe **by construction** | safe |
| flash_attention, scaled_dot_product_attention | unmasked `S` — **was the bug; fixed 2026-08-27** | fixed |
| **layernorm** | `diff = row − mean` **unmasked** | **THE remaining instance — flagged, unfixed** |

Two observations worth recording: (i) the pattern is 3-for-64, and the three
instances are the only kernels where the reduced quantity is a *function of
the loaded value that is nonzero at the sentinel* (`exp(0−m)`, `(0−mean)²`)
— every kernel where the sentinel annihilates naturally is clean, and every
kernel where it doesn't except layernorm carries an explicit mask; (ii) this
also **rules out the padded-lane class for the 12 still-unadjudicated softmax
search flags** — the softmax kernel is `-inf`-masked, and its flagged shapes
include pow2 (512,2048) anyway. Those remain open with a different mechanism
(`tile_coverage_softmax_positivity`, likely output-underflow at range ≥ 1e3
patches vs a strict-positivity check — unadjudicated, per the standing
follow-up list).

## 4. Escalation and the fix's regression criteria

No claim dependency forces urgency, but the fix is now fully scoped and
cheap (`diff = tl.where(mask, row - mean, 0.0)`, the groupnorm/instancenorm
pattern). When shipped, its regression evidence should include: (a) 40/200
corpus verdict + catch-attribution diff (expect exactly one attribution
change: `wrong_variance_estimate` loses `[L3]cross_shape`, keeps its trigger
catch — assert this, don't assume 0 diffs like the attention fix); (b)
`cross_shape` sub-outcomes at (1000,333) for a correct candidate; (c) pow2
bitwise-identity spot-check; (d) updated CHECK_ABLATION numbers or a
superseding note.

## 5. Reproduce

```bash
PY=.venv/bin/python
$PY verification_runs/layernorm_mask_bug_2026-08-27/probes/ln_blast.py   # E1/E2/E3
# data/curve_layernorm.log — banked CURVE evidence (catch attribution, per-shape subs)
# kernel-source sweep extracts: grep commands recorded in data/sweep_notes.md
```

## 6. Limits

- The E1/E3 emulations use fresh corpus-distribution draws (cross_shape's
  GPU inputs are device-RNG and not reproducible); margins are 200×+ on both
  sides of the adjudication, far beyond CPU/GPU reduction-order noise.
- f322abe4's counterfactual is seed-indeterminate (3/10) and is reported as
  such, not resolved.
- The CHECK_ABLATION corrections are derived (−1 catch), not re-run; the doc
  notes say so. Re-running the ablation belongs with the fix.
- The sweep is a source audit plus construction arguments; it did not
  execute all 64 kernels at non-pow2 shapes. The three positives were each
  confirmed by emulation against banked GPU behaviour; the negatives rest on
  the sentinel-annihilation/explicit-mask patterns being visible in source.
