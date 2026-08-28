# Attention under the Gram-matrix law — the law holds everywhere, the "+17% correlation residual" was mostly sampling noise, and the falsification run found a real bug in two shipped reference kernels

**Measured 2026-08-27.** CPU stage on the Apple-silicon dev machine against the
banked 2026-08-25 T4 measurements; GPU stage on a fresh Colab T4 (session
`attngram`, provisioned and stopped; `torch 2.11.0+cu128`, `triton 3.6.0`).
Probes in `probes/`, data + run logs in `data/` (banked Q/K/V:
`qkv_corpus.npz`, `attn_gram_qkv.npz`). **Nothing in the checker or the
kernels was changed.** This closes the extension flagged in
`../../theory_audit_2026-08-27/FINDINGS.md` §1.5.

---

## Verdict up front

| question | answer |
|---|---|
| Does the Gram-matrix law (theory_audit H1) extend to attention? | **YES — and more strongly than to scans**, because here it required an exact per-invocation Jacobian, not a family-level closed form. 36 banked-corpus invocations: mean z = **+0.13 ± 0.17**, worst \|z\| = 2.01, medians 1.0000 / 0.9985 / 0.9848 per op. |
| Does it explain the +8–17% M3 residual quantitatively? | **It explains it away.** The structural (correlation) part is only **+3.3% median (max +7%)** — m3/gram 1.004–1.070. The quoted "+17%, +10%, +9%, +8%" reproduce exactly (m3/meas = 1.166, 1.104, 1.075, 1.096) and sit at z = −1.66, −0.28, −0.24, −0.73 under the law: **the tail of single-40-sample-draw noise around a small true correction, not a large systematic.** |
| Out-of-sample falsification (new GPU run, 3 shapes never measured) | **2 of 3 shapes clean** (128×64 mean z −0.13; 256×16 mean z +0.00). The third, 100×32, deviated −10% for flash/sdpa only — and the deviation is **fully attributed** (below): the kernels compute a different function there. Re-predicted from the kernel-faithful Jacobian: all 9 points \|z\| ≤ 1.16. |
| Scale invariance | **Worst y spread across delta_scale ∈ {1e-4, 1e-3, 1e-2} = 1.05%** over 36 triples (same deltas reused — the attention analogue of the scan four-decimal test). |
| By-product | **A real bug in two shipped reference kernels** (`flash_attention`, `scaled_dot_product_attention`): when `N % BLOCK_N ≠ 0`, padded key columns enter the softmax denominator. Silent, up to **97% output error at N=1**, and present at shapes the specs themselves list. §4 |

---

## 1. Method

The perturbation is applied to Q only, and output row `i` depends only on
`Q_i`, so the exact first-order law is block-structured:
`s/σ = max_{i,d} |⟨J_{(i,d)}, z_i⟩|` with one iid `z_i ~ N(0, I_D)` per row
and the full within-row `D_v×D_v` Gram — the shared-softmax-denominator
correlation M3 discards. Unlike the scans (exactly linear, shape-only Gram),
attention's Gram is input-dependent, so every invocation gets its **own exact
Jacobian** (`torch.autograd.functional.jacobian`, float64) at the corpus's own
replayed inputs — the numpy `default_rng(0)` stream reproduces the T4 round's
inputs bit-for-bit (verified: replayed σ matches all 36 banked σ to ≤ 1.2e-7
relative). Zero fitted constants anywhere; `y_pred = E[q95_40(max|Jz|)]/L`
with `L` the exact max row norm.

Harness anchors: the M3-orthogonal simulator reproduces the banked
`gen_native` y_M3 to 0.3/0.4/2% (causal/sdpa/flash matching entries), and the
banked K=400 MC `L` sits at 0.95–1.07× the exact `L` (shared-draw estimator,
CV 3.5%, no selection bias — consistent).

## 2. Banked-corpus stage (36 invocations, T4 measurements of 2026-08-25)

| op | n | y_meas/y_pred median [range] | worst \|z\| | m3/gram median | m3/meas median |
|---|--:|---|--:|--:|--:|
| `causal_flash_attention` | 6 | 1.0000 [0.9155, 1.1091] | 1.33 | 1.039 | 1.035 |
| `flash_attention` | 24 | 0.9985 [0.8098, 1.1997] | 2.01 | 1.031 | 1.024 |
| `scaled_dot_product_attention` | 6 | 0.9848 [0.8911, 1.1834] | 1.68 | 1.026 | 1.058 |

Mean z over all 36 = **+0.13** against an expected sd of 0.17 — no systematic.
The spread of single-invocation y (±8%, one 40-sample draw each) is exactly
what the law predicts, and it is the whole story of the old "+8–17%" numbers:
`generalization/FINDINGS.md` §B.2's four largest deviations were four single
draws from the j=0 slice. **The real correlation correction for this family is
+3–4%** — attention rows share a softmax denominator, but each row couples only
its own `D_v = 32` outputs out of `m = 2048`, so the effective-m reduction is
mild. (The scans, whose rows are globally nested, genuinely carry +24.7%.)

## 3. Out-of-sample GPU stage — 108 measurements, predictions pre-registered

Fresh T4 run: 3 ops × shapes {(64,32) anchor, (128,64), (100,32), (256,16)} ×
3 seeds × delta_scale {1e-4, 1e-3, 1e-2}; inputs drawn from deterministic
numpy seeds and banked (`attn_gram_qkv.npz`); CPU predictions computed from
the same inputs and banked before comparison (`attn_gram_predictions.json`).

| shape | status | n | y_meas/y_pred median | mean z |
|---|---|--:|--:|--:|
| 64×32 | corpus anchor | 27 | 1.0164 | +0.23 |
| 128×64 | out-of-sample | 27 | 0.9957 | −0.13 |
| 256×16 | out-of-sample | 27 | 1.0132 | +0.00 |
| **100×32** | out-of-sample | 27 | **0.8930** | **−1.03** |

Scale invariance: worst y spread across the three delta_scale arms (same
generator seed ⇒ same delta directions) = **1.053%** over 36 triples.

**The 100×32 deviation split by operator is the attribution:** flash mean
z −1.41, sdpa −1.20, **causal −0.49 (clean, like its other shapes)**. 100 is
the only tested N with `N % BLOCK_N ≠ 0`.

## 4. The attribution — and the kernel bug it uncovered

Source inspection of the shipped reference kernels: K/V loads are masked to
zero for padded kv positions, **but `S` is never masked**, so every padded
column contributes `exp(0 − m)` to the online-softmax denominator.
`causal_flash_attention` is immune by accident: its causal mask
`q_idx ≥ kv_idx` excludes padded columns because every padded `kv_idx`
exceeds every valid `q_idx`.

Verified, not just argued (`probes/attn_padded_confirm.py`):

- **(a) The GPU computed the emulated buggy function, bitwise.** At all nine
  100×32 inputs, a CPU emulation with padded columns in the denominator
  matches the banked GPU `out_max` to ≤ 6.3e-7 relative, while the true
  softmax-attention differs by **8.7–15.9%** (flash/sdpa). Causal matches
  both.
- **(b) The law is not the deviant — the function is.** Re-predicting y from
  the **kernel-faithful** Jacobian restores every point: 9/9 with \|z\| ≤ 1.16.

So the Gram law survives its falsification run intact, and the run's one
"failure" was the measurement apparatus: **`flash_attention` and
`scaled_dot_product_attention` reference kernels silently compute the wrong
function whenever `N` is not a multiple of `BLOCK_N = 32`.** Measured error of
kernel vs true attention on random inputs:

| shape | flash | sdpa | note |
|---|--:|--:|---|
| (1, 64) | **96.6%** | **96.9%** | `scaled_dot_product_attention` spec lists (1, 64) in `valid_shapes` |
| (65, 64) | 22.1% | — | `flash_attention` spec lists (65, 64) — the shape added to expose the `drop_last_tile` **mutant** puts the **reference** in its own buggy regime |
| (100, 32) | 12.5% | 15.7% | this round's out-of-sample shape |
| (333, 64) | 2.2% | 2.0% | sdpa spec lists (333, 64) |
| (192, 64) | 0.000% | — | control: multiple of 32, exactly clean |

**Every prior banked measurement is unaffected** — the 2026-08-25 corpus ran
attention only at N = 64, and this round's other shapes are multiples of
32/16. But the specs' own `cross_shape` lists include (1, 64), (65, 64) and
(333, 64): any checker run that exercises those shapes compares candidates
against wrong ground truth, and a **correct** candidate would differ from the
reference by up to ~97% at N=1. Whether any recorded benchmark verdict was
actually affected is not established here and is the follow-up to do before
fixing. **RESOLVED 2026-08-27:
`../../attention_mask_bug_impact_2026-08-27/FINDINGS.md` — no recorded
verdict or reported number changes; the checker had in fact caught this bug
live three times in the 2026-07-23 flash search run, where it was booked as
"invalid input".**

**FLAGGED, NOT FIXED, deliberately:** the fix (mask `S` for `kv_offsets ≥ N`,
one `tl.where` in each kernel) changes reference-kernel behaviour and hence
potentially every downstream verdict at those shapes; per this project's
rules it needs its own change with before/after evidence, not a rider on a
theory round.

> **FIXED 2026-08-27** — `verification_runs/attention_mask_fix_2026-08-27/`.
> Correctness ≤ 9.1e-7 at all previously-affected shapes; full 40/200 corpus
> regression with **zero** verdict or catch-attribution changes; the 100×32
> measurements re-taken here land on the math-Jacobian Gram predictions
> (flash 0.895 → 0.977 median meas/pred) with causal bitwise unchanged.

## 5. What this changes in prior documents

- `generalization/FINDINGS.md` §B.2's reading of the four flash deviations as
  "rows sharing a softmax denominator" over-prediction: the mechanism is real
  but its magnitude is +3–4%, not +8–17%; the quoted numbers were single-draw
  noise. Correction note added there.
- `phase1_derivations_2026-08-27/GPU_NATIVE.md` §4 cites "attention's +17%" as
  the previous worst case of the correlation mechanism; note added pointing
  here (the corrected ordering is: scans +24.7% structural, attention +3–4%
  structural, conv +1%, matmul ~0).
- `theory_audit_2026-08-27/FINDINGS.md` §1.5's flagged extension is closed;
  note added.

## 6. Reproduce

```bash
cd verification_runs/adaptive_tol_theory_2026-08-25/attention_gram
PY=../../../.venv/bin/python
$PY probes/attn_gram_cpu.py       # replay corpus inputs, exact J, banked-vs-law (36 inv)
$PY probes/attn_gram_predict.py   # pre-registered predictions for the GPU shapes

export HOME=~/.colab-home
colab new --gpu T4 -s attngram
colab upload -s attngram TritonBench/reference/flash_attention.py /content/flash_attention.py
colab upload -s attngram TritonBench/reference/causal_flash_attention.py /content/causal_flash_attention.py
colab upload -s attngram TritonBench/reference/scaled_dot_product_attention.py /content/scaled_dot_product_attention.py
colab exec -s attngram -f probes/attn_gram_gpu.py --timeout 900     # 108 measurements, ~3 min
colab download -s attngram /content/attn_gram_gpu.jsonl data/attn_gram_gpu.jsonl
colab download -s attngram /content/attn_gram_qkv.npz data/attn_gram_qkv.npz
colab stop -s attngram

$PY probes/attn_gram_compare.py       # F1/F2/F3 scoring
$PY probes/attn_padded_confirm.py     # bitwise attribution + kernel-bug confirmation
```

## 7. Limits

- The exact-Jacobian route costs one autograd Jacobian per invocation (fine at
  m ≤ 8192; not a closed form). No attention closed form is claimed — the
  scan family remains the only closed-form case.
- The corpus-stage comparison inherits the banked round's single-40-sample
  measurement per invocation; per-invocation z is the right unit and no
  invocation exceeds \|z\| 2.01 of 36 (expected max ≈ 2.2).
- Curvature is covered only to the linearisation defect (banked ≤ 0.3% on
  ordinary inputs; adversarial/saturating attention inputs remain excluded
  exactly as in the original round).
- The kernel-bug magnitude table is CPU-emulated on random inputs (the
  100×32 entries are GPU-bitwise-verified); (1,64)/(65,64)/(333,64) were not
  re-measured on the T4, and no claim is made about which recorded verdicts,
  if any, were affected.
