# Method formalization round: the procedure is written down operator-agnostically, and it survived a pre-registered blind test on an operator and kernel it had never seen

**2026-08-27.** Two deliverables: `METHOD.md` (the operator-agnostic
procedure — the generalizable contribution, with the existing 51+ measured
operators as its validation set rather than its content) and the **blind
generalization test** reported here. GPU stage on the same Colab T4 session
as the Gram-screen arms (`kccgram`, stopped); CPU stages on the dev machine.
Probes in `probes/`, banked inputs/predictions/measurements in `data/`.

## The blind test, and why it is blind

Operator: **`logcumsumexp`** (rowwise log-prefix-sum-exp). Verified by grep
to appear in no spec, no reference kernel, no banked measurement, and no
prior derivation in this repository. Kernel under test: **ATen's shipped
CUDA implementation** (`torch.logcumsumexp`) — code this project did not
write and had never run. All predictions were computed in float64 on CPU
and banked (`data/blind_predictions.json`, `blind_inputs.npz`, including
the exact 40 deltas per configuration and their directional derivatives)
**before** the kernel was invoked; the GPU stage (`blind_measure.py`) draws
no randomness and the comparison (`blind_compare.py`) edits nothing. One
defect found and fixed in the *comparison script* after first run — a
units slip (dividing by std(x) instead of the delta std), caught because
the paired ratios were exactly 1.0 while y was off by exactly 1/delta_scale;
predictions and measurements were not touched.

## What step (a) alone produced, before any measurement

Differentiating the math definition (procedure step a):

- each Jacobian row is a **prefix softmax**, so row 0 is exactly `e_0` and
  no row norm can exceed 1 ⇒ **L = 1 exactly, input-independently** — a
  sharper structural statement than any per-input bound;
- the Gram matrix has **nested supports** (scan-like) with
  **input-dependent weights** (attention-like) — a family combination the
  corpus does not contain;
- step (b) then predicts the taxonomy class is **input-dependent**:
  m3/gram ≈ 1.00 at randn and sorted inputs (independent-row behaviour)
  but 1.22–1.30 at saturating x50 inputs (correlated-row, approaching the
  scans' 1.231) — membership is a property of `JJ^T(x)`, not of the
  operator name.

## Results (18 configs: 3 shapes × {randn, ×50, sorted} × 2 seeds)

From `data/blind_compare_out.txt`:

| criterion (pre-registered) | outcome |
|---|---|
| distributional Gram law: all \|z\| ≤ 3, family mean ≈ 0 | **PASS** — mean z = −0.13 (expected sd 0.24), worst \|z\| = 1.38 |
| paired screen: median \|log10 r\| < log10 2 at every config, *including* the saturating regime | **PASS** — median \|log10 r\| = 0.0000 at all 18; worst single delta 0.012 |
| classification: m3/meas > 1 wherever derived m3/gram > 1.05 | **PASS** — measured m3/meas 1.17–1.37 exactly on the six ×50 configs; 0.97–1.06 elsewhere |

The pre-registered *qualitative* prediction also held: a saturated
prefix-LSE degrades into a smooth running-max and **stays in scope**
(unlike attention's saturated softmax·V) — the shipped kernel measures
Jacobian-consistent to 3 decimal places even at ×50 inputs, while the same
procedure run on the corpus the same day (../gram_screen_2026-08-27/) shows
attention's `multi_tile_rescaling` deviating by 2.26× median. The method
distinguishes two operators that *both* "saturate a softmax", from
structure, before measurement.

## What this licenses, and what it does not

Licensed: the claim in METHOD.md §4 — for operators built from linear maps
and C¹ nonlinearities with known derivatives, steps (a)–(d) are mechanical
and transfer sight-unseen, including to a family combination (nested +
input-dependent Gram) not present in the validation corpus, against a
foreign kernel. The blind test exercised every step: structural L (found
an exact constant), Gram classification (found input-dependence and its
size), directional-derivative validation (paired ratios ≈ 1 to 4 decimals),
taxonomy filing (correlated-row *conditionally*).

Not licensed, unchanged from METHOD.md: closed forms for new families
(logcumsumexp got exact *simulation*, not a formula); non-C¹ points
(l1norm's kink measures 1.44× in the corpus run — detected, not modelled);
n = 1 operator in this blind test — one success is existence, not a rate;
and the m=1 diagnostic blindness is untouched by any of this.

## Reproduce

```bash
cd verification_runs/method_formalization_2026-08-27
PY=../../.venv/bin/python
$PY probes/blind_predict.py          # stage 1: derives + banks (CPU, f64)
# stage 2 on a T4: upload data/blind_inputs.npz, data/blind_predictions.json,
#   probes/blind_measure.py; run; download /content/blind_gpu.jsonl
python3 probes/blind_compare.py      # stage 3: scoring (no torch needed)
```
