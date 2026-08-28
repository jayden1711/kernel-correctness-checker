# Item 1d — corpus regression for the interleaved best-of-N timing fix

**Colab T4, 2026-08-21.** `run_benchmark.py`, full corpus, all 11 systems.
Baseline: the repo's banked `benchmarks/autokernel/files/results_raw.json`
(pre-fix). Post-fix output: `results_raw_POSTFIX.json` in this directory.

## Result

| | compared | differing |
|---|---:|---:|
| **mutant verdicts** (catch / no-catch) | **440** | **0** |
| **reference verdicts** (false positive / not) | **2200** | **5** |

**All four `your_checker` systems — `full`, `structural only`, `numeric only`,
`algebraic only` — show 0 differences on both mutants and references.** The 1d
fix lives in `check_kernel_executed`, which is `your_checker`'s Layer 1, so that
is the population the claim is about, and it is unchanged.

## The 5 reference differences are the documented `frobenius_norm` flake

Every one is `frobenius_norm`, and every one is in an `autokernel_gate` variant:

```
  autokernel_gate                     frobenius_norm  FP True  -> False
  autokernel_gate (faithful)          frobenius_norm  FP False -> True
  autokernel_gate (faithful)          frobenius_norm  FP True  -> False
  autokernel_gate (faithful, rtol=0)  frobenius_norm  FP False -> True
  autokernel_gate (faithful, rtol=0)  frobenius_norm  FP True  -> False
```

Two things make this attributable to the known flake rather than to the fix:

1. **They flip in BOTH directions** (3 lost, 2 gained) — `SESSION_HANDOFF.md` §3
   states the signature explicitly: "expect `frobenius_norm` determinism rows to
   differ in both directions and treat that as noise. Real regressions do not
   flip back and forth."
2. **The `autokernel_gate` baselines do not call `check_kernel_executed` at
   all** — they are independent re-implementations with their own bitwise
   determinism stage. `frobenius_norm`'s reference uses `tl.atomic_add`, whose
   non-associative float accumulation makes any bitwise comparison flip run to
   run. This project's own checker is immune (it uses a tolerance, not
   `torch.equal`) and shows **zero** determinism FPs, as before.

So the fix is verdict-neutral on the corpus, and the residual differences are in
baselines it cannot reach.
