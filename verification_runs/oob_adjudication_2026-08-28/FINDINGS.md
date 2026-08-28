# OOB adjudication: a spec-construction artifact, proven at the byte level — the kernels compute their spec exactly on valid inputs, and the variant feeds them invalid ones

**Adjudicated 2026-08-28** on a Colab T4 (session `kccoob`, provisioned and
stopped; `torch 2.11.0+cu128`, `triton 3.6.0`) plus local queries over every
banked arm and the search DBs. Probes in `probes/`, evidence in `data/`.
**Nothing was fixed**: per house rules the fix is specified (§5) and deferred
to its own change with the regression criteria below. This closes the flag
raised in `../theory_closure_2026-08-28/FINDINGS.md` §3(b).

## Verdict up front

| question | answer |
|---|---|
| Real kernel bug or spec/harness artifact? | **Spec/harness artifact.** `_non_power_of_two` in `verification/specs/layernorm.py` and `rmsnorm.py` hardcodes width 333 and re-wraps the **captured** gamma/beta — constructing an input tuple that violates the operator's own shape contract (`gamma: (n_cols,)`, stated in the spec's docstring) whenever the harness's base width < 333. On valid-length companions the kernels reproduce their own arithmetic to fp32 precision (≤ 1e-6); the mathematical function simply does not exist for `len(gamma) = 128, n_cols = 333`, so there is no "ideal math that passes where the kernel fails" — the NORM_ADJUDICATION question resolves against the spec construction. §2 |
| Is the OOB read real? | **Yes — proven at the byte level, not inferred.** Recovered per-column effective gamma/beta from the kernel output match the *contents of adjacent allocations*, mapped by pointer arithmetic on actual `data_ptr()`s: 205/205 leaked columns land in named neighbor tensors, values agree to ~1e-8; freeing a neighbor and re-allocating different contents at the same address changes exactly the leaked columns while in-bounds columns stay bit-identical. The output depends on the contents of unrelated allocations — which no function of the op's inputs can. §1 |
| Which shapes trigger it? | Base width < 333 only: leaked columns = 333 − len(gamma) (205 at the autokernel corpus's width 128; 0 at the spec corpus's 512/1024-wide primaries and at the (1000, 333) cross-shape, which builds its own consistent companions). The spec's own `valid_shapes` contains one latent trigger, (2048, 128), reachable if any harness ever uses it as the primary. §3 |
| Blast radius (independent recount) | **Confirms the closure round's finding, more broadly**: zero catches attributed to `adversarial_non_power_of_two` in ANY bank (n_samples 7 arms, scope 4, gram 2, check_timing — 288 layernorm + 270 rmsnorm affected records total); no scope-round margin minimum is attained on these records; the gram round never evaluated them (fail-open, already annotated); of 17 layernorm/rmsnorm search proposals exactly one ran in the exposure window ((64,64)) and its recorded verdict (`Reference failed: unit_variance`, hits gap-confirmed elsewhere) is independent of the variant. **No recorded verdict, catch, FP, or reported number depends on the OOB lanes.** §4 |
| Third instance of the non-pow2 family — common root cause? | **They rhyme; they are not one bug.** The two kernel-side instances (flash/sdpa softmax denominator, layernorm variance) share a real mechanism — a padding sentinel that is not annihilating under the downstream op — and that family was already swept to closure (exactly 3 of 64 kernels). This third is harness-side with a different mechanism (a width assumption in variant construction). The only true common cause is a coverage property: non-pow2 widths are the corpus's least-exercised region, and all three were found by measurement instruments, not by the test suite. One systemic fix does exist for the harness side — a spec-contract validation — and is specified in §5. §6 |
| compute-sanitizer | **Non-probative in this environment and documented as such**: it reports 0 errors even on a positive-control Triton kernel that unconditionally reads 205 floats past a 128-float allocation (cached AND uncached allocator). The adjudication rests on the value-level evidence, which needs no instrumentation. This also explains why the corpus never faulted. `data/sanitizer.log` |

## 1. The byte-level proof (`probes/oob_gpu.py`, stage A/C)

At the exact corpus configuration — `x (64, 333)` fp32, `gamma/beta (128,)`
allocated in tight sequence with named poison blocks after them:

- **Method validation**: recovered per-column (gamma, beta) on the in-bounds
  columns j < 128 match the true tensors to 4.6e-8 / 4.2e-8 (least squares
  against the kernel-faithful normalized values, 64 rows per column).
- **The leak, mapped by address**: for every leaked column j ∈ [128, 333),
  the float at `gamma_ptr + 4j` was located inside a *named* neighboring
  tensor (205 mapped, 0 unmapped) and the recovered value equals that
  tensor's content: layernorm's effective gamma continues into **beta's
  storage** then **poisonA** (beta's own leak continues into poisonA then
  poisonB, max rel err 1.6e-8); rmsnorm's effective gamma continues into its
  poison neighbors (2.4e-8). The layernorm gamma-leak "2.9% max rel err" is
  a relative-metric artifact on near-zero neighbor values; absolute
  agreement is at the in-bounds level.
- **Content dependence**: re-running with tensors held is bitwise identical
  (which is why the banked records were bit-identical across runs — the
  allocator layout is stable in-session, the reseeding-sweep anomaly that
  raised this flag). Freeing poisonA and re-allocating different values at
  the same address (verified same `data_ptr`) changes **only** columns
  ≥ 128; columns < 128 stay bit-identical.

## 2. The kernel is not the defect (`probes/oob_gpu.py`, stage B)

With valid-length companions (512-wide and exactly-333-wide) at the same
`x (64, 333)`:

| comparison | max abs diff |
|---|---:|
| layernorm kernel vs **kernel-faithful** float64 (unmasked pad-variance, gamma[:333]) | 9.5e-7 |
| rmsnorm kernel vs float64 math | 1.0e-6 |
| layernorm kernel vs **ideal** (masked-variance) math | 3.3e-2 |

First two rows: on valid inputs the kernels compute exactly their own
arithmetic, at fp32 precision — no new kernel bug. Third row: the 3.3e-2 is
the **already-known, separately-tracked** unmasked-pad-lane variance bug
(`../layernorm_mask_bug_2026-08-27/`), whose magnitude at randn width-333
inputs is hereby restated at this configuration (per-row mean² tails reach
~1.6% variance error → ~1e-2-scale output deviations at tail entries). It is
not part of this adjudication's defect.

Where the construction is wrong, precisely: `verification/specs/layernorm.py`
`_non_power_of_two()` (returns `torch.randn(n_rows, 333)` unconditionally)
plus `get_adversarial_inputs()` (re-wraps captured gamma/beta:
*"gamma/beta held fixed at whatever was captured, only x varies"*); same
pair in `rmsnorm.py`. The construction implicitly assumes
`len(gamma) ≥ 333`. Notably `masked_cumsum.py`'s author saw this exact trap
and declined a non-pow2 variant because *"the mask is a companion"* — the
hazard was known once and never systematized.

A second, converting defect: the reference **wrappers validate nothing**
(`torch.nn.functional.layer_norm` would raise on the same inputs), so an
invalid tuple becomes silent garbage instead of a loud
`reference_failure_kind` event. That robustness gap is what let this run
for three rounds undetected.

## 3. Exposure map

| harness | layernorm/rmsnorm base width | exposed? |
|---|---|---|
| autokernel corpus (`tritonbench_registry`) | 128 | **yes — the banked instance**, 205 leaked columns |
| `run_checker.py` primary | 512 | no |
| spec `valid_shapes` cross-shape runs | builds matching companions per shape | no |
| spec `valid_shapes` as primary: (2048, **128**) | 128 | **latent yes** if ever used as primary |
| adversarial search proposals | LLM-chosen | 1 of 17 in window; verdict independent (§4) |
| native/phase runs, attention_gram, method blind test | primary-only, no battery | no |

Also swept: layernorm and rmsnorm are the **only** two specs with a
shape-changing adversarial variant wrapped in captured width-dependent
companions (cumsum family's width-333 variants are single-tensor; rope,
groupnorm, instancenorm, batchnorm, cross_entropy, losses, matvec/batched/
diagonal/triangular variants are shape-preserving or fully regenerated;
matmul regenerates both operands).

## 4. Blast radius, established independently (`probes/blast_recount.py`, `data/`)

Recounted from scratch across **all** banked arms rather than inheriting the
closure round's figure: affected records n_samples 112+105 (7 arms),
scope_detect 64+60, gram_screen 32+30, check_timing 80+75. Zero catches
attributed to the check in any bank; the scope round's quoted per-op margin
minima are attained by `perturbation_tolerance` (layernorm) and
`adversarial_large_variance` (rmsnorm), not by the affected records; the
gram round's records carry no gram signal at all (fail-open, previously
annotated). Search DBs: 17 layernorm/rmsnorm proposals, one at width < 333,
whose verdict string neither mentions the variant nor depends on it (it
died at reference `unit_variance` validation — one of the 11 layernorm
reference-suspect records the norm adjudication already classified). The
affected records appear in aggregate audits (H2/H3 denominators) only as
generic sensitivity vectors; the closure round's over-weighting annotation
already covers them, with the added precision that these 31-per-arm vectors
measure a *memory-content-dependent* function, not the operator.

## 5. The fix, specified and deferred

1. **Spec fix** (both files): make the variant width-adaptive and
   contract-respecting — `w = 333 if n_cols >= 333 else <largest non-pow2
   width ≤ n_cols>` (e.g. `n_cols - 1` when that is not a power of two,
   else `n_cols - 3`), keeping the captured companions valid because the
   kernel then reads `gamma[:w]` in-bounds. This preserves the variant's
   tile-boundary intent at every base shape, including (2048, 128).
2. **Wrapper hardening** (both reference wrappers): assert
   `gamma.numel() == n_cols` (and beta) so an invalid tuple fails loudly
   into the `reference_failure_kind` machinery instead of silently reading
   memory.
3. **Contract test** (systemic, harness-side): for every spec, run
   `get_adversarial_inputs` at each `valid_shapes` entry (and the
   registry's corpus shapes) and assert every returned tuple satisfies the
   spec's own shape contract. This is the one test that would have caught
   this class before it ran, and it encodes what masked_cumsum's author
   knew.

**Regression criteria for the fix round**: (i) corpus arms re-run — verdicts
byte-identical to the pre-fix A arm (the variant's records change, no
verdict may); (ii) the two classes stop being reseeding-collapsed (rmsnorm's
bit-identity was OOB-content-driven) and the Gram screen evaluates them
(`gram_n_valid = 20`) and stays silent; (iii) the contract test passes at
every valid shape; (iv) exactly zero catch-attribution changes (no catch
ever came from these records).

## 6. Three findings, one family, not one bug

| | flash/sdpa softmax denominator | layernorm variance | this: OOB companions |
|---|---|---|---|
| side | kernel | kernel | **spec/harness** |
| mechanism | pad lanes enter denominator: `other=0` not annihilating under `exp(·−m)` | pad lanes enter variance: `other=0` not annihilating after mean subtraction | variant width 333 assumed ≤ companion length |
| trigger | N % 32 ≠ 0 | n_cols not pow2 AND mean ≠ 0 | base width < 333 |
| found by | Gram law deviation (−10% at 100×32) | norm adjudication of reference-suspect records | reseeding sweep bit-identity anomaly |
| status | fixed 2026-08-27 | flagged, unfixed | **this round: flagged, fix specified** |

The two kernel instances share a genuine root cause — the
padding-sentinel-neutrality assumption — and that family was swept to
closure at exactly 3 of 64 kernels (`../layernorm_mask_bug_2026-08-27/`).
This third instance is mechanistically unrelated: no sentinel, no kernel
arithmetic error — a harness-side width assumption. **There is no single
defect connecting the three and hence no single fix.** What connects them
is that all three live at non-power-of-two widths — the region the
default-shape-heavy corpus exercises least — and that none was caught by a
test: each was found by a measurement instrument doing something else
(Gram screen, record adjudication, replica census). The transferable lesson
is §5.3's contract test for the harness side, and — already done — the
sentinel sweep for the kernel side. A fourth instance, if it exists, should
be looked for where those two sweeps do not reach: candidate/mutant kernels
(the sweeps covered references) and any future spec whose variant changes a
shape.

## 7. Limits

- The poison experiment proves the mechanism at the corpus *configuration*;
  the banked records' exact leaked values are unrecoverable (that memory is
  gone) — irrelevant to the verdict, since §4 shows nothing depends on
  them.
- compute-sanitizer's failed positive control is reported as an environment
  fact (T4, CUDA 12.8, triton 3.6.0 JIT launches); it was not root-caused.
  Do not cite "sanitizer-clean" as evidence of in-bounds behaviour for
  Triton kernels on this stack.
- The recovery arithmetic for layernorm assumes the kernel-faithful
  variance (with the known pad term); its 4.6e-8 in-bounds agreement is
  simultaneously a re-confirmation of that variance formula at width 333.
- Stage B's "kernel == own arithmetic" is at one input draw per width; the
  claim it supports (no *additional* kernel defect at these shapes) is
  narrow and matches the prior rounds' characterization.

## 8. Reproduce

```bash
# GPU stage (T4):
export HOME=~/.colab-home && colab new --gpu T4 -s kccoob
tar --exclude='__pycache__' -czf /tmp/tb.tgz TritonBench/reference TritonBench/__init__.py
colab upload -s kccoob /tmp/tb.tgz /content/tb.tgz   # + probes/oob_gpu.py, probes/repro_oob.py
# extract, run oob_gpu.py; optionally compute-sanitizer on repro_oob.py (see data/sanitizer.log)
colab stop -s kccoob

# local stages:
python3 probes/blast_recount.py
python3 probes/searchdb_exposure.py
```

---

> **FIX SHIPPED, 2026-08-28 — `../oob_fix_2026-08-28/FINDINGS.md`.** All
> four §5 regression criteria met: verdicts byte-identical to the pre-fix
> arms (40/40, 0/200 both arms), catch attribution unchanged, the contract
> test in `tests/verification/specs/test_spec_contracts.py` passes at every
> valid shape (with a negative control pinning rejection of the pre-fix
> construction), and **the falsifiable prediction held**: both classes'
> reseeding collapse dissolved (rmsnorm 15 records → 15 distinct) and the
> Gram screen now evaluates them (`gram_n_valid = 20` everywhere) at ratio
> ≤ 1.0001 — silent, deep in-scope. The wrapper `ValueError` asserts are
> exercised on GPU (3/3 mismatch cases raise). The fix is
> draw-then-slice, so every deterministic record outside the two classes is
> bit-identical to pre-fix; the only exceptions are the frobenius
> atomic-add records, shown to be equally unstable between the two pre-fix
> rounds.
