# Inventory pass — every mathematical claim found outside verification_runs/

**Compiled 2026-08-27** by a repo-wide sweep of `verification/`, `benchmarks/`
(incl. all theory .md docs), `scripts/`, `tests/`, and the root .md files.
Excluded: `verification_runs/` (already audited), `KernelBench/`, `TritonBench/`
(third-party). Categories: **(a)** one of the seven documented results or a
known corollary; **(b)** trivial/standard; **(c)** potentially new — stated or
exploited but never derived, validated, or documented as a theory result.

The seven documented results appear in: `benchmarks/NUMERICAL_THEORY.md`
(§ tolerance invariance, class split), `RESULTS_SUMMARY.md` §4,
`SESSION_HANDOFF.md`, `verification/layer2_numeric_oracle/structural_l.py`
(closed forms, M3, conv identity, scan +24.7%), `scope_detect.py` (taxonomy),
`CORPUS_EXPANSION_PLAN.md` (scan forms, RoPE, m=1 mechanism).

## Category (c) — potentially new, by area

### scope_detect.py
- `:73-82` geometric-midpoint threshold rule, stated as maximising the smaller
  log-margin — asserted optimality, never proved. (c)
- `:47-52` the 900% constant-defect signature (`|s−10s|/s = 9`) — exact, trivial
  once stated. (c, small)
- `:84-110` **defect screen falsified and unresolved**: in-scope worst 9.605% vs
  out-of-scope least 6.6% — classes overlap; open item.
- `:60-68` **`CV ≤ 0.7555` asserted as "a correct property of the linear
  regime"** with no derivation. → **Resolved this audit (§H5 of FINDINGS): it is
  the half-normal CV √(π/2−1); numerically supported as a sharp ceiling with
  equality at rank-1.**
- `:112-117` median-vs-minimum estimator argument for the s/ulp screen. (c)
- `:150-154` defect statistic converged by 20 deltas (0/854 disagreements
  vs 40). (c)
- `:125-131` "a correction with no known sign is not a correction" — signed-bias
  claim about the estimator under scope divergence. (c)

### structural_l.py / M3 machinery
- `:467-484` M3's orthogonal-rows assumption — the load-bearing modelling step;
  its validity condition undocumented. → **Partially resolved this audit: for
  the scan family the exact correlated law is now derived and validated (§H1).**
- `:480-483` M3 validated at n=40, shipped default 20 — acknowledged
  unquantified extrapolation. (c, open)
- `:492-497` profile-truncation rule (1e-3·max, 8:1 subsample) — magic
  constants with an implied tail-negligibility bound. (c)
- `:143-154` `_qlin` exactness claim vs torch.quantile, untested in-repo. (c)
- `:305-309`, `:342-348` hardsigmoid 96.7% zero rows / triangular_matmul 41.7%
  structural zeros — degenerate-profile regime, flagged not fixed. (c)
- `:350-352` **stale in-code prediction** (m=1 losses expected to drag M3 down)
  — refuted by the Phase-1 GPU round; the comment still says otherwise. (c, fix)
- `:385-388` conv input-independence equivalence class, never formalised. (c)
- `:52-60` closed forms verified on ordinary inputs; 75% of perturbation-routed
  calls are adversarial variants — quantified scope gap (partially closed by
  structural_l round M3). (c)

### perturbation.py
- `:26-35` prefix-monotonicity identity of the sensitivity vector (exploited by
  the n_samples round, never written up as a result). (c)
- `:175-183` RNG batching non-equivalence with measured witness. (c)
- `:160-174` bit-identity argument for removing per-sample `.item()`. (c)

### Layer 1
- `runtime_guards.py:414-446` + `SESSION_HANDOFF.md:1795` — heavy-tailed
  contention ratio (p99=11.45, max grows with n: 23.3@560 → 51.2@2765);
  "no constant derived from a finite sample is provably safe"; robust `min`
  estimator argument. Strongest undocumented statistical result in the repo. (c)
  *(DOCUMENTED 2026-08-28: Fréchet-domain model, α ≈ 2–3.5, max-growth fit,
  and the post-fix FP surface (onset ~1.2–1.3×, dead zone ≥ 2×) —
  `../contention_tail_2026-08-28/FINDINGS.md`.)*
- `tile_coverage.py:95-118` NaN sentinel likelihood + absorbing-element
  argument. (b/c)
- `ast_analysis.py:330-358` `max_torch_ratio=0.5` magic threshold; known
  unsound for fused kernels (CORPUS_EXPANSION_PLAN §241-248). (c, open defect)
- `CHECK_ABLATION_FINDINGS.md:270-299` OR-composition can-only-reduce-FP claim;
  argmax tail-risk independence argument; 8-seed stability result. (c)

### Layer 3 tolerances
- `matmul_properties.py:32-56` distributivity tolerance derivation
  (cancellation floor 2e-3 + separation vs 23+). Real per-check derivation. (c)
- precision-coercion factor `0.9` shared across 4 operators, uncalibrated. (c)
  *(MEASURED 2026-08-28: live on only 4 corpus records, dead zone
  (0.58, 0.998) — `../l3_margins_2026-08-28/FINDINGS.md`, which also covers
  the unit-variance atol asymmetry and the non-round probe constants.)*
- `groupnorm/instancenorm` unit-variance atol 3e-2 vs layernorm's 1e-3 —
  unexplained 30× asymmetry. (c)
- non-round probe constants family (3.7, 2.5, …, shift 50.0 for cross_entropy
  vs fp32 exp overflow at ~88). (c)
- `pool_properties.py:13-24` avg-pool shift-equivariance holds only with
  padding=0 (count_include_pad) — derived conditional invariance. (c)
- `gelu_properties.py`, `swish_properties.py` non-monotonicity dips
  (−0.170 @ −0.752; −0.278 @ −1.278) — measured, never derived. (c)
- `kernelbench_operator_registry.py:44-60` shape-collision identification
  argument. (c)

### Adversarial search
- `strategy/beam.py`, `greedy.py`, `diverse.py` — scoring constants
  uncalibrated; **docstring/code mismatches verified this audit**: beam
  promises −2 per errored mutant, greedy promises +2 per no-gap catch and −3
  per errored mutant; none implemented. **Diversity penalty proved inert at
  defaults (§H4 of FINDINGS).**
- `executor.py:1014-1027` near-miss hint threshold 1e-4. (c)
- `schemas.py:360-378, 437-441, 496-506` derived shape constraints, the
  retracted N≥16 claim, silent-garbage constraint class. (c)
- `CFA_NONHIT_ROOTCAUSE.md:166-183` context-effect 1.9× proposals-to-hit,
  n=9, no CI. (c)

### Theory docs (claims beyond the seven)
- `NUMERICAL_THEORY.md` §3.1-3.4: four operator-specific closed-form blindness
  conditions (softmax `e^{−v} ≤ 1.344/(n−64)`; first_tile tail-mass; **GELU
  blind/visible bands with sup gap 0.00949 near |x|=2**; skip_eps ε/2σ²
  conditioning). Derived + validated in-sample; not among the seven. (c)
- §4.2 fp-absorption closed threshold `(n−P)e^{−v} < ½ulp(m)` → v>40.9 fp64 /
  21.5 fp32. (a-adjacent; the formula itself un-elevated). (c)
- §4.3 **rtol on index-valued outputs silently rescales the test with array
  size** (`a + r·j` grows in j). Clean, general, one line. (c)
- §5 wrong_dtype fp16-exactness masking condition — "consistent with all six
  observations, not separately falsified". (c, open)
- §5.4 skip_boundary_tiles — both proposed masking conditions falsified;
  explicitly open. (known)
- §7 two named conjectures: "bug in the kernel of the evaluation map" framing;
  blind-set measure question. (c, open)
- §8 attention obstruction: sequential-recurrence residuals don't decompose. (c)
- `BUG_CLASS_THEORY.md` — 120/120 predictive claim (a); leakage ablation
  112/120 and 83/120 (c); `wrong_variance` 1/n gap-existence scaling (c);
  three "no masking input found" negative existence claims (c).
- `CORPUS_EXPANSION_PLAN.md` — L_f ≤ L_g·L_h non-composition consequence (c);
  autograd-JVP cost crossover in m (c); cumprod exclusion = third structural-
  exclusion class (c); family-collapse claim (27 forms → 6 families) (c);
  BatchNorm/Dropout purity violations (c).
- `SOTA_CHECKS_REGISTRY.md:30,47-55` fixed-tolerance-causes-FP claim
  **contradicted** by `autokernel/AUTOKERNEL_BASELINE_AUDIT.md:133-141` —
  unretracted. (c, doc defect)
- `BENCHMARK_RESULTS.md:270-277` sigmoid unstable_exp overflow threshold ~88.
  (c) `:585-600` rule-of-three FP bound + population-mismatch argument. (b/c)
- noise floors: ±7% latency (`SESSION_HANDOFF.md:1410`), ~6% unseeded verdict
  floor (`:1766-1772`), jitter attribution (`BENCHMARK_RESULTS.md:398-403`). (c)

### Specs
- `base_spec.py:29-52` batch-samples validity conditions (a-adjacent, stated
  estimator-correctness condition). (c)
- huber kink placement, std one-pass cancellation, scan alternating-signs
  cancellation probe, conv border-isolation argument, depthwise channel
  isolation, RoPE norm preservation, frobenius silent-wrongness note. (b/c)

## Category (b) — standard, listed for completeness
Textbook invariants used as Layer-3 checks (softmax rows sum to 1, shift/scale
invariance, zero-mean/unit-variance, distributivity, convex-hull bounds,
permutation invariance), `_is_pow2`/`_next_pow2` bit tricks, rtol/atol
crossover arithmetic, precision/recall arithmetic on rounded inputs.
