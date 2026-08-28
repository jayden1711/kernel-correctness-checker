# Cross-candidate reference sharing does not exist — 26 of 7144 executions, 0.004% of runtime

**Measured 2026-08-25 on a Colab T4** (shipped-warm Triton cache). Probe
`probe_refshare.py`, driver `refshare.sh`, raw records `reference_calls.json.gz`
(7144 reference executions, one row each).

**No repo code was changed.** The probe wraps `entry["torch_ref_fn"]` at the
call site; every reference execution in the numeric layer funnels through that
callable, so wrapping it captures all of them without touching the library.

---

## Verdict

**The hypothesised redundancy is not there.** The reference output is *not*
recomputed identically across candidates sharing an `(operator, shape, dtype)`
tuple, because those candidates never see the same input values.

| | executions | time | share of corpus run |
|---|---:|---:|---:|
| **cross-candidate duplicates (the lever)** | **26 of 7144** | **0.0023 s** | **0.004%** |
| within-trial duplicates (a different lever) | 919 | 0.1003 s | 0.193% |
| both together | 945 | 0.1026 s | 0.197% |

**The ceiling is 0.004% of corpus runtime.** That is not a small win; it is no
win. Stop here.

---

## Why — inputs are drawn fresh per candidate

`tritonbench_registry._mk_single` (and every sibling generator) is:

```python
def _mk_single(rng):
    return (rng.normal(size=(64, 128)).astype(np.float32),)
```

`harness.run()` threads one advancing `np.random.default_rng(seed)` through
every trial, and `checker_adapter._get_torch_inputs` calls `input_fn(rng)` once
per candidate check. So two candidates at the same `(operator, shape, dtype)`
receive **different tensors**. The shape and dtype repeat; the values do not.

Measured directly: fingerprinting every reference execution by
`(shape, dtype, sum, sumsq, min, max)` gives 1469 distinct fingerprints across
7144 executions, and of the 5675 redundant ones **99.5% are inside a single
trial**. Only 26 span more than one trial, and those are degenerate constants
(e.g. `check_weight_magnitude`'s `large_uniform = full(shape, 1e4)`, which is
input-independent by construction).

**A cache keyed on `(operator, shape, dtype)` would therefore be unsound, not
merely ineffective** — it would return a stale output computed from a different
random draw, on nearly every lookup. This is disqualifying on correctness before
it is uninteresting on performance.

---

## Where reference time actually goes

Reference-only execution is **0.91 s = 10.8% of serialised checker time**
(`KCC_CHECK_TIMING`-style sync around each call; absolutes are upper bounds,
shares are meaningful).

| call site | executions | ms | % of reference time | cacheable? |
|---|---:|---:|---:|---|
| `_time_once` (delegation detector) | 4130 | 359.9 | **39.4%** | **NO** |
| `check_kernel_executed` | 637 | 184.9 | 20.3% | maybe |
| `other` | 503 | 157.2 | 17.2% | maybe |
| `check_perturbation_tolerance` | 940 | 109.0 | 12.0% | maybe |
| `check_determinism` | 600 | 60.6 | **6.6%** | **NO** |
| `check_all_tiles_visited_generic` | 200 | 21.7 | 2.4% | maybe |
| `check_weight_magnitude` | 88 | 11.3 | 1.2% | maybe |
| remainder | 46 | 7.8 | 0.9% | maybe |

**46.1% of all reference execution is uncacheable by semantics**, for two
reasons that are not caveats but hard blocks:

1. **`_time_once` — 57.8% of all reference executions by count.** This is
   `check_kernel_executed`'s delegation detector
   (`runtime_guards.py:451-465`), which runs the reference 10 times
   (`_ROUNDS=5 × _CALLS=2`) purely to measure elapsed time, and whose verdict is
   `t_cand < t_ref * 0.1`. **A memoised reference returns in ~0 s, so `t_ref`
   collapses and the check's comparison becomes meaningless.** Serving these
   from a cache does not speed the check up; it deletes it.
2. **`check_determinism` — 600 executions.** Its verdict *is* repeated
   execution: it runs the same function three times to detect non-deterministic
   reductions. A cache makes it pass trivially and unconditionally.

---

## Correctness risk beyond the keying problem

Two further hazards, either of which independently disqualifies a naive cache:

- **`frobenius_norm`'s reference is not deterministic.** It accumulates through
  `tl.atomic_add` (`TritonBench/reference/frobenius_norm.py:22`), and float
  addition is non-associative, so its output differs bitwise run to run. This is
  the documented flake behind the 3–5 of 2200 reference-verdict movements seen
  in every corpus run this month. Caching it would **freeze one draw** and
  silently suppress a known, published source of variance — changing a
  benchmark's semantics as a side effect of a latency change, which is the exact
  substitution this project has been burned by (SESSION_HANDOFF §7, the unseeded
  executor; §2.5's item 2b).
- **RNG consumption.** Reference executions inside
  `check_perturbation_tolerance` are interleaved with `torch.randn_like` draws
  for the 20 perturbation deltas. Serving a reference from cache does not by
  itself change the stream, but any restructuring that skips the surrounding
  work does — the same hazard that required `KCC_ABLATION_SEED` in
  `verification_runs/check_timing_2026-08-25/`.

---

## The one real (and still tiny) finding

Within a single candidate's check, the same reference input is executed up to
**29 times**. Excluding the two semantically-uncacheable sites, **919 of 2414
remaining executions are within-trial duplicates, worth 0.1003 s = 0.193% of
corpus runtime.**

That is a per-candidate memoisation, not the cross-candidate sharing this
investigation set out to find, and at under 0.2% it sits below even the 2–3%
band every other check-level lever landed in.

| lever | corpus saving |
|---|---:|
| shipping a warm Triton cache | **75%** |
| `n_samples` 20 → 5 | 3.2% |
| `n_samples` 20 → 10 | 2.0% |
| removing `check_weight_magnitude` entirely | 1.9% |
| within-trial reference memoisation | 0.19% |
| **cross-candidate reference sharing** | **0.004%** |

---

## Recommendation

**Do nothing here.** Not "the win is small" — the mechanism the investigation
was looking for does not exist, and the keying scheme it would require is
unsound against inputs that are freshly drawn per candidate.

This also closes the search for algorithmic levers at check level. Five
independent investigations (process model, compile caching, per-check
redundancy, `n_samples`, reference sharing) have now each landed at ≤3%, against
75% for the one infrastructural change. **The checker's per-candidate cost is
not where the remaining time is**, and further check-level work should be
justified by correctness or coverage, not by latency.
