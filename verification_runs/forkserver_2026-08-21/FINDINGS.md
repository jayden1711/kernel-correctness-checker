# forkserver — removing `import torch` from per-execution startup

**Measured 2026-08-21 on a Colab T4** (torch 2.11.0+cu128, Python 3.12.13).
Harnesses: `preflight.py`, `replay_fs.py`, `diag_flip.py`, `diag_contention.py`.
Raw records: `preflight.json`, `replay_fs.json`, `diag_flip.json`,
`diag_contention.json`, and the four `.log` files.

## What changed

`execute_proposal_batch` can now create its child by forking a **torch-preloaded
forkserver** instead of booting a fresh interpreter. Batching (item 2) cut how
many processes a proposal costs, `N+1` → `1`; this cuts what each one costs to
create. The two are orthogonal and compose — they are two switches, not one.

`execute_proposal` (one subprocess per kernel — the batch fallback and the
`--no-batch` arm) **stays on spawn, deliberately.** It seeds nothing, and under
fork "unseeded" would not mean "independent": every child would inherit the
forkserver's generator and draw identical tensors for every proposal. Probe 3
below measures exactly that.

Off by default. `--forkserver` turns it on.

## The headline

Real recorded proposals replayed through both arms, 4 worker threads, identical
proposal set per arm, no LLM. **Passes run B1 → C → B2**: B1 and B2 are the same
arm, so if they agree, run order and warm caches are not what produced C.

| operator | kernels/proposal | B1 (spawn) | **C (forkserver)** | B2 (spawn) | order drift | effect |
|---|---:|---:|---:|---:|---:|---:|
| `causal_flash_attention` | 2 | 13.59s | **7.71s** | 13.65s | **0.5%** | **0.565x (−43.5%)** |
| `flash_attention` | 5 | 14.04s | **8.99s** | 14.01s | **0.2%** | **0.642x (−35.8%)** |

Per-proposal medians. Pass totals: CFA 132.6s → **74.2s** → 133.3s;
FA 42.8s → **26.5s** → 42.4s. `start_method` was `forkserver` on every C record
and `spawn` on every B record — the run is known to have actually forked, not to
have silently fallen back.

**The prediction was made before the run and held.** Predicted 0.60x (CFA) and
0.62x (FA) from the startup arithmetic; measured 0.565x and 0.642x.

**Note the ratio is WORSE than batching's 0.47x/0.20x even though forkserver
removes a larger share of startup.** After batching, startup is a smaller slice
of what remains — the reference kernel's Triton compilation now dominates. "85%
of startup" is not "85% faster", and the two figures are about different
denominators.

## Where the startup went (`preflight.py` probe 4)

Median over 6 children, first child excluded (see caveat):

| phase | spawn | **forkserver** |
|---|---:|---:|
| `pre_module_ms` | 67.2 ms | 17.1 ms |
| `torch_import_ms` | **1527.3 ms** | **0.0 ms** |
| `cuda_init_ms` | 240.2 ms | 233.5 ms |
| **total to ready** | **1825.2 ms** | **251.7 ms** |

`import torch` goes to **exactly zero** — it is a `sys.modules` hit in a forked
child. **CUDA init is unchanged (240 → 234 ms), which is the design working, not
a shortfall:** initialising CUDA inside the forkserver would leave every fork
holding an inherited, unusable context. That 10% is deliberately still paid per
child, and is what a persistent pool would have to take on the risk of removing.

## The gate — four pre-flight probes, all on real hardware

| probe | question | result |
|---|---|---|
| 1 | does `import torch` in the forkserver poison CUDA in forks? | **No.** 6/6 forks ran a real Triton `softmax` from the corpus, max_err ≤ 1.5e-8 |
| 2 | is forking from torch's threaded runtime safe under load? | **Yes.** 20/20 across 4 threads in 31.0s, 0 stuck, 0 errors |
| 3 | what does an unseeded fork inherit? | **The hazard is real** — see below |
| 4 | where does startup go? | table above |

### Probe 3 — the inherited-RNG hazard, confirmed on real torch

Six forked children, **no seeding**, each drawing `torch.randn(4)`:

```
forkserver:  1 distinct draw across 6 children
             all six returned [-0.22193453, -0.4135631, 1.49266148, -0.75363588]
spawn:       3 distinct draws across 3 children
```

Bit-identical, every time. This is why `execute_proposal` stays on spawn and why
the batched child's `torch.manual_seed(_seed_for(proposal_id))` is load-bearing
rather than incidental. It is guarded offline by
`tests/instrumentation/check_forkserver_executor.py`, which drives the real child
with a generator read **at draw time** and pairs every claim with a mutation that
must break it — one dropping the seed, one applying it after materialization.
Both fire.

## Verdicts — the part that needed the most care

**Forkserver introduces no semantic change:** both arms batch, both seed from the
proposal id, only the start method differs. So the bar was zero movement.

The single-pass comparison appeared to fail that bar:

| comparison | result |
|---|---|
| A1 vs A2 — unbatched, **unseeded** (instrument control) | **4 of 40** disagree |
| B1 vs B2 — batched + spawn, seeded | **0 of 80** disagree |
| **C vs B2** — forkserver vs spawn | **2 of 80** disagree |
| C vs B1 — the *other* spawn pass | the **same** 2 pairs |

The instrument control fires, so the comparator is known to be able to see a
disagreement — without it, "0 of 80" would have been unfalsifiable.

**But "the floor is 0 and forkserver exceeded it" is the WRONG reading, and
chasing it is what produced the actual finding.** Two follow-ups:

**`diag_flip.py` — re-run the 2 proposals sequentially, 3x per arm.** Both passed
**6/6 in both arms**, 15 checks each, no errors. So the flip needs concurrency.

**`diag_contention.py` — 3 more passes per arm under the same 4-thread load,
capturing per-check detail** (the A/B recorded only `passed_checker`, which is
why the failing check was unknown):

| pass | arm | reference failures | `kernel_executed` flip |
|---|---|---:|---|
| spawn_0 | spawn | 5 / 40 | — |
| forkserver_1 | forkserver | 6 / 40 | `4aff5bf2` |
| spawn_2 | **spawn** | 6 / 40 | **`aff0c7f8`** |
| forkserver_3 | forkserver | 6 / 40 | `8bb756a5` |
| spawn_4 | spawn | 5 / 40 | — |
| forkserver_5 | forkserver | 6 / 40 | `25fc1d7f` |

**The spawn arm flips too.** The 5-proposal out-of-domain baseline is *identical
across all six passes*; the extra failure is always `kernel_executed`, and it
lands on a **different proposal every time** — six distinct proposals across the
eight observations, including the A/B's two.

### Conclusion, and it is not the one the A/B suggested

**Forkserver changed no verdicts by any mechanism of its own.** The 2-of-80 is a
**pre-existing, contention-sensitive false positive** in
`check_kernel_executed`'s delegation detector
(`verification/layer1_structural/runtime_guards.py:404-437`):

```python
if torch.equal(out1.float(), ref1.float()):     # ALWAYS true for the reference
    ...                                          # time 10 candidate calls
    ...                                          # time 10 reference calls
    if t_ref > 0 and t_cand < t_ref * 0.1:
        return False, "...candidate is Nx faster. Likely delegating to reference."
```

When the kernel under test **is** the reference, `torch.equal` is trivially true
and the check reduces to timing one function against itself. Under 4 workers
contending on one T4 that ratio is a lottery, and it reported **11.3x, 15.3x,
10.9x and 12.9x** "speedups" of the reference over itself.

**`B1 vs B2 = 0 of 80` was UNDERPOWERED, NOT EXACT** — §5 instance 12, in the
same shape: a control that was correctly structured, validated, and simply had
too few passes to resolve the effect. One pair of passes samples a per-pass event
that occurs roughly 1-in-3; observing 0 says almost nothing. **The honest report
is "failed to resolve", not "the floor is zero".**

**Not established:** whether forkserver *raises* the flip rate (3/3 passes vs
1/3). It is plausible — compressing startup means more of the wall time is
concurrent GPU work — but at n=3 per arm this comparison cannot resolve
anything, and it must not be written down as if it had.

## What this exposes, and what it does not

**A pre-existing Layer-1 false-positive source, logged and NOT fixed here.** It
is distinct from §3.0, which fixed the *probe*; §6.1 explicitly recorded the
delegation detector as "untouched and still reached on every non-ghost path". It
is a timing race that can only be triggered by concurrency, which is why a
single-threaded corpus run never saw it.

**No published number is affected.** `run_benchmark.py` does not run 4-way
concurrent executions, and `results.md` still reports `kernel_executed` as
40 ran / 0 caught / 0 FPs. What it does affect is the **adversarial search's
reference-failure rate**, which runs 4 workers — the same rate item 1a is about.
Fixing it is a checker change needing its own before/after; it is written up in
`SESSION_HANDOFF.md` as open item **1d**, not folded into a latency change.

## Caveats that must travel with these numbers

- **The replay is a higher-contention regime than a real search.** Four threads
  spawn continuously here; in a real run each worker also spends ~6s per proposal
  on its LLM call, so fewer startups overlap. Absolute seconds are inflated in
  **both** arms; the ratio is what transfers.
- **`preflight.py`'s absolute startup is not the production startup.** It times a
  bare `import torch` + CUDA init (1825ms), where the production child also
  imports the schemas, materializer and per-operator specs under contention
  (~6185ms in item 2's measurement). The 85%→0% shape of `torch_import_ms`
  transfers; the absolute totals do not.
- **The forkserver daemon's own boot is NOT isolated in probe 4.** The daemon was
  already warm from probes 1-2, so the "first child" row does not include it. It
  is a one-time cost of roughly one `import torch` per search process, amortised
  over every proposal in the run, but it has not been measured on its own.
- **End-to-end search wall time will improve by less than these figures**, since
  execution is ~71% of it. Projecting onto item 2's numbers: a 1-mutant operator
  at ~16.9s per proposal post-batching becomes roughly **~11.6s**, about −31%.

---

# Follow-up (2026-08-21): does forkserver make item 1d's race worse?

**The pilot could not answer this and said so** — 1 of 3 spawn passes flipped
against 3 of 3 forkserver passes, which at n=3 resolves nothing. This is the
powered re-test. Harness `race_rate.py`, analysis `analyze_race.py`, raw
`race_rate.jsonl` (2765 executions), full output `race_analysis.txt`.

## Design

Interleaved arms (never blocked), alternating which arm leads each pair, so a
drift over the run is common-mode rather than landing on one arm. Sample size
computed **before** the run: 921/arm to resolve the pilot-sized effect at 80%
power. One JSONL line per trial with `fsync`, resume by reading its own output,
and a watcher downloading every 5 minutes — which is why **three separate VM
reclamations cost 3 passes in total rather than the experiment**.

Two endpoints, because the binary one is rare-event limited: the flip itself
(`delegation_ratio > 10`), and the full `delegation_ratio` distribution, now
recorded on every reference execution rather than only when it trips.

## Result

| | spawn | forkserver |
|---|---:|---:|
| flips | **12 / 1400** | **22 / 1365** |
| rate | **0.86%** | **1.61%** |
| 95% CI | [0.49%, 1.49%] | [1.07%, 2.43%] |
| per 80-proposal search | 0.69 spurious ref failures | 1.29 |

Two-proportion **z = +1.65, p = 0.10**. Mann-Whitney on the ratio distribution
**p = 0.29**. **Minimum detectable effect at this n: 2.53x — and the observed
effect is 1.88x, i.e. BELOW the experiment's own resolution.**

**So this FAILED TO RESOLVE. It is not a null result and must not be written up
as one** (§5 instance 12, for the third time in this project).

What it did establish:

- **The direction replicates in 3 of 3 independent sessions** — ratios 1.25x,
  2.40x, 1.79x, on three different physical VMs. Not a machine artifact.
- **The arms are identical through the bulk.** At thresholds 2-4 the ratio is
  0.94-0.98x; p50 is 0.92 vs 0.91. Divergence appears only in the 5-10 band.
- **Effects at or above 2.53x are ruled out.**
- **The two most extreme outliers in the whole dataset (51.24 and 23.26) are
  both in the SPAWN arm.** The extreme tail is not forkserver-specific.

Best estimate: forkserver **probably** raises the rate about 1.9x, from ~0.9%
to ~1.6%, i.e. **about 0.6 extra spurious reference failures per 80-proposal
search**. Certifying that at p<0.05 needs ~2750/arm against the ~1400/arm here
— roughly four more GPU-hours across three more sessions. **That was judged not
worth spending, because the decision below is the same either way.** Reopen it
if certification is wanted for its own sake.

## What this changes about item 1d

**A fixed threshold is NOT a safe fix, and the earlier suggestion of "raise it
to 50x" is withdrawn.** The observed maximum grew from **23.26 at n=1750** to
**51.24 at n=2765** — the tail keeps extending with sample size, so no constant
derived from a finite sample is provably safe:

| threshold | flips, both arms pooled (n=2765) |
|---:|---:|
| 10x (current) | 34 (1.23%) |
| 20x | 2 (0.07%) |
| 30x-50x | 1 (0.04%) |
| 60x+ | 0 |

The right fix is **timing robustness, not a bigger constant** — best-of-N, or
interleaving the two timing loops instead of running them back to back, so a
single scheduling stall cannot decide a verdict. A 51x apparent "speedup" of the
reference over *itself* is what the current construction admits.

## The decision, and why

**`use_forkserver` stays OFF.** The rule this was measured against was: enable
it if it is *genuinely no worse*; leave it off if it is worse or still unclear.
It is not genuinely no worse — the point estimate is consistently ~1.9x with
3-of-3 directional replication — and it is not certified either. That is
"unclear", and the rule says leave it off.

**What unblocks it is fixing 1d, not more trials of this experiment.** The
phenomenon being compared is entirely a false positive of a check whose
threshold sits at roughly the 98th percentile of its own noise. Fix that and the
rate goes to near zero in *both* arms, at which point forkserver's measured
**−43.5% / −35.8% per-proposal latency** can be taken with nothing traded away.

Sequencing, explicitly: **fix 1d → re-run this comparison → enable forkserver.**
Enabling it first would buy a 40% speedup at the cost of roughly doubling a
known-defective check's false-positive rate, which is the wrong order.
