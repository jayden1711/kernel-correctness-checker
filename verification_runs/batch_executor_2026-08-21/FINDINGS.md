# Item 2 — subprocess-spawn reduction by proposal batching

**Measured 2026-08-21 on a Colab T4** (torch 2.11.0+cu128, triton 3.6.0).
Raw records: `replay_ab.json`, `replay.log`. Harness: `replay_ab.py`.
Coordinator end-to-end: `coord_smoke.py`, `smoke_batched.db`, `smoke_single.db`.

## What changed

`executor.execute_proposal` spawned one interpreter per **(proposal, kernel)**
pair. `execute_proposal_batch` spawns one per **proposal**, running the reference
and every mutant in it. Spawns per proposal drop from `N+1` to `1`: 2→1 for the
16 single-mutant operators, 5→1 for matmul and flash_attention.

The original path is unchanged and still exists — it is the fallback whenever a
batch cannot finish, and the unbatched arm of the A/B (`--no-batch`).

## The headline number

Real recorded proposals replayed through both arms, 4 worker threads, identical
proposal set per arm. **Passes run A1 → B → A2**: if the two unbatched passes
agree, run order and warm caches are not what produced the difference.

| operator | kernels/proposal | A1 (single) | **B (batched)** | A2 (single) | order drift | effect |
|---|---:|---:|---:|---:|---:|---:|
| `causal_flash_attention` | 2 | 28.11s | **13.19s** | 27.67s | **1.6%** | **0.47x (−53%)** |
| `flash_attention` | 5 | 70.16s | **14.09s** | 70.34s | **0.3%** | **0.20x (−80%)** |

Per-proposal medians. Spawns per proposal: 2.00 → 1.00 and 5.00 → 1.00.
**Zero fallbacks fired** — `exec_mode` was `batched` on all 140 batched kernel
records; no poisoned contexts, no missed deadlines.

### Why the win exceeds the spawn-count ratio

Per-kernel work, by position in the batch (median ms):

| | position 0 (reference) | positions 1..N (mutants) |
|---|---:|---:|
| single | 6365 | 4906 – 7332 |
| **batched** | 6709 | **24 – 257** |

The reference module is loaded **once** per batch, so the Triton compilations the
checker triggers through it — perturbation's 20 samples, the cross-shape sweeps —
are paid once and reused by every later kernel. The mutants become almost free.
That is why flash_attention gains 80% where spawn count alone predicts 60%.

## Where the ~6.2s of startup actually goes — the number that decides what is next

| phase | median | share |
|---|---:|---:|
| `torch_import_ms` | **5241 ms** | **85%** |
| `cuda_init_ms` | 645 ms | 10% |
| `pre_module_ms` (interpreter + mp bootstrap) | 255 ms | 4% |
| `spec_import_ms` | 41 ms | <1% |
| `materialize_ms` | 3 ms | <1% |

**`import torch` is 85% of startup; CUDA context init is 10%.** This was the one
quantity the plan could not bound, and it reverses the ranking of the two
deferred options: a `forkserver` with `set_forkserver_preload(["torch"])` removes
the 85% while keeping one process per execution — same isolation, same timeout
semantics, a few lines of diff — where a persistent pool removes 95% but takes on
process/CUDA-context lifetime, crash-respawn and cross-execution state. **Do
forkserver before considering a pool.**

## The declared semantic change, measured

Batching materialises the inputs **once** per proposal from a `proposal_id`-derived
seed, so every kernel sees identical data. The single path seeds nothing, so each
spawned process drew its own tensors — meaning reference and mutants were
previously compared across *different* random draws.

| | disagreeing (proposal, kernel) pairs |
|---|---|
| A1 vs A2 — two **unbatched** passes, both unseeded | **5 of 80** (CFA), 2 of 60 (FA) |
| A2 vs B — unbatched vs **batched + shared + seeded** | **2 of 80** (CFA), 2 of 60 (FA) |

**The change moves fewer verdicts than the existing path moves against itself.**
Checker-pass rates are identical (CFA 34/80 both arms; FA 8/60 both arms), every
disagreement is on the `reference` kernel, and they flip symmetrically in both
directions — the signature of marginal inputs, not a systematic shift. The
batched path is additionally reproducible; A1 vs A2 shows the single path is not.

## Caveats that must travel with these numbers

- **The replay is a higher-contention regime than a real search.** Four threads
  spawn continuously here; in a real run each worker also spends ~6s per proposal
  on its LLM call, so fewer startups overlap. Absolute seconds are inflated in
  **both** arms. Projecting the measured per-kernel structure onto the banked
  2026-08-20 CFA timeline (reference 10.78s, mutant 9.96s) gives 20.74s → ~10.8s
  per proposal, i.e. **0.52x** — close to the measured 0.47x, so the ratio
  transfers; the absolute numbers do not.
- **End-to-end search wall time will improve by less than these figures**, because
  execution is only ~71% of it. For a 1-mutant operator: ~26.3s per proposal
  becomes ~16.9s, about **−36%**. Multi-mutant operators gain more.
- **The smoke-test DBs are wiring evidence, not measurements.** Their two arms ran
  in sequence with no order control, so `torch_import_ms` reads 5.9s in the arm
  that ran cold and 2.5s in the arm that ran warm. That is the very confound the
  A1/B/A2 structure exists to remove; do not quote those two DBs as timings.
- **`benchmarks/run_random_baseline.py` was deliberately NOT batched.** It feeds a
  published same-budget comparison, and batching it would change that baseline's
  input semantics (shared + seeded) as a side effect of a latency change. It is a
  one-line change and its own decision.
