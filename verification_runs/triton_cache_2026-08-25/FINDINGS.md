# The Triton on-disk cache is 75% of the corpus benchmark's wall clock

**Measured 2026-08-25 on a Colab T4** (Tesla T4 15360 MiB, torch 2.11.0+cu128,
Python 3.12.13, session `kccmeas`). Harness: `driver.sh` (four invocations of an
**unmodified** `benchmarks/autokernel/files/run_benchmark.py`), environment probe
`probe_env.py`. Raw records: `timings.txt`, `gpu.csv`, `load.txt`,
`cache_pre.txt`, `cache_after_run*.txt`, and `run{1,2,3,4}/`.

**No repo code was changed to produce these numbers.** The driver only invokes
the existing entry point and samples the machine around it. That is deliberate:
the point was to measure the path as it ships, not a modified one.

---

## The headline

`run_benchmark.py` had **never been wall-clock timed** — `run_benchmark.py`,
`harness.py` and `reporting.py` contain no `perf_counter` outside the per-call
system timers, and `results.json` carries no run duration. The first number below
is therefore the first real cost on file for this path.

| run | wall | condition |
|---|---:|---|
| **run1** | **241.7s** | process cold, Triton disk cache **empty (verified)**, page cache cold |
| **run2** | **60.8s** | process cold, Triton disk cache **warm** |
| run3 | 59.3s | replicate of run2 |
| run4 | 233.4s | Triton cache **cleared**, everything else warm — control |

- **run1 -> run2 is 3.98x**, saving **180.9s = 75% of run1**.
- **run2 vs run3: 2.5% apart.** The win replicates; it is not a one-off.
- **run4 vs run1: 3.4% apart.** The control fires.

**The control is the reason to believe this.** `rm -rf ~/.triton/cache` with
*everything else left warm* — same live VM, same warm page cache, same warm pip
state, same process-start conditions — returns the run to within 3.4% of cold
cost. Without run4, the run1->run2 delta would have been confounded by page cache
and general session warmth, and "the cache is worth 4x" would have been a guess
dressed as a measurement. run3 and run4 were added for exactly this reason and
are the difference between a number and an attributed number.

---

## Where the 181s actually lives

The harness's own timer is **invariant** across all four runs:

| run | wall | timed by harness | untimed (`_warm()` + imports) |
|---|---:|---:|---:|
| run1 | 241.7s | 43.1s | **198.6s** |
| run2 | 60.8s | 43.9s | **16.9s** |
| run3 | 59.3s | 42.4s | 16.9s |
| run4 | 233.4s | 44.7s | 188.7s |

The cache removes **91.5% of the untimed region** and **nothing at all** from the
timed region. Two consequences worth stating plainly:

1. **`_warm()` relocates the compile cost out of the latency table without
   removing it from the clock.** That is what it was designed to do (see its
   docstring), and it succeeded — the latency table is honest. But it means the
   published p50/p90 figures and the run's wall time are measuring disjoint
   things, and optimising against the former never touched the latter.
2. **The floor for this path is ~60s**, of which ~44s is the harness's own timed
   work (Python/numpy check logic, perturbation sampling, shape sweeps). Getting
   below that means attacking the timed region, not compile.

---

## Correctness control — a warm cache moves no verdict

Comparing every run against run1, per (system, corpus entry):

| run | mutant verdicts differing | reference verdicts differing |
|---|---:|---:|
| run2 | **0 of 440** | 3 of 2200 |
| run3 | **0 of 440** | 5 of 2200 |
| run4 | **0 of 440** | 5 of 2200 |

All four `your_checker` systems report **100% catch / 0% FP in every run**.

Every one of the reference movements is `frobenius_norm` inside an
`autokernel_gate` variant — the known-inherent atomic-add bitwise determinism
flake documented in SESSION_HANDOFF §3, and the identical signature to the
delegation-fix corpus regression (5 of 2200, same operator, same baselines,
flipping in both directions). **It also appears in run4, whose cache was
cleared**, so it is background flake and not a cache effect. There is no
mechanism by which it could be one: Triton keys its cache on the specialization
that determines the generated code, so a hit returns the same cubin the miss
would have compiled.

---

## vCPU count and GPU utilization

`nproc` = **2**. Intel Xeon @ 2.00GHz, `MemTotal` 13,286,944 kB,
`cpu.max` = `max 100000` (no cgroup quota), `sched_getaffinity` = 2. A genuine
`n1-highmem-2`. The inference from the handoff's "1.5GB of 12.9GB" was correct,
and any plan resting on more than 2 cores is resting on nothing.

GPU utilization sampled at 1 Hz across each run:

| run | wall | mean | p50 | p90 | max | samples >50% | **GPU-busy sec** |
|---|---:|---:|---:|---:|---:|---:|---:|
| run1 (cold) | 241.7s | 2.55% | 0% | 10% | 43% | 0 | **6.2** |
| run2 (warm) | 60.8s | 8.07% | 1% | 28% | 37% | 0 | **4.9** |
| run3 (warm) | 59.3s | 9.98% | 1% | 35% | 41% | 0 | **5.9** |
| run4 (cleared) | 233.4s | 2.36% | 0% | 9% | 38% | 0 | **5.5** |

During run1, **70% of samples read exactly 0%** and 85% read <=5%. Binned over
time, the compile-dominated first 150s never exceeds a 1.2% mean:

```
t=  0- 30s  mean  0.0%   max  1%      t=120-150s  mean  1.2%   max 16%
t= 30- 60s  mean  0.5%   max  9%      t=150-180s  mean  4.5%   max 36%
t= 60- 90s  mean  0.8%   max 11%      t=180-210s  mean  4.3%   max 26%
t= 90-120s  mean  0.4%   max  6%      t=210-240s  mean  8.7%   max 43%
```

**The number that settles it is the last column.** Absolute GPU work is ~5-6
seconds in every run, whether that run took 60s or 242s. The 181s the cache
removes was **100% host-side**. This is not "CPU-bound compile overlapping
GPU-bound verification" — there is no meaningful GPU-bound phase to overlap with,
and a pipelining scheme would have had ~5s of GPU work to hide 180s of compile
behind.

CPU load average during run1: mean **1.17**, max 1.37, against 2.0 for both
cores. One core saturated, ~0.7 core idle — some headroom for parallel compile,
capped hard at 2x by the vCPU count. **Read run1's figure only:** loadavg is a
1-minute EWMA, so run2's and run3's (60s runs) are contaminated by the run
before them.

---

## Cache state on a fresh VM — nothing is durable

Probed **before anything imported triton**, on a fresh session:

```
TRITON_CACHE_DIR: <unset>      TRITON_HOME: <unset>      HOME=/root
~/.triton              does not exist
/root/.triton/cache    does not exist
find / -maxdepth 6 -type d -name cache -path '*triton*'  ->  (empty)
```

`HOME` sits on the overlay root filesystem, which dies with the VM. Nothing in
the repo sets these either. **Every fresh Colab session starts with an empty
Triton cache and rebuilds all of it.**

After one full corpus run:

| | |
|---|---:|
| raw size | 118,555,552 B (113 MiB) |
| directories | 1,180 |
| files | 8,730 |
| **distinct compiled specializations** | **1,086** |
| gzipped | 21,697,329 B (20.7 MiB) |
| time to tar+gzip | 4.3s |

1086 specializations across 180.9s of recovered compile is **~167ms per
specialization**. File mix: 1086 each of `cubin`/`ptx`/`llir`/`ttir`/`ttgir`/
`source`, 2172 `json`, 42 `so`. By bytes: llir 26.7 MB, cubin 22.9 MB, ptx
21.8 MB, ttgir 9.1 MB, ttir 8.0 MB, json 2.1 MB.

---

## What this corrects

The prior investigation declined to estimate this ceiling and guessed ~55% from
`forkserver_2026-08-21/preflight.json` probe 1 (2755ms -> ~1180ms for the same
kernel across processes on one VM). **The real figure is 75% of wall / 91.5% of
the untimed region.** The estimate was too conservative, and the correction
changes the ranking: the cache lever is larger than the "scope the run to the
checker systems" lever (~3x, derived), costs no code, and carries no semantic
risk — where scoping requires reworking `harness.run()`'s single shared
`np.random.default_rng(seed)` into per-cell derived seeds, which is a declared
change to input draws.

**The remaining unknown named there — upload time for the 21.7 MB cache — is now
measured; see the next section.**

**Not established here:** anything about the adversarial search path. These four
runs measure `run_benchmark.py` only. The search proposes novel shapes per
iteration, so its specializations are far less likely to repeat and the cache
should be expected to help it less — by how much is unmeasured.

---

## Shipping the cache to a fresh VM — measured, and it wins

**Measured 2026-08-25, sessions `kcccache` (build) and `kccship` (restore).**
Harnesses `buildcache.sh` and `ship.sh`; records in `shipped_cache/`.

The prior report flagged upload time as the one unmeasured quantity and warned
that the 743 KB / 1.8s reference point would not extrapolate. It does not:

| operation | size | measured |
|---|---:|---:|
| `colab upload` of the cache, trial 1 | 21.7 MB | **8.18s** |
| trial 2 | 21.7 MB | **7.86s** |
| trial 3 | 21.7 MB | **7.47s** |
| **mean** | **21.7 MB** | **7.84s** |
| `colab upload` of the source tarball (same session, reference) | 743 KB | 1.61s |
| `colab download` of the cache | 21.7 MB | 3.56s |
| `tar xzf` on the VM | 21.7 MB | 0.95s |

Marginal throughput is **~3.4 MB/s** on top of **~1.5s fixed overhead**. Scaling
the 743 KB point linearly would have predicted **52.6s** — **6.7x too
pessimistic**, because that point is ~90% fixed overhead. Declining to
extrapolate it was correct.

### It is a genuine 100% cache hit on different hardware

The cache was built on `kccmeas`/`kcccache` and restored onto `kccship`, a
**different physical T4** (`GPU-e550199c-…` vs `GPU-d6066aae-…`).

| | before run | after run |
|---|---:|---:|
| cache files | 8,730 | **8,730** |
| `.cubin` specializations | 1,086 | **1,086** |
| bytes | 118,584,224 | **118,584,224** |

**The run added zero new cache entries.** Not one specialization missed. This is
the check that makes the upload number meaningful — an upload time for a cache
that did not transfer would have been worthless. (The restored raw size is
118,584,224 B against the source VM's 118,555,552 B, a 28 KB difference in
directory block accounting after untar, not content: file and cubin counts are
identical.)

### End-to-end cost on a fresh session

| | |
|---|---:|
| cold run, cache built from scratch | **241.7s** |
| **run with shipped cache, fresh VM** | **69.9s** |
| upload + untar to get there | 7.84 + 0.95 = **8.8s** |
| **net saving per fresh-session run** | **~163s** |

**Verdict control on the shipped run**, against run1 on the original VM:
**0 of 440 mutant verdicts differ**, 4 of 2200 reference verdicts differ — all
four `frobenius_norm` inside `autokernel_gate` variants, the same known
atomic-add flake as everywhere else in this document. Timed-by-harness total
**43.2s**, against run1's 43.1s: unchanged, as expected.

The shipped run is 69.9s rather than the 60.8s of an on-VM warm run because it is
the *first* run in its session — cold page cache over the freshly extracted
files, cold everything else. **69.9s is the honest fresh-session figure** and the
one the recommendation rests on.

### Recommendation

**Ship the cache. Make it the default for fresh-session runs, including one-off
runs — not only iteration bursts.** 8.8s to save ~163s is a 19:1 return, and it
removes the reason the keep-alive-session form existed: with shipping, the *first*
run of a session is already fast, so there is no longer a tradeoff against idle
compute-unit burn. Keeping a session alive remains marginally faster per run
(60.8s vs 69.9s) and is the better choice while actively iterating in a session
that is already open, but it is no longer the thing to reach for first.

Two caveats, neither blocking:

- The cache is keyed on Triton version, GPU architecture and kernel source. It
  was validated for **triton 3.6.0 / torch 2.11.0+cu128 / sm_75 (T4)**. A
  different GPU class (L4, A100, H100) or a torch/triton upgrade invalidates it
  — **degrading to a miss and a normal cold run, never to a wrong answer.**
- It goes stale only if a `TritonBench/reference/*.py` kernel changes. Editing
  `verification/` — the actual iteration loop — does not invalidate a single
  entry, which is precisely why the 181s was pure waste.

---

## Reproducing

```bash
export HOME=~/.colab-home
colab new --gpu T4 -s <name>
tar --exclude='__pycache__' --exclude='.venv' -czf kcc.tgz \
    verification benchmarks scripts tests TritonBench
colab upload -s <name> kcc.tgz /content/kcc.tgz
colab upload -s <name> driver.sh /content/driver.sh
# driver.sh nohups itself; poll /content/meas/timings.txt for progress
```

To reproduce the shipped-cache result, build the cache once with
`buildcache.sh`, `colab download` `/content/triton_cache.tgz`, then on a fresh
session upload it and `cd /root && tar xzf /content/triton_cache.tgz` before
running — that is what `ship.sh` does.

`driver.sh`, `buildcache.sh` and `ship.sh` in this directory are the exact
scripts that produced these numbers.
