# Cache-shipping is now the standard run path: `scripts/colab_bench.sh` took a fresh VM from 252s cold to 79s shipped with zero new compiles, the regression is 40/40, 0/200 through both paths, and the staleness guard passed all 11 degradation checks — a mismatch costs a cold run, never a wrong answer

**Built and GPU-validated 2026-08-28** (sessions `kccflip` cold+harvest,
`kccship2` fresh-VM ship; both stopped). This adopts the
`triton_cache_2026-08-25` round's recommendation ("ship the cache, make it
the default for fresh-session runs") — measured there at 241.7s→69.9s with
an 8.8s ship cost, 19:1 — by turning the manual `ship.sh` step into the
normal workflow. Artifacts: `scripts/colab_bench.sh` (dev-side one-liner),
`scripts/vm_cache_tool.py` (VM-side probe/guard/harvest),
`probes/staleness_test.py`, logs in `data/`, results in
`results_gpu/kccflip_20260828_083346/` and
`results_gpu/kccship2_20260828_084137/`.

## 1. What ships

**One command replaces the whole manual GPU flow** (documented in
SESSION_HANDOFF.md §0, above the manual staging section it automates):

```bash
scripts/colab_bench.sh              # fresh T4: stage, ship-or-build cache,
                                    # run corpus benchmark, download results,
                                    # print catch/FP, harvest, stop
```

Mechanics: stage source (TritonBench included — the §0 trap), provision,
probe the VM (`vm_cache_tool.py probe` → Triton version, torch version, GPU
name, compute capability, sha256 of every `TritonBench/**/*.py`), look up
`.triton_cache_store/triton_cache_<triton>__<cc>__<srchash>.tgz`
(gitignored, on the dev machine). On a hit, upload (~21.7 MB) and run the
VM-side **guard**, which re-derives all five manifest fields on the live VM
and extracts only on a full match. On a miss or a STALE verdict, the run
proceeds cold and the wrapper **harvests** the newly built cache back into
the store, keyed for next time — so the second-ever session onward ships
automatically, with nobody remembering anything.

## 2. End-to-end validation, both paths, fresh VMs

| session | path | benchmark wall | regression (your_checker full) |
|---|---|---:|---|
| `kccflip` (fresh) | MISS → cold → harvest | **~252s** | **catch 1.0, FP 0.0** (40/40, 0/200) |
| `kccship2` (fresh, different VM) | HIT → guard SHIPPED → warm | **~79s** | **catch 1.0, FP 0.0** (40/40, 0/200) |

- The 252s cold / 79s shipped pair reproduces the measured 241.7s / 69.9s
  within poll granularity (30s polling; both runs carry it) plus ~9s of
  wall drift — the 2026-08-25 economics transfer to the automated path.
- **Zero new compiles through the shipped path**: cache file count after
  the `kccship2` run is 8786, identical to what the guard extracted, with
  1093 cubins — a genuine 100% hit on different physical hardware, same
  criterion the manual round used.
- Harvest cost is invisible in practice: packaging on the VM plus the
  21.7 MB download rides the session that already paid for the cold run.

## 3. The staleness guard, verified rather than assumed

The cache key is (Triton version, torch version, GPU name, compute
capability, kernel-source hash); the tarball carries the manifest, and the
VM guard re-checks every field against the live environment before
touching `$HOME`. `probes/staleness_test.py`, run on the real VM with the
real harvested tarball — **11/11**:

| test | outcome |
|---|---|
| T1 genuine tarball | SHIPPED, cache present |
| T2 doctored Triton version | STALE naming the field, nothing extracted |
| T3 doctored kernel-source hash | STALE naming the field, nothing extracted |
| T4 manifest missing entirely | STALE no-manifest, nothing extracted |
| T5 after a refusal, a fresh Triton kernel compiles and runs correctly, populating a new cold cache | pass |
| restore | genuine tarball ships again |

T4 caught a real bug on its first run: `tarfile.extractfile` raises
`KeyError` for an absent member rather than returning None, so the
intended no-manifest branch was dead code and the refusal came from the
catch-all instead (same safe outcome, wrong reason string). Fixed in
`vm_cache_tool.py` and re-verified — the probe exists precisely so that
"degrades safely" is a measurement, not a reading of the code.

Failure-direction note, stated because it is the design: every guard
error path — mismatch, missing manifest, unreadable tarball, probe
exception — prints STALE and exits 0, so the caller's only possible
degradation is a cold run. Triton's own content-addressed cache keying
would additionally just miss (not mis-serve) on stale entries; the guard
means that property is belt-and-braces, not load-bearing.

## 4. Scope and limits

- The wrapper automates the **standard corpus-benchmark run**. Probes,
  ablation arms and the adversarial search keep the manual flow (the
  handoff section covering it is unchanged, below the new one-liner). The
  cache store is per-dev-machine, keyed per (Triton, cc, source) — a new
  Triton version or edited kernel source degrades to one cold run and
  re-harvests.
- Wall times are 30s-poll-granular; the precise timing evidence remains
  the 2026-08-25 round's (four instrumented runs + controls). This round's
  numbers confirm transfer, not re-measure the mechanism.
- The `kccflip` cold run doubles as the post-forkserver-flip corpus
  regression (the flip touches only the adversarial-search executor, which
  the corpus benchmark never enters; the regression ran anyway, per the
  adoption protocol) — see `../forkserver_default_2026-08-28/`.
- The source hash covers `TritonBench/**/*.py` only. Kernels compiled from
  elsewhere (none in the current corpus path) would not invalidate the
  key; they would simply miss and compile cold, which is correct but worth
  knowing before adding new kernel source roots.

## Reproduce

```bash
scripts/colab_bench.sh -s anyname          # cold on first use, ships after
# staleness probe (on a session holding a harvested /content/triton_cache.tgz):
#   upload probes/staleness_test.py + scripts/ tree, then
#   PYTHONPATH=/content python3 /content/staleness_test.py
```
