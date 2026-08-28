# Session handoff

Written at the end of a working session on the three-layer kernel correctness
checker. Everything a new session needs is in this file plus the artifacts it
points to. **This conversation's history does not need to be replayed.**

---

## 0. GPU ACCESS — RESOLVED, READ THIS FIRST

**A GPU is available.** Access is via the Colab CLI (`google-colab-cli`),
authenticated against `jaydenvasquez1711@gmail.com`. Provisioning and remote
execution were both confirmed working end to end on 2026-08-20: a T4 session
came up in 13s and reported `torch 2.11.0+cu128`, `cuda_available: True`,
`Tesla T4` (capability 7.5), `triton 3.6.0`.

**Every item previously marked BLOCKED ON GPU has now been run** (#1, #2 and
§2.4, all on 2026-08-20). Their numbers exist and are in the repo — see §1's
table for where. Nothing in this file is GPU-blocked.

The old rule still applies to anything new: a number that requires a GPU does
not exist until someone runs it. The fix is now to run it, not to note it as
blocked.

### The working path

Credentials live in `~/.colab-home`, **not** the standard `~/.config`, because
`~/.config` is root-owned on this machine and the CLI cannot create its log dir
there. Every invocation must carry the prefix:

```bash
export HOME=~/.colab-home        # required, or the CLI dies with PermissionError

colab new --gpu T4 -s <name>     # provision (~13s); also L4, G4, H100, A100
colab upload -s <name> <local> <remote>
colab exec -s <name> -f <script> --timeout 300   # default timeout is 30s, usually too short
colab download -s <name> <remote> <local>
colab stop -s <name>             # idle sessions burn compute units — always stop
```

Sessions are cheap to recreate (13s), so do not hold one open across a review
or a long analysis pause. Batch the runs that need one session, then stop it.

### The one-liner for a corpus benchmark run (added 2026-08-28)

**Everything below in this section — staging, TritonBench, cache shipping —
is automated for the standard corpus-benchmark case:**

```bash
scripts/colab_bench.sh                    # fresh T4, auto cache ship/harvest
scripts/colab_bench.sh -s mysess -k       # named session, keep it alive after
```

It stages the source (TritonBench included), provisions the session, ships
the Triton cache from the local store `.triton_cache_store/` when one matches
this VM (keyed on Triton version + GPU compute capability + kernel-source
hash; ~9s to ship, saves ~170s — `verification_runs/triton_cache_2026-08-25/`
measured the 19:1 return, `verification_runs/cache_automation_2026-08-28/`
validated the automation), runs `run_benchmark.py`, downloads results to
`results_gpu/`, prints the your_checker catch/FP regression line, and — on a
cold run — harvests the newly built cache into the store so the *next*
session ships it. A stale or mismatched cache degrades to a normal cold run
via the VM-side manifest guard (`scripts/vm_cache_tool.py`); it can cost a
compile, never a wrong answer. The manual flow below remains correct for
non-standard runs (probes, ablation arms, adversarial search).

### Staging the corpus — `TritonBench/` is required and easy to miss

**Upload `TritonBench/` as well as the source tree, or the benchmark dies before
it starts.** `tritonbench_registry.build_corpus()` does
`importlib.import_module(f"TritonBench.reference.{ref_file}")`, so a VM without
it fails with `ModuleNotFoundError: No module named 'TritonBench'` at
`my_corpus` import time — before a single check runs. `TritonBench/` is a
top-level directory, not under `benchmarks/`, which is exactly why it gets left
out of a tarball built from the obvious subdirectories.

```bash
tar --exclude='__pycache__' --exclude='.venv' -czf kcc.tgz \
    verification benchmarks scripts tests TritonBench    # <-- TritonBench included
```

Then on the VM: extract to `/content`, `pip install litellm python-dotenv` (the
only two `pyproject.toml` deps Colab lacks — torch, triton and numpy are
preinstalled), and run with `PYTHONPATH=/content` so `verification/` resolves.

**`KernelBench/` is NOT needed** — despite being 9.9M and sitting next to
`TritonBench/`, it appears only in `kernel_adapter.py` docstrings describing a
calling convention. Nothing imports it. Leave it local.

Measured cost, whole staging step: **~21s** — 13.8s to `tar` the source tree off
Drive File Stream (263K) plus 0.3s for `TritonBench` (46K), 1.9s + 1.5s to
upload both, 2.5s to extract. The Drive read dominates and is a one-time cost
per session, not per file.

### Two settled findings — do not rediscover these

1. **The `jupyter-kernel-client==0.15.0` pin is required.** `colab_cli` 0.6.0
   declares this dependency *unpinned*; resolvers pick 1.0.1, which renamed
   `KernelClient` to `JupyterKernelClient`, and `colab exec` then crashes with
   `AttributeError: module 'jupyter_kernel_client' has no attribute
   'KernelClient'`. `colab new` still works, so the break surfaces only when you
   try to run something. On any fresh install:
   ```bash
   uv pip install --python <tool-python> "jupyter-kernel-client==0.15.0"
   ```

2. **Do not use `colab drivemount`.** Despite the repo living on Drive under the
   same account, mounting is the wrong path. It triggers a *second* OAuth grant
   against a different client (`947318989803-…`) requesting **full
   `auth/drive`** plus `drive.activity.readonly`, `drive.photos.readonly` and
   `peopleapi.readonly` — far broader than the `drive.file` scope the CLI itself
   holds — and it blocks on an interactive "Press Enter after you have granted
   access", so it cannot run unattended (it fails with `ValueError: mount
   failed` after a 120s timeout). `colab upload` is unattended and takes **~21s
   for everything the corpus needs** (see the staging section above for the
   breakdown and the `TritonBench/` requirement). Zero benefit, much broader
   grant.

### Operational risk: the VM can be reclaimed mid-run, without warning

**Assume any session can die at any moment and design runs so that costs
minutes, not the whole job.** Observed 2026-08-20: a T4 running the benchmark
was reclaimed mid-run. The CLI reported

```
[colab] Session 'kcc' appears to be lost (404/401). Cleaning up.
```

and `/tun/m/assignments` came back `{"assignments":[]}` — the VM was simply
gone, taking the in-flight run with it.

**It was not our workload.** The re-run was instrumented: RAM held flat at
~1.5GB of 12.9GB and GPU at ~129MiB for the whole run, and a fresh T4
provisioned 14s later. Not OOM, not quota exhaustion — free-tier reclamation.
Do not spend time debugging your script when this happens; re-provision.

Consequences to design around:

- **Never rely on a single long `colab exec`.** Launch under `nohup` writing to
  a log, return immediately, and poll. A lost session then costs one relaunch
  (~2 min including re-staging) instead of the entire run.
- **Download artifacts as soon as they exist**, not at the end of the session.
  Anything still only on the VM is lost when it goes.
- **Checkpoint anything long.** `run_benchmark.py` has no checkpointing — it
  writes its three outputs only at the very end, so a reclamation at 90%
  produces nothing. It completed in well under 10 minutes, which is why this
  was survivable; a longer job needs incremental writes first.

  **Worked example, 2026-08-21: a ~105-minute run was reclaimed at ~75% and
  cost 3 passes instead of the whole job.** `verification_runs/forkserver_2026-08-21/race_rate.py`
  is the pattern to copy for anything long. Three properties, all of which were
  needed: (1) **one JSONL line per trial, `flush()` + `fsync()` per pass**, so
  the on-disk state is never more than one pass stale; (2) **resume by reading
  its own output** — it takes `max(pass_idx) + 1` as its starting point, so
  relaunching after re-staging simply continues, and the arms stay balanced;
  (3) **a parallel watcher that downloads the artifact every 5 minutes**, so the
  local copy is the real checkpoint rather than the VM's. Without (3), (1) and
  (2) still lose everything the reclamation takes — the file has to leave the VM
  on a timer, not at the end.

  The **adversarial search is different and already safe on this axis**: it
  commits every proposal, execution, and verdict inside the iteration loop
  (`coordinator.py:277/296/310/319`), and `--resume RUN_ID` exists
  (`store.py:425`). Only the summary row and the result JSON are written at
  completion.

**Snapshotting the search DB: use the backup API, NOT a plain `colab download`.**
`store.py:151` sets `PRAGMA journal_mode=WAL`, so committed rows sit in
`search_history.db-wal` until a checkpoint. Mid-run, the main `.db` file was
**4096 bytes while the `-wal` held 976KB** — downloading just the `.db` yields a
near-empty file and a false sense of safety, which is worse than knowing you
have no backup. Take a consistent snapshot on the VM first, then download that:

```python
con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
out = sqlite3.connect("/content/snapshot.db")
con.backup(out); out.close(); con.close()   # includes WAL, safe against a live writer
```

Verified: the snapshot was 143KB and contained the rows the 4096-byte `.db` did
not. (Downloading `-wal` and `-shm` alongside the `.db` also works, but the
backup API is one file and is consistent by construction.)

### Still true: no CPU fallback

The TritonBench corpus is 29 real `@triton.jit` kernels and 15
`load_inline`-compiled CUDA kernels. There is **no CPU fallback path** —
`my_corpus.py` and `tritonbench_registry.py` both say so in their docstrings.
The dev machine is an Apple-silicon Mac, so anything touching the corpus runs on
Colab or not at all.

Do not substitute CPU approximations, simulated results, or estimates for a
number that requires a GPU. That rule survives unchanged — the fix now is to
run it, not to note it as blocked.

**Secondary environment note:** the repo lives on Google Drive File Stream.
Multi-file reads and `.venv` imports are extremely slow — importing `torch` from
`.venv` stalled for 10+ minutes at 0% CPU on network I/O and had to be
abandoned. Use `grep`/`sed` single-pass over many files rather than per-file
reads, and prefer stubbed-dependency tests (see §5) over importing `torch`.
This is also why the tar-and-upload step costs 13.8s: that is the Drive read,
and it is a one-time cost per session, not per file.

---

## 0.5 START HERE FOR THE RESULTS — `RESULTS_SUMMARY.md`

**Added 2026-08-21.** This file is a working log, ~1800 lines, and is the wrong
thing to hand anyone who wants the outcome. `RESULTS_SUMMARY.md` at the repo
root is a one-page assembly of what is established: the accuracy table against
all ten baselines, the corrected 29x latency multiple, the 36.2% -> 17.1% ->
0.0% false-positive sequence, the tolerance-invariance result with its float32
caveat, and the matmul prediction with its retraction.

**It contains no new claims.** Every figure names the artifact it came from and
was re-checked against disk when written. Where two artifacts disagree it says
so rather than picking one — see its §1 note on the faithful-gate FP units
(`results.json` 0.5%/1.0% vs this file's §1 prose, formerly 1%/2%), which
was **reconciled 2026-08-28**: the artifacts agree with each other and the
prose was a transcription error, corrected in place in both files.

---

## 1. Done — artifact locations

Contents are not re-summarised here; open the files.

| Item | Artifacts | Status |
|---|---|---|
| **#1** autokernel_gate baseline audit | `benchmarks/autokernel/AUTOKERNEL_BASELINE_AUDIT.md`, `benchmarks/autokernel/files/autokernel_faithful.py` | Analysis complete. Faithful re-implementation written and **construction-validated only** — argument arity/shape/dtype across 12 families via a stubbed `torch`. **RUN 2026-08-20 on a Colab T4.** Faithful gate: **80% catch / 0.5% FP (1 of 200)**; `rtol=0` variant: **80% catch / 1.0% FP (2 of 200)**; vs the old approximation's 68% / 18%. [CORRECTED 2026-08-28: this row previously said "1% FP / 2% FP", which matched neither artifact of the run — `results.json` records rates 0.005/0.010 with n_fp_checks=200 and frobenius per-op 1/5 and 2/5, internally consistent, and `results.md`'s 0%/1% is the integer rounding of the same values. The prose figures were a transcription error, not a different measurement; see theory_closure_2026-08-28/FINDINGS.md §4.] The correction is measured: +12pp catch, FP 18%→0.5%. The rtol reading is near-neutral (identical catch rate, +0.5pp FP). Numbers in `benchmarks/autokernel/files/results.md`. |
| **#2** per-check ablation instrumentation | `benchmarks/CHECK_ABLATION_FINDINGS.md`, `benchmarks/analyze_check_ablation.py`; instrumentation in `benchmarks/autokernel/files/checker_adapter.py`, `harness.py`, `run_benchmark.py`; compound-check decomposition in `verification/checker.py` (`_check_cross_shape`) and `verification/layer2_numeric_oracle/shape_generalization.py` (`check_weight_magnitude`) | **COMPLETE 2026-08-20.** Instrumented re-run landed; ablation generated from 1343 real check records into `benchmarks/CHECK_ABLATION.md`. The run exposed a reader defect that made the table unbuildable — fixed in three places with a permanent negative control (§3.3 of that doc, §5 instance 7). Locally verified by 44 + 16 assertions. Static findings in that doc §1 needed no GPU and are final. |
| **#3** per-operator layer attribution | `benchmarks/LAYER_ATTRIBUTION.md` (generated by `benchmarks/layer_attribution.py`) | **COMPLETE. No GPU needed** — built from the existing `results.json`. Key finding: structural and algebraic catch sets are a **strict subset** of numeric's; numeric alone accounts for **40/40 mutants across all 29 operators**, so its dominance is uniform, not concentrated in a few mutant-heavy operators. |
| **#6** causal_flash_attention non-hit root cause | `adversarial_results/CFA_NONHIT_ROOTCAUSE.md` — **note: repo root, not under `benchmarks/`** | Root-caused. Three causes: (a) 16 of 21 wired operators missing from `OPERATOR_CONTEXT`, (b) `build_feedback_hints` hardcodes a magnitude diagnosis regardless of actual failure mode, (c) `BUG_PATTERN_HINTS` missing the `"wrong_causal_mask"` key. **All three fixes landed in §2.1** — plus two more found while fixing them (`first_tile` had no hint either; the same magnitude misdiagnosis was independently baked into the refine-turn template). Verified by `tests/instrumentation/check_adversarial_search_fixes.py`. **Re-run COMPLETE 2026-08-20** (§2.4): 80 proposals, NO_HIT, with `not_caught: []` and `caught_no_gap: ["wrong_causal_mask"]` on all 80 — a clean negative result confirming §7.6's second predicted branch. |

Note on paths: `LAYER_ATTRIBUTION.md` is **generated** — re-run
`python3 benchmarks/layer_attribution.py` to regenerate it; do not hand-edit.

### Tooling — regenerate reports without a GPU

**Read this before starting a Colab session to test a reporting change.**

```bash
python3 benchmarks/regenerate_report.py        # rewrites results.md + results.json
```

`benchmarks/regenerate_report.py` re-renders `results.md` and `results.json`
from the existing `benchmarks/autokernel/files/results_raw.json`. Plain
`python3` — no venv, no torch, no corpus, no GPU, runs in under a second.

It exists because changing report *formatting* was costing a full benchmark
run. `build_markdown()` takes a `corpus` argument but uses it for only
`len(corpus)` and the set of `e["op"]` (`reporting.py:10-12`), both of which
are reconstructible from `results_raw.json`'s `mutant_results`. That was the
whole reason a re-run looked necessary. This happened on 2026-08-20: the
p50/p90/p99 latency columns landed in `harness.py`/`reporting.py`, but the
generated `results.md` kept the stale mean-only table until it was re-rendered
this way.

**It re-renders, it does not re-measure.** Every number it writes comes verbatim
from `results_raw.json`. If you changed the checker, a check, or the corpus, you
need a real GPU run — this would faithfully re-print the *old* numbers. Use it
only when what changed is how results are summarised or displayed.

Useful second job: **proving a summarize/reporting change is additive.** Diff
before/after and the change should touch only what you intended. That is how
the 2026-08-20 percentile work was verified — `results.md` came back
byte-identical apart from the new columns and their note, and `results.json`
gained 4 keys with 0 removed and 0 pre-existing values altered across all 11
systems.

The same trick applies to `layer_attribution.py` and `analyze_check_ablation.py`
above: all three read persisted artifacts, so none of them needs a GPU.

**`benchmarks/bug_class_theory.py` (item #4, added 2026-08-21) is a fourth, and
it is a different KIND of offline tool — worth knowing about before reaching for
a GPU session to answer a "why did this input hit" question.** It does not
summarise a stored result; it *recomputes* one. It rebuilds each proposed input
from its `TensorDescriptor` and runs the reference and every mutant in pure
Python, which is enough to reproduce **120 of 120** recorded verdicts across five
operators. Anything answerable from "what would this input have done" is
therefore answerable offline in about a second. It carries four controls that
must fire, printed against the trivial "never a hit" baseline (100/120) so a
control landing near it is not misread as a strong degradation.

---

## 2. In-flight — pick up here

**Status: §2.1, §2.2, §2.3 and §2.5 are COMPLETE (closed this session).
§2.4 is also COMPLETE as of 2026-08-20 — the GPU run landed. Nothing in §2 is
blocked.**

Original scoping note, kept because it still explains the numbering: §2.1-§2.3
needed no GPU. (Stated as section numbers, not "items 1-3" — this project
numbers its work items #1-#9, and project items #1 and #2 are exactly the
GPU-blocked ones.)

### 2.1 Fix #6's root causes — **COMPLETE**

All three fixes landed and verified by `tests/instrumentation/check_adversarial_search_fixes.py` (sections 1-3), including a negative control proving the startup assertion actually aborts. Details below kept for context.

Three fixes in `verification/adversarial_search/`:

- Write the **16 missing `OPERATOR_CONTEXT` entries** (`prompts/base.py:83`;
  currently only softmax, layernorm, matmul, flash_attention, rmsnorm). Each
  needs tensor keys, exact rank and shape convention, reference formula, and
  known bug patterns — matching the depth of the five that exist.
- Fix the **`build_feedback_hints` dispatch bug** (`executor.py`): it emits
  "Reduce input magnitude by 10x or use a simpler fill pattern" for *any*
  reference failure, including rank/shape/compile crashes where that advice is
  actively misleading.
- Add the missing **`"wrong_causal_mask"` key to `BUG_PATTERN_HINTS`**, plus a
  startup assertion that every mutant id in `MUTANT_PATHS` has a hint, so a
  silent `.get(..., "")` miss cannot recur.

**Before fixing: grep the existing search logs and
`adversarial_results/search_history.db` for other operators that received the
same wrong magnitude hint, and report that before changing anything.** The
prior session found the failure mode scales with how non-obvious an operator's
calling convention is — `instancenorm` shows the same rank confusion (9 rank-4
vs 9 rank-3 proposals, 83% reference-failure rate). The blast radius is probably
wider than one operator, and that scope should be known before the fix.

### 2.2 Fix the persistence gap in the verdict path — **COMPLETE**

`not_caught` / `caught_no_gap` / `mutant_records` added additively; `missed_mutants` preserved as their union so stored runs stay comparable. Verified by replaying all 260 recorded verdicts under both reconstructions and requiring exact **id-set** equality. Details below kept for context.

`ProposalVerdict` (`verification/adversarial_search/schemas.py:124`) persists
nothing per-mutant. `_evaluate_verdict`
(`verification/adversarial_search/coordinator.py:385-390`) collapses two
opposite outcomes into one `missed_mutants` bucket:

- the checker **did not catch** the mutant, and
- the checker **did catch** it but naive allclose also did, so there is no gap.

These are opposite results for the project's central claim and are recorded
identically. Split into `not_caught` vs `caught_no_gap` and persist per-mutant
`passed_checker` / `passed_naive`.

**This is the same bug class as item #2's original finding** — diagnostic detail
computed, then discarded before it reaches disk. **Reuse #2's persistence
mechanism if it fits** (structured records carried through an additive optional
field, raw dump alongside the summary; see `CHECK_ABLATION_FINDINGS.md` §2 and
the `_try`/`_summarize` docstrings in `checker_adapter.py`).

### 2.3 Scope-only sweep for other computed-then-discarded detail — **COMPLETE**

Sweep run; findings recorded below and promoted to §4 as candidate items. **Nothing was fixed**, as scoped.

Now that #2 and #6 are **two confirmed instances** of the same pattern, grep the
pipeline for others. **Scope and report only — do not fix in the same pass.**

**Shape A — computed, then discarded.** A value is computed, used for a boolean
verdict, and dropped before serialisation; or several distinct outcomes are
collapsed into one bucket before persistence. Confirmed: #2's per-check detail,
#6's `missed_mutants` bucket, and the adversarial-search feedback hints (never
persisted at all — §2.1's audit had to reconstruct them from `reference_passed`
plus `failure_summary`).

**Shape B — silently dropped on key collision.** A dict keyed on something that
is not unique, so a second entry overwrites the first with no error. **Confirmed
instance, carried forward from §2.1 and needing a real fix, not the stopgap it
currently has:** `BUG_PATTERN_HINTS` is keyed by mutant id alone, but mutant ids
are **not unique across operators** — `flash_attention` and
`scaled_dot_product_attention` both have a mutant named `wrong_mask`, requiring
opposite advice (an off-by-one causal mask that should exist, versus a mask
applied where none should be). Adding the second entry silently replaced
flash_attention's working hint. It was caught only by a duplicate-key line
count (31 entry lines vs 30 dict keys), not by any test.

  *Current state:* one merged entry worded to cover both readings, plus an
  in-file warning. That is a stopgap — it degrades hint specificity for both
  operators and breaks again the moment a third `wrong_mask` variant appears.
  *Real fix:* key by `(operator, mutant_id)` with fallback to the bare id; the
  lookup in `format_first_turn` already has `operator` in hand. Also add a
  duplicate-key guard, since Python dict literals cannot report collisions.

Shape B is worth sweeping for alongside Shape A: both are silent data loss at a
persistence or lookup boundary, and neither raises.

**Search surfaces — broader than the codepaths already touched.** §2.1's grep
looked at check-dispatch and persistence code and concluded the hardcoded
magnitude misdiagnosis lived in one place (`executor.build_feedback_hints`). It
did not. The **same wrong advice was independently baked into the refine-turn
template** at `prompts/base.py:534` ("If the reference failed, reduce
magnitudes"), where it re-asserted the misdiagnosis on every turn and would have
silently undone the executor fix. It was found only while editing that file for
an unrelated reason.

So the sweep must cover, at minimum:

- **Prompt and refine-turn templates** (`verification/adversarial_search/prompts/`)
  — f-strings and literal instruction text can encode a diagnosis, a default, or
  a fallback just as much as a dispatch function can, and they are not reached by
  grepping for the function that "owns" the behaviour.
- Check-dispatch and verdict code (the original §2.1/§2.2 surface).
- Persistence and serialisation boundaries (`to_dict`, `summarize`, DB writes).
- Dict/registry literals where the key space is enumerable (Shape B).

**The generalisable lesson:** grepping for a behaviour's *owning function* finds
one instance. Grep for the **behaviour's text and its defaults** — the advice
string, the fallback value, the `.get(k, default)` — across the whole subsystem,
including non-executable prose that reaches a model.

---

### §2.3 SWEEP RESULTS (scope only — nothing below is fixed)

**A1. `KernelExecutionResult.check_results` is never persisted. [highest value]**
The executor computes full per-check pass/fail/details for the reference *and*
every mutant on every proposal (`executor.py:167-184`), uses it transiently for
`passed_checker`, `failure_summary` and hints, then drops it. The history DB has
four tables (`runs`, `proposals`, `verdicts`, `memory_items`) and no
execution-results table; `save_verdict` writes the verdict only. `mr.error`
(type, message, traceback snippet) is discarded the same way.
*This is #2's defect in the adversarial-search subsystem*, and it is why the
causal_flash_attention post-mortem had to reconstruct conclusions from proposal
shapes. §2.2 partially remedies it — `mutant_records` now keeps the two
booleans per mutant — but *which check* failed, and with what detail, is still
lost on every run. Fix is the same shape as #2: an additive per-proposal record
plus a raw dump alongside the summary.

**A2. Feedback hints are never persisted.** (Confirmed, already noted in §2.1.)
`build_feedback_hints` output goes into the prompt and vanishes; the §2.1 audit
had to reconstruct which proposals got which advice from `reference_passed`
plus `failure_summary`.

  **Largely mooted by A1 — probably needs no separate work.** A hint is a pure
  function of `check_results` via `_diagnose_reference_failure`, so now that the
  check results are stored per execution, the hint any proposal received is
  *recomputable* rather than lost. Persisting the hint text as well would only
  guard against the diagnosis function itself changing between runs. Worth a
  decision, not an implementation, unless that versioning matters.

**B1. `BUG_PATTERN_HINTS` keyed by mutant id alone.** Confirmed instance
(`wrong_mask`), currently carrying a stopgap. Real fix specified above.

**B2. Seven parallel operator tables with no enforcement. [new]**
`schemas.REQUIRED_TENSOR_KEYS`, `materializer._OPERATOR_TENSOR_KEYS`,
`prompts.OPERATOR_CONTEXT`, `runner._REFERENCE_MAP`, `runner._MUTANT_MAP`,
`executor.SPEC_MAP`, `executor.FUNC_NAMES` — all 21 keys, and **verified aligned
today**. Nothing enforces that. Adding an operator means editing seven places,
and a miss fails differently in each: `validate_proposal` hard-rejects, the
materializer raises "Unknown operator", the executor raises `KeyError` — but
`OPERATOR_CONTEXT` **falls back to a bare label and silently degrades search
quality**, which is exactly how causal_flash_attention burned 120 proposals.
`materializer._OPERATOR_TENSOR_KEYS`'s own docstring says it "must match
exactly" — a comment doing an assertion's job. Fix: one import-time cross-check,
same pattern as `validate_bug_pattern_hints()`.

**P1. Residual template heuristic.** `prompts/base.py:549` still asserts a
default diagnosis in prose — *"if large values failed, try structural
patterns"*. Same species as the magnitude bug (advice baked into a template
rather than derived from the failure), but lower severity: it is conditional and
no longer contradicts the per-failure hint. Confirmed no other hardcoded
magnitude directives remain anywhere in the subsystem.

**P2. Silent `.get()` defaults, low severity.** `worker.py:161-162` defaults
`rationale` and `predicted_failure_mode` to `""`, so a malformed LLM response
parses as a valid proposal with no stated hypothesis and nothing reports it.
Benign today because both fields are advisory, but it is the same silent-default
shape and would hide a systematic parsing regression.

**Ranking for a fix pass:** A1 (largest diagnostic loss, known-good fix
pattern), then B2 (cheap assertion, prevents a repeat of the exact CFA failure),
then B1 (needs a lookup-signature change), then P1/P2 (cosmetic/low).

### 2.4 CFA re-run at normal budget — **COMPLETE (2026-08-20)**

Ran on a Colab T4: `causal_flash_attention`, default flags (`--max-iter 20`,
`--strategy beam`, `--workers 4`) = **80 proposals**, the normal budget, not the
120 the original burned. 556s wall time.
Artifacts: `adversarial_results/cfa_rerun_2026-08-20/` (result JSON, history DB,
run log). The Aug-6 `search_history.db` and `CFA_NONHIT_ROOTCAUSE.md` were left
untouched.

**Result: NO_HIT — and this time that is an answer, not an ambiguity.**

`CFA_NONHIT_ROOTCAUSE.md` §7.6 predicted, in advance, one of two outcomes:
convergence in the context-equipped range, *or* a clean negative result because
the bug is gross enough that naive testing already catches it. **The second
branch is what happened, unanimously across all 80 proposals:**

| field | value | count |
|---|---|---|
| `not_caught` | `[]` | 80 / 80 |
| `caught_no_gap` | `["wrong_causal_mask"]` | 80 / 80 |
| `hit_mutants` | `[]` | 80 / 80 |

The checker caught `wrong_causal_mask` on **every** proposal, and naive allclose
caught it too — so there is no adversarial gap to find. The search did not fail;
it correctly reported that the thing it was looking for does not exist.

**This closes the withdrawn claim in §2.5 / `CHECK_ABLATION_FINDINGS.md` §3.2
with direct evidence rather than inference.** Before §2.2, "checker missed it"
and "checker caught it, no gap" both collapsed into `missed_mutants` and were
indistinguishable — which is exactly how the original `hit_mutants: []` reading
went wrong. The split fields now separate them, and `not_caught` is empty in all
80. §2.1-§2.3 are confirmed working end to end on real data.

**Unplanned finding — §3.0 gets 20 more data points.** The *reference* (correct)
kernel failed on **29 of 80 proposals (36%)**:

- **20× `kernel_executed`** — precisely the shift-invariance false positive
  documented in `CHECK_ABLATION_FINDINGS.md` §3.0, which had 30 occurrences
  across all prior history (25 on this operator). This single run adds 20 more,
  raising the priority of that fix.
- 9× `nan_inf` + `dtype_preserved` together.

A correct kernel failing Layer 1 on more than a third of proposals is a
first-order problem for the project's 0%-false-positive headline, and it is now
measured on a clean run rather than inferred from search history.

### 2.5 Update pending-corrections tracking — **COMPLETE**

Both entries written to `benchmarks/CHECK_ABLATION_FINDINGS.md` §3.2.

The pending-corrections table lives in `benchmarks/CHECK_ABLATION_FINDINGS.md`
§3. Add two entries:

1. **The `hit_mutants: []` self-correction.** An earlier claim in this project —
   that `hit_mutants: []` across all 120 causal_flash_attention proposals proved
   the mutant never failed the checker — **does not follow** and was withdrawn.
   See `CFA_NONHIT_ROOTCAUSE.md` §4. `benchmarks/LAYER_ATTRIBUTION.md` already
   carries the corrected wording; anything else repeating the original claim
   needs the same fix.
2. **#8(b) constraint.** Extending the `InputProposal` schema for the 8 excluded
   operators (`cross_entropy`, `groupnorm`, 6 pooling ops) **must include
   writing their `OPERATOR_CONTEXT` entries in the same change, not afterwards.**
   Those are exactly the operators with the least obvious conventions — int64
   class-index targets, `num_groups`, `kernel_size`/`stride`/`padding`. Shipping
   the schema without the context entries would reproduce #6's failure eight
   more times.

---

## 3. Known-wrong claims still sitting in `BENCHMARK_RESULTS.md`

**Left uncorrected deliberately**, per the user's instruction to hold edits to
that file until re-run data lands, so §4 is edited once rather than twice.
**Do not "helpfully" fix these without checking with the user first.**

1. **§4's autokernel_gate FP-mechanism sentence.** Blames the
   "fixed-tolerance adversarial-stability stage" plus a "bitwise determinism
   check false-positiving on `frobenius_norm`'s atomic-add." Both are wrong.
   The actual cause was **two arity/dtype bugs in the re-implementation's own
   input generator** — every FP was an exception, not a tolerance comparison,
   and `frobenius_norm` has 0% FP with no operator failing the determinism
   stage. Traced in full in `AUTOKERNEL_BASELINE_AUDIT.md` §3; both bugs are
   fixed in `autokernel_faithful.py`.

2. **§8.5's "22 checks" coverage-depth figure**
   (`BENCHMARK_RESULTS.md:353-367`, table row
   `| **your_checker** | **22** | **structural (8) + numeric (6) + algebraic (8)** |`).
   Contradicted by item #2's measured counts: **62 Layer-2 check instances**
   (4 fixed + 58 adversarial) and **69 Layer-3 property instances**.

   **This is not a find-replace.** It is not yet determined whether this is a
   units mismatch (check *types* vs check *instances*) or a real error.
   `benchmarks/sota_checks_registry.py` explicitly says its algebraic rows are
   "catalogued by property TYPE," which points at units — **but its own type
   counts do not reconcile either**: it lists 8 algebraic types against **29
   distinct property names** measured, and 6 numeric types against **4 fixed
   checks + 36 distinct adversarial names** measured. So the by-type reading
   does not rescue the figure as it stands. **Resolve the units question first,
   then correct — and note that the same figure is repeated in §8.5's closing
   prose ("the win is 22 checks *and* 100% recall...").**

3. **The 68% / 18% autokernel_gate headline itself.** The 18% false-positive
   rate will very likely drop **toward 0%** once the faithful re-implementation
   runs, and the 68% catch rate will likely **rise** (the old gate was 100x
   looser on tolerance, ran no real shape sweep, and omitted a whole stage).
   **Do not cite 68%/18% anywhere further until that lands.** This is the
   project's single largest claimed margin and it currently rests on two bugs in
   our own re-implementation.

### KNOWN AND INHERENT — `frobenius_norm` determinism FPs are flaky by design

**This is not a bug, and it is not a regression. Do not re-investigate it as
one.** Logged here because it looks exactly like a real defect the first three
times you meet it.

`frobenius_norm`'s reference kernel uses `tl.atomic_add` for cross-block
reduction. Atomic adds across concurrent thread blocks are correctly
synchronised — no race, no UB — but floating-point addition is non-associative,
so the same partial sums arriving in a different order change the result's last
few ULPs. **Any check that compares runs bitwise will therefore flip randomly
on this operator, run to run, on identical inputs.**

Measured across three independent 2026-08-20 runs, `autokernel_gate`'s bitwise
determinism check flagged `frobenius_norm` **1/5, 2/5 and 0/5** reference
trials. Nothing in the code changed between them.

**Consequences for anyone reporting numbers:**

- **Never report an FP rate touching `frobenius_norm` as a single-run point
  estimate.** Run it several times and give the range, or say explicitly that
  the figure is one draw from a distribution. A single run will silently
  over- or under-state it by a couple of percentage points of the corpus.
- This affects the **bitwise** checkers: `autokernel_gate` and both
  `autokernel_gate (faithful)` variants. This project's own checker is immune —
  `check_determinism` was already changed to a tight tolerance rather than
  `torch.equal` precisely because of this operator
  (`verification/layer1_structural/runtime_guards.py`, the `FIXED:` comment).
  `your_checker (*)` showed **zero** determinism FPs in every run.
- When diffing two benchmark runs, expect `frobenius_norm` determinism rows to
  differ **in both directions** and treat that as noise. Real regressions do not
  flip back and forth.

---

### LAYER REORDER (2026-08-20) — numeric now runs LAST

`KernelChecker.run` executes **structural → algebraic → numeric**, with numeric
reached only when neither cheap layer caught the bug. Labels swapped to match:
**algebraic is Layer 2, numeric is Layer 3.**

**Why:** numeric is the expensive layer (warm p50 **15.71ms** vs algebraic
**1.17ms**, structural **3.97ms**) and the layers short-circuit.

**Why it is safe:** the catch sets are nested — structural (4 of 40) ⊂
algebraic (18) ⊂ numeric (40) — so anything algebraic catches, numeric would
have caught. The reorder changes which layer *reports* a catch, never whether
there is one. **This is an empirical property of the current corpus, not a
theorem:** `tests/instrumentation/check_layer_order.py` asserts the containment
and fails loudly if a future check breaks it, because the whole safety argument
rests on it.

**Expected benefit is BOUNDED — never quote it without the trial mix.**
Modelled on the 240-trial benchmark (40 mutant + 200 reference):

| class | n | delta |
|---|---:|---|
| caught by algebraic, not structural | 14 | **−74%** (19.68 → 5.14ms) |
| numeric-only mutants | 22 | +1.17ms each (pay algebraic first) |
| **reference (correct) kernels** | **200** | **zero change** |
| net on this corpus | 240 | **≈ −2.6%** |

Correct kernels pass every layer, so nothing short-circuits and order is
irrelevant for them — which is why 200 of 240 trials see no benefit at all. A
mostly-buggy workload would gain much more; a validation workload gains
~nothing.

**MEASURED 2026-08-21 — report this honestly, it is not a clean win.**
Corpus total **6.750s → 6.553s = −2.9%** (model said −2.6%), verdicts identical
across all four `your_checker` systems; the only diffs were 3 `frobenius_norm`
FP flips in the two `autokernel_gate (faithful)` baselines, which is the known
inherent bitwise-determinism flakiness documented in §3 and not a reorder
effect.

**The latency number is NOT confirmed.** The three single-layer ablations
cannot be affected by a reorder — each runs one layer unconditionally — yet
they moved **+7.2% / +6.1% / −2.7%** between the same two runs. Run-to-run
noise exceeds the effect, so −2.9% is consistent with the model and also
consistent with zero. Do not cite it as a measured speedup. The experiment is
underpowered by construction, since 200 of 240 trials are reference kernels the
reorder cannot help.

**The MECHANISM is confirmed, and that evidence is timing-independent:** of the
14 mutants caught by algebraic-but-not-structural, numeric checks ran on
**14/14 before** and **2/14 after**. Twelve now skip the expensive layer
outright. That is the reorder working; the aggregate latency is simply too
small a signal to resolve on this trial mix.

**Carry this into any re-run — the compile-order caveat:** #7a step 2 showed
JIT compilation is charged to whichever path first touches a kernel/constexpr
pair (an 84% misattribution). Numeric running last makes it the layer most
likely to inherit that confound: on a correct kernel all three layers run, and
numeric's cross-shape sweep is precisely what visits novel shapes. Keep
`harness._warm` on, and **if numeric's measured share jumps after the reorder,
suspect the confound before concluding numeric got slower** — nothing about
numeric's work changed.

**`layer_convention` marker.** New results files carry
`layer_convention: "structural_algebraic_numeric_v2"` (`harness.py`). A stored
`layer: 2` means NUMERIC before this date and ALGEBRAIC after. **Scope note so
nobody over-reacts:** no current reader keys on the numeric layer value —
`analyze_check_ablation.py` and `layer_attribution.py` both attribute by SYSTEM
NAME, which did not change. The marker is for future analysis and for reading
old artifacts by hand, not a repair of a live breakage.

**Directory names deliberately unchanged.** `verification/layer2_numeric_oracle/`
and `verification/layer3_properties/` still encode the old numbering in import
paths used across ~17 files. Renaming is mechanical but touches every importer
and buys nothing functional. The adapter's internal helpers were renamed
descriptively instead (`_run_structural` / `_run_numeric` / `_run_algebraic`) —
encoding position in an identifier is what made this confusing in the first
place.

---

### PERTURBATION BATCHING (2026-08-21) — Stage A worked, Stage B did not

`check_perturbation_tolerance` ran 20 samples in a loop, each costing a kernel
launch **and** a `.item()` GPU sync (~159 calls per 40-mutant pass → ~19,000
launch+sync pairs per benchmark).

**Whole-system latency was unmeasurable and is not reported.** Systems that
cannot be affected drifted **−30.2% to +2.2%** within-session. Per-check
`duration_ms` isolates it:

| | median | total | vs prior |
|---|---:|---:|---|
| baseline | 4.22ms | 701ms | — |
| **Stage A** — drop the per-sample `.item()` | **3.33ms** | **559ms** | **−21.1% / −20.2%** |
| Stage A+B — also batch the kernel calls | 3.33ms | 601ms | **+0.0% / +7.4%** |

Control (non-perturbation checks, same system, same runs): −3.6% then **+8.0%**.
So Stage A clears the control drift; **Stage B's +7.4% IS the control drift**,
and its median change is exactly zero.

**Why: the syncs were the serializer, not the launches.** `.item()` stalled the
pipeline 20 times per call, so launches could never overlap. Remove the syncs
and they pipeline on their own, leaving batching nothing to recover.

**RECOMMENDATION — flip `batch_samples` to default off.** Stage B is
implemented, bit-identical and per-operator gated, but carries a real
silent-wrongness surface for **zero measured gain**: batching a global-reduction
operator loosens `adaptive_tol` rather than erroring (measured on a
frobenius_norm stand-in: 0.001218 → **0.778163, 639x looser**, no exception).
Keep the machinery for a future case (much larger `n_samples`, launch-bound
operators); do not keep it enabled on this evidence. Verdicts identical across
all four `your_checker` systems in all three runs.

---

## 4. Not started

Items #4-#9 are the original numbered work items. The **A1/B2/B1/P1/P2** rows are new candidates surfaced by §2.3's sweep — they are scoped and ranked, but **not approved for implementation**. Deciding whether to pick one up, or to clear and start fresh, is an explicit open decision.

### ✅ CLOSED 2026-08-21 — §3.0 fixed in code. (Was: resolved by documentation.)

> **Both halves are now done, in the order recorded here.** 2026-08-20 took the
> documentation path and scoped the claim. 2026-08-21 fixed the underlying
> `check_kernel_executed` defect — see **§6.1** for the mechanism, the shipped
> ladder, and the T4 numbers. Doing it as its own pass is what surfaced that
> §3.0's recommended perturbation rescues **0 of 20**; folded into the
> documentation change, it would have shipped as a fix and changed nothing.
> **The MVP boundary held throughout: B2/B1/P1/P2 remain parked.**
>
> The 0%-FP scope caveats from 2026-08-20 were **left in place** — still true,
> and relaxing them needs the re-run in §7 item 1.

**The 2026-08-20 call, kept for the record: documentation path, not code path.**
The 0%-FP claim is now qualified with the input distribution it was measured on,
everywhere it appears. `check_kernel_executed` was **not** touched *on that date*
and remained an open, scoped code defect for its own future session — which is
exactly what 2026-08-21 was. The MVP boundary stands; B2/B1/P1/P2 remain parked.
(`runtime_guards.py` did carry uncommitted changes in `git diff` as of
2026-08-20, dating from 2026-08-06 and concerning `check_determinism`'s
atomic-add tolerance — nothing was added to that file on 2026-08-20. The
probe-ladder changes now in that file are from 2026-08-21.)

Sites qualified: `BENCHMARK_RESULTS.md` §1 (claim + scope note above the headline
table), §8.3 (rule-of-three bound, with the sampled population stated), §7, §11;
`benchmarks/CHECK_ABLATION_FINDINGS.md` §0 and §3.0 (status note).

**Load-bearing distinction, preserved in every edit:** the corpus 0% and the
adversarial 36% are **not contradictory** — different input distributions, and
the published corpus numbers stand unchanged. The exposure was rhetorical: "0%
false positives" read as "on any correct kernel". Nothing was retracted; the
claim was scoped.

The original reasoning is kept below, because the decision only makes sense with
the case for the alternative visible.

---

**Original framing (superseded by the decision above):** Everything else in
this section is correctly parked behind the "stop adding scope" boundary. This
one was flagged separately because the 2026-08-20 CFA run changed its evidentiary
status, and the boundary decision was made before that data existed.

**What changed.** `check_kernel_executed`'s false positive on shift-invariant
operators (`CHECK_ABLATION_FINDINGS.md` §3.0) previously rested on **30
occurrences across all recorded search history** — real, but accumulated over
many runs and inferred from a history DB. The clean 80-proposal run adds **20
more in a single run**, and puts a number on it: the **reference kernel — the
correct one — failed 29 of 80 proposals, 36%**, of which 20 are
`kernel_executed`.

**Why it may outrank the boundary.** This is not a new feature or a code-quality
sweep; it is measured evidence against the project's central claim. The headline
is **0% false positives**, and `benchmarks/autokernel/files/results.md` still
reports 0% FP for `your_checker (full)` across the corpus. Those two facts are
not yet reconciled:

- The corpus benchmark and the adversarial search exercise the checker on
  **different input distributions**. The search deliberately generates
  adversarial inputs; the corpus uses fixed ones. A 0% FP on the corpus and a
  36% reference-failure rate under search are not arithmetically contradictory.
- But they are *rhetorically* contradictory, and the paper's claim is the
  general one. "0% false positives" invites the reading "on any correct kernel",
  which this data contradicts for at least one operator family.

**The narrow question to decide:** is §3.0 a **defect fix** (inside the MVP
boundary — the checker is wrong and the headline number depends on it) or
**scope expansion** (outside it — a known limitation to document and defer)?

Arguments each way, stated plainly:

- **Fix now:** it is a Layer-1 soundness bug with a measured 36% rate, on the
  layer producing the headline. §3.0 already contains the diagnosis and the
  mechanism, so this is implementation, not investigation. Shipping a
  0%-FP claim while holding contradicting measurements is the kind of thing
  §5's pattern list exists to prevent.
- **Defer:** the boundary was set deliberately; the corpus 0% figure is not
  itself falsified; and the cleanest resolution may be a **documentation**
  change (scope the claim to the corpus and its input distribution) rather than
  a code change, which is far cheaper and does not reopen the checker.

**Recommended if the answer is "fix":** scope it to `check_kernel_executed`
alone — a shift-invariance-aware perturbation, or gating the check off for
operator families where it is unsound. Do **not** let it reopen the wider §2.3
sweep (B2/B1/P1/P2 stay parked regardless).

**Recommended if the answer is "defer":** the minimum honest step is to qualify
the 0%-FP claim wherever it appears with the input distribution it was measured
on, and cross-reference §3.0. That is a §3-style documentation correction, not
new scope.

| Item | Reason |
|---|---|
| **#4** adversarial-input-to-bug-class theory table | **✅ DONE 2026-08-21.** `benchmarks/BUG_CLASS_THEORY.md`, generated by `benchmarks/bug_class_theory.py` — **no GPU, no torch, no numpy, ~1s**. Do not hand-edit; re-run it. The blocking condition was satisfied: #6's fixes landed and §2.4/§7-item-1 re-ran, so the history DB now carries the split verdict fields this needs. **Headline: the search is adversarial against the BASELINE, not against the kernel.** Rebuilding each proposed input from its descriptor and running reference + mutants in pure Python predicts **120 of 120** recorded verdicts from `is_hit = reference_valid AND naive_allclose misses some mutant` — a formula with **no term for the checker catching the bug**, because the checker caught it on every valid proposal (falsifier: reference-valid + naive-blind + not-a-hit, **0 of 120**). Two mechanisms, not one: **exact masking** (mutation semantically inert — `gamma≡1`, no ties — error exactly 0) and **tolerance straddling** (error nonzero but under `atol=1e-3, rtol=1e-2`). §5 of that file lists the limits; the biggest is coverage, 5 of 9 operators and 20 of 23 hits, with matmul/attention deliberately excluded rather than transcribed. |
| **#5** math/theory writeup | **PARTIAL — substantial draft landed 2026-08-21: `benchmarks/NUMERICAL_THEORY.md`.** Derives `BUG_CLASS_THEORY.md`'s empirical result from the operators' arithmetic rather than restating it. **§2 is the load-bearing section and is standalone-legible:** the *tolerance-invariance* claim — the baseline is blind for EVERY tolerance pair `(a,r) ≥ 0` iff the residual `R(x) ≡ 0` — with proof, a decision procedure (re-run at `a=r=0`), and the measured split **9 exact-masking / 11 tolerance-straddling of 20 simulated hits**. The consequence is the part worth carrying into the paper: **the honest headline is two numbers, not one.** Class T is contingent on `rtol=1e-2` and a reviewer may fairly answer "then tighten your baseline"; class E is unconditional — no allclose test at any tolerance or precision can catch it — and is the only claim that argues for property checking as a *category*. **Precision caveat that must travel with the counts:** the simulation is float64, the kernels float32, and the fp16-absorption boundary differs (v>40.9 vs v>21.5), so the on-hardware split is likely **11/9**, not 9/11. Derivations complete and checked for `softmax` (both mutants), `gelu`, `instancenorm`, `rmsnorm`, `layernorm`, `argmax`, and **`matmul` (§5, added the same evening — 8 of 8 cells predicted in both directions)**. **OUT-OF-SAMPLE TEST RUN 2026-08-21 — the first prediction in this project made BEFORE the data rather than fitted to it, and it cut both ways.** Artifacts: `verification_runs/matmul_prediction_2026-08-21/`. **CONFIRMED:** §5.3 predicted that zeroing `A[:, K/2:]` makes `matmul:partial_k_reduct` class E (residual identically zero) and therefore a hit on the first attempt. It was **credited on 3 of 3 zeroed proposals and 0 of 1 un-zeroed control** — the control holds shape, fill and strides fixed and varies only the zeroing, so the masking is attributable to the residual and nothing else. `swapped_strides` also behaved as derived (visible whenever `A.stride() != B.stride()`). **FALSIFIED in the same run:** `skip_boundary_tiles` came back masked at `M=N=100`, where §5's tile-alignment condition says it should be plainly visible. A second hypothesis (all-ones fills make `C` constant, so an out-of-bounds store rewrites the same value) motivated a fourth proposal with non-constant `C` — **it stayed masked there too.** Two mechanisms proposed, both dead; **no third is offered, and the condition is OPEN.** **This retracts part of §5.1's "8 of 8 cells":** every recorded matmul proposal is simultaneously tile-aligned AND constant-output, so the data could never have distinguished those conditions — the skip_boundary cells were right for the wrong reason. Same confound shape as "9 of 9 softmax hits are patched" (74% of non-hits are too) and the retracted non-power-of-two diagnosis. **The lesson, and it is the night's cleanest instance: a condition that fits every observation is not thereby the operative one, and only a constructed separating input can tell. One GPU run overturned a claim that looked fully supported.** **Still open:** `skip_boundary_tiles`'s masking condition, the two attention operators (§8), no related work, §7 is an explicit sketch. |
| **#7** latency work | **✅ #7 IS CLOSED — NO OPEN SUB-THREADS.** #7a (metric fix, root-cause, both measurement defects fixed and re-measured) and #7b (search-latency measurement) are both complete; the `autokernel_gate (faithful)` p90 tail was investigated and closed as expected behaviour; #7a step 3 was deprioritised when its premise dissolved. **One actionable follow-up exists and is NOT approved and NOT started:** reduce subprocess-spawn cost in the adversarial search — a persistent executor pool, `fork` where CUDA permits, or batching a proposal's two kernels into one subprocess (that alone halves spawn count). It is **new work requiring its own go-ahead**, exactly as #7a step 3 and the faithful-gate tail did before they were closed out. Do not start it as a continuation of #7. Detail below. **#7a step 1 COMPLETE and banked (2026-08-20); #7a step 2 and #7b NOT started — each needs its own go-ahead, not automatic continuation.** Planned in plan mode as flagged. **Step 1 (metric fix):** `harness.summarize()` now emits `p50/p90/p99/max_latency_ms` alongside an unchanged `mean_latency_ms`; `reporting.py` surfaces them with a "read p50, not the mean" note; `results.md`/`results.json` regenerated offline from `results_raw.json` (no GPU); `BENCHMARK_RESULTS.md` §8.1.1 added and its two mean-driven prose claims caveated. **Why it came first:** the published mean is heavy-tailed and unreliable — mean/p50 up to 33.5x, and the slowest 10 of 240 trials are 49% of total time — so optimising against it would have targeted the wrong thing. **Profiling baseline now in hand, no re-derivation needed:** checker cost concentrates by operator (flash_attention 18.3%, avg_pool1d 9.7%, matmul 7.7%; top 10 ops = 71.3%), and for #7b the CFA run shows kernel execution is only **1.4%** of search wall time — 98.6% is LLM + orchestration, so the obvious target there is the wrong one too. **Step 2 COMPLETE (2026-08-20) — the tail is Triton JIT compilation, not checker work, and it is a measurement-order artifact.** `systems` is the OUTER loop (`harness.py:82`): `your_checker (full)` runs 8th and its ablations 9th-11th, so `full` pays every compile the checker's cross-shape sweep triggers and the ablations then run against a warm cache. Measured: `full` totals 43.4s vs 18.3s for all three ablations combined — **25.1s excess = 58% of full's time** — despite `full` short-circuiting and running a SUBSET of their checks; it exceeds the ablation sum in **36 of 40 entries**. `avg_pool1d/wrong_divisor` is 4218ms in `full` vs 122ms across all three ablations: **the same checks, 35x cheaper when run second.** Answered from existing data — no GPU run needed. **Two measurement defects follow, both caveated in `BENCHMARK_RESULTS.md` §8.1, neither fixed:** (a) the layer-cost comparison ("numeric only at 64% of the latency") is confounded by dict order; (b) `harness.allclose_system` (`harness.py:33-40`) times only the numpy comparison — kernels run BEFORE its timer — so "354x faster" compares comparison-time against full-pipeline-time. Catch/FP rates are unaffected throughout; only latency multiples are in question. Per-check `duration_ms` was added to `_try` anyway (negative control in `check_item2_instrumentation.py`, verified to fire) and will enrich the next real run. **BOTH MEASUREMENT DEFECTS NOW FIXED AND RE-MEASURED (2026-08-20).** `harness._warm()` warms the kernel cache per (system, entry) before timing, and all seven systems share one timer scope (input generation through verdict). Result: **84% of `full`'s measured time was Triton JIT compilation** — corpus total **42.4s cold to 6.9s warm**, mean/p50 **5.6x to 1.2x**. The "354x faster than allclose" figure was wrong by ~8x; like-for-like it is **29x**. "numeric only at 64% of full's latency" survives as **75% of p50** — the finding held, the multiple was measured on a confounded quantity. Canonical numbers: `BENCHMARK_RESULTS.md` §8.1.2; cold-cache comparison retained at `benchmarks/autokernel/files/results_raw_cold.json`. **Semantics verified, not assumed:** zero verdict changes across every cold/warm comparison. Residual check-level jitter (7-11 of 1343) is pre-existing GPU float non-determinism — **two independent COLD runs differ by 6 of 1343** with warming on neither side. `_warm` also restores torch's global RNG (warmup does advance it via `torch.randn` in `shape_generalization.py:73/125`); that restore is principled but did **not** reduce the jitter, which is how we know the residual is float non-determinism rather than RNG drift. **#7a STEP 3 (optimisation) — DEPRIORITISED, NOT PAUSED. Do not restart it without new evidence.** Its premise dissolved: step 3 existed because the checker looked expensive, but step 2 showed **84% of that was Triton JIT compilation** attributed by dict order. The honest steady-state is **~21ms p50 per check, 6.9s across the whole corpus**. No case has been made that this needs optimising, and the number that motivated the work was measuring something else. Reopen only if a concrete requirement appears (a latency budget, an interactive use case) — not on the strength of the old pre-warming figures, which are superseded and should not be cited. **`autokernel_gate (faithful)` p90 tail (326ms vs 11.85ms p50) — INVESTIGATED AND CLOSED, expected behaviour, no action.** Not a defect and not a duplicate of §3's faithful-gate thread, though it shares that thread's origin. The cost is **78% attributable to 6 attention entries** (median **1714ms**, against **64ms** for the other 34); the old `autokernel_gate` shows no attention premium at all (19.5ms vs 20.5ms). Cause: stage 2 runs a real sweep of **8 shapes x 3 dtypes = 24 configs**, each invoking reference and candidate, with attention shapes reaching `(256, 64)` and attention cost being O(seq^2 * d). The old gate ran 3 draws at ONE fixed shape and one dtype, which is exactly the omission the audit corrected (`AUTOKERNEL_BASELINE_AUDIT.md`: the old gate "ran no real shape sweep, and omitted a whole stage"). **So the tail IS the correction working as designed** — the faithful gate costs more because it does the work the paper specifies. Distinct from §3 item 1, whose bugs were arity/dtype faults in `_adversarial_stability_inputs` producing false positives: different stage, different mechanism, different symptom. Answered entirely from data in hand; no GPU run needed. **#7b COMPLETE (2026-08-20) — the adversarial search is PROCESS-STARTUP bound, not LLM bound.** Answered from the existing `search_history.db` (microsecond `created_at` on proposals/executions/verdicts reconstructs a per-worker timeline); no instrumentation was needed for the finding and no GPU run was spent on it. Per worker over the 556.2s / 80-proposal CFA run: **subprocess spawn + torch/triton import 394.7s (71%)**, LLM call + parse 117.7s (21%), in-kernel work 1.9s (0.3%), residual coordination ~8%. Medians per proposal: execute phase **20.28s** of which in-kernel is **0.03s**; LLM + parse **6.03s**; per single execution the spawn-to-result interval is **10.25s**. **Cause:** `executor.py:219` uses `mp.get_context("spawn")` — a fresh interpreter per execution — and `executor.py:21` imports torch at module scope, so all 160 executions re-import torch/triton and re-init CUDA. `wall_time_ms` is recorded INSIDE the subprocess (`executor.py:145-152`) and times only the kernel call, so a subprocess structurally cannot measure its own startup — which is why this was invisible and landed in the 'orchestration' bucket. **Retries were NOT a factor:** the `_call_and_parse` loop (`worker.py:97`, MAX_RETRIES=2) would show as gaps at 2-3x median; the LLM+parse distribution is tight (p10 5.27s / p50 6.03s / p100 9.85s) with **0 of 76 gaps above 2x median**. Every proposal parsed first try. Latent cost, not a present one. **Supersedes the earlier framing:** '1.4% execution / 98.6% LLM + orchestration' was right that execution is negligible and wrong about what fills the rest — it is process startup, not the LLM. **Instrumentation shipped:** `KernelExecutionResult.total_wall_time_ms` (parent-stamped around `Process.start()/join()`), a nullable `executions.total_wall_time_ms` column with an idempotent `_migrate_unlocked()` ALTER TABLE, verified on a COPY of the real 160-row DB (rows and columns preserved, nothing removed, idempotent). NULL means never measured — never 0.0, which would claim a free spawn. Negative control in `check_execution_persistence.py` §6, confirmed to fire when the write is removed. **I did NOT instrument `litellm.completion`:** it would refine a bucket already bounded to 21% and 5.3-9.9s, where parse is a regex plus `json.loads`. Add it only if that 21% becomes the target. **Fix direction (NOT approved, not started):** process reuse — a persistent executor pool, `fork` where CUDA permits, or batching a proposal's two kernels into one subprocess (that alone halves spawn count). Full plan and numbers: `~/.claude/plans/shimmering-swimming-sun.md`. |
| **#8** operator/kernel coverage expansion | Not started. **Flagged for plan mode before implementation** — largest scope, three sub-parts, real architectural decisions. See §2.5, item 2, for a hard constraint on #8(b). |
| ~~**A1** persist `check_results` in the adversarial search~~ | **DONE.** New `executions` table in `search_history.db` (one row per proposal x kernel) carrying full `check_results` incl. `layer` and `details`, plus `ExecutionError`; `passed_checker`/`passed_naive`/`error_type`/`n_checks`/`n_failed` denormalised for querying. Written per-execution (not batched) so a crash mid-mutant-loop keeps what ran. Verified by `tests/instrumentation/check_execution_persistence.py` — 23 assertions incl. migration safety against a **copy** of the real 262-proposal DB and three negative controls confirmed to trip. Nothing pre-existing changed. |
| **A2** persist feedback hints | **CONSIDERED, NOT IMPLEMENTED — decision, not an omission.** A hint is a pure function of `check_results` via `_diagnose_reference_failure`, so now that A1 stores the check results per execution, the hint any proposal received is **recomputable** rather than lost. Persisting the text as well would only guard against `_diagnose_reference_failure` itself changing between runs. Implement only if that function's determinism across versions becomes a concern. |
| **B2** enforce agreement across the 7 operator tables | **Candidate from §2.3's sweep — scoped, NOT approved.** All 21 keys aligned today, nothing enforces it. Six tables fail loudly on a miss; `OPERATOR_CONTEXT` falls back to a bare label and silently degrades search quality — the causal_flash_attention failure mode. Cheap import-time assertion, same pattern as `validate_bug_pattern_hints()`. **Suggested second.** |
| **B1** key `BUG_PATTERN_HINTS` by `(operator, mutant_id)` | **Candidate from §2.3's sweep — scoped, NOT approved.** Mutant ids are not unique across operators; `wrong_mask` collides between `flash_attention` and `scaled_dot_product_attention` and needs opposite advice. Currently a stopgap merged entry. Needs a lookup-signature change plus a duplicate-key guard. **Suggested third.** |
| **P1/P2** residual template heuristic and silent parse defaults | **Candidates from §2.3's sweep — scoped, NOT approved.** `prompts/base.py:549` asserts a default diagnosis in prose; `worker.py:161-162` defaults `predicted_failure_mode` to `""` so a malformed LLM response parses as valid. Both low severity. **Suggested last.** |
| **#9** paper framing note | Not a code task. Position the argument as "kernel correctness in the context of kernel generation." Keep #2-#8 results organised to support it; act on it in the writing pass only. |

---

## 5. Verification assets — RESOLVED, now in the repo

The local suites that validate the shipped instrumentation were previously in
`/tmp` and would not have survived. **They now live in `tests/instrumentation/`
and the `/tmp` copies have been deleted** so no stale duplicate can be imported
by accident.

| File | Assertions | Covers |
|---|---|---|
| `tests/instrumentation/check_item2_instrumentation.py` | 44 | `_try` four-valued outcome mapping; subcheck passthrough; `_summarize` joined string byte-identical to pre-change; `summarize()` output byte-identical with/without `check_records`; `harness._call` accepting 3- and 4-tuple systems |
| `tests/instrumentation/check_ablation_report.py` | 13 | `analyze_check_ablation.py` against a hand-derived fixture, incl. error-not-counted-as-catch, identical-catch-set redundancy, never-ran roster entry, operator with no properties, deliberate crash-as-catch |
| `tests/instrumentation/check_autokernel_faithful_construction.py` | 12 families x (8 sweep + edge) shapes x 3 dtypes, **plus 3 negative controls run every time** | `autokernel_faithful` argument construction: arity per family, dtype leaks, sweep length, stage-3 coverage. **Numerics still unvalidated — needs a GPU.** |
| `tests/instrumentation/check_adversarial_search_fixes.py` | ~60 | §2.1 + §2.2: `OPERATOR_CONTEXT` covers all 21 wired operators with tensor keys and a stated rank; `BUG_PATTERN_HINTS` covers every real mutant id; the `_resolve_paths` startup assertion **actually aborts** on a deliberately-unhinted mutant; `_diagnose_reference_failure` replayed over the **122 real reference failures**; the `not_caught`/`caught_no_gap` split incl. a control reproducing the CFA case |
| `tests/instrumentation/check_execution_persistence.py` | 23 | **A1**: migration safety on a copy of the real DB (existing tables unchanged, rows AND columns); `check_results`/`ExecutionError` round-trip field-for-field incl. `layer`; pre-existing outputs unchanged; a negative control proving the CFA question is unanswerable before and answerable after; crash-partial persistence |
| `tests/instrumentation/check_kernel_executed_probe.py` **(needs real torch — see below)** | 76 recorded cases x 4 controls x 8 seeds | **§3.0**: the probe-ladder fix, replayed against the **real recorded tensor descriptors** from the search history. 25 recorded false positives must pass; 51 recorded passes must still pass; a genuine ghost must still be caught; each rung measured **in isolation** (leave-one-IN, not leave-one-out — see §5 instance 11); verdicts stable across 8 seeds. **Control 0 runs the OLD check and requires it to fail all 25**, without which "25/25 now pass" is unfalsifiable |
| `tests/instrumentation/check_shape_constraints.py` | 30 | **1b**: `SHAPE_CONSTRAINTS` replayed over **410** recorded reference executions (9 operators) — 0 inputs whose reference passed are now rejected; all **23** confirmed hits still proposable; three negative controls (over-tight table must lose 6 hits, empty table must catch 0 of 65, removing an operator must trip the coverage validator); plus the worker-survival regression for `ProposalRejected` |
| `tests/instrumentation/check_batch_executor.py` §11 | +9 | **2c**: which start method was ASKED for vs USED; preload is exactly `["torch"]`; `start_method` recorded on every result incl. the crash path; `execute_proposal` never requests forkserver; the unavailable-forkserver fallback is recorded as `spawn`, not as the forkserver that was requested |
| `tests/instrumentation/check_forkserver_executor.py` | 16 | **2c**: the inherited-RNG hazard. Drives the real `_run_batch_in_subprocess` with a stub generator read **at draw time**, from a "forkserver-inherited" sentinel state — so it can distinguish "seeded then drew" from "drew then seeded", which no source-text check can. Two mutation controls that must FIRE: seed deleted, and seed applied after materialization. Plus the `_MODULE_IMPORT_PID` guard that renames the startup keys if the stamps are ever taken in another process |
| `tests/instrumentation/README.md` | — | Why these are not pytest tests, and the two guarantees worth understanding before editing them |

### How to run them — this changed

They are **standalone scripts, not pytest tests**, and are named `check_*.py`
rather than `test_*.py` so that `pytest.ini`'s `python_files = test_*.py` does
not collect them. **Do not rename them.**

```bash
# all of them, fail-loud:
for f in tests/instrumentation/check_*.py; do python3 "$f" >/dev/null || echo "FAIL $f"; done
```

Plain `python3` — no venv, no numpy, no torch, no pytest. Exit 0 = pass.
**Nine of the ten pass on the dev machine and were re-run green on 2026-08-21**
(the suite grew by `check_batch_executor.py`, `check_shape_constraints.py` and
`check_forkserver_executor.py` since the "six of seven" count was written).

**`check_kernel_executed_probe.py` is the one exception and will report FAIL
locally — that is expected, not a regression.** It guards a *numerical* property
(whether two float32 outputs are bitwise equal is the entire question), so a
shape-recording stub has nothing to say about it. It needs a real `torch`, but
**not a GPU** — the Colab VM's CPU is enough. Run it in whatever session you
already have open:

```bash
PYTHONPATH=/content python3 /content/tests/instrumentation/check_kernel_executed_probe.py
```

**They must stay outside pytest.** They replace `sys.modules["torch"]`
(and numpy) with stubs process-wide. `tests/conftest.py` imports the real
`torch` at module scope and every `tests/verification/*` test depends on it, so
collecting these in the same process would corrupt the rest of the suite. The
stub approach is also the only practical one here (see §0 on Drive/venv
slowness), and it is not weaker for this purpose: the defects being guarded
against are arity/shape/dtype bugs, which a shape-recording stub catches more
directly than a real CPU run.

### A finding from the port itself, worth carrying forward

`check_autokernel_faithful_construction.py` **passed while testing the wrong
file** when first moved: its `sys.path` still pointed at a scratch copy under
`/tmp/akf`, so it validated a stale module rather than the repo's own. A green
run proved nothing. This was caught only by running a **negative control** —
deliberately breaking the thing the test guards and confirming it fails.

It is fixed, and **the three negative controls are now wired into the script's
normal execution path** — every run mutates a copy of the module's source three
ways (shortened sweep, layernorm arity bug, dtype leak; the last two being
exactly the bug classes item #1 found in the old baseline) and requires the
checks to fail on each. A control that does not trip fails the run, and so does
an anchor string that no longer matches the source, so a refactor cannot silently
disarm the self-check. It also prints `module under test: <path>` every run;
**if that path is ever not the repo's own
`benchmarks/autokernel/files/autokernel_faithful.py`, the run is worthless.**

A second finding from wiring those controls in: **one of the three negative
controls was itself passing for the wrong reason.** The dtype-leak mutation
originally injected a stub-only symbol (`T`, defined in the test, not in the
module) into the mutated source. The mutant raised `NameError`, the script
exited non-zero, and the control looked like it had worked — right exit code,
wrong reason. It never detected a dtype leak at all. This was caught only by
noticing the control reported no failure *sample* alongside its non-zero exit,
and it would otherwise have been locked in as "3/3 confirmed". It is now a
genuine leak (the builder hardcodes `float32`, ignoring the requested dtype),
which is why it reports 484 detections rather than an empty message.

That one is worth dwelling on: it is a *verification artifact* that passed for
the wrong reason — one level deeper than the others, and the reason a bare
"exit code was non-zero" is not sufficient evidence that a negative control
works. Check *what* it detected, not just *that* it failed.

Generalise the lesson: this project has now found the same shape of problem
**thirteen times** — work that appeared verified but was not:

1. Item #1's crash-scored-as-catch (an exception counted as a caught mutant,
   which produced the entire reported 18% false-positive rate).
2. Item #2's dropped attribution (per-check detail computed every run, then
   discarded before it reached disk).
3. Item #6's discarded verdict detail (`missed_mutants` collapsing "not caught"
   and "caught, no gap" into one bucket, making the run undiagnosable).
4. A ported test importing a stale copy under `/tmp` and passing while
   validating the wrong file.
5. A negative control passing via `NameError` instead of detecting the defect
   it was written to detect.
6. A post-change equality assertion that **never ran**. After splitting
   `BUG_PATTERN_HINTS` into two dicts, the verification step re-parsed the file
   with `ast.literal_eval` to confirm no entry had changed — but the new
   definition is `{**GENERIC_SEED_HINTS, **MUTANT_HINTS}`, which
   `literal_eval` cannot evaluate. It raised inside the checker instead of
   comparing anything, while the file itself was correct and every test passed.
   The check reported nothing and verified nothing. Same species as instance 5:
   **a verification tool that fails or short-circuits without checking what it
   claims to check.** Re-verified properly by importing the module and diffing
   against a pre-change snapshot.
7. **A fixture that only modelled shapes the reader could already handle.**
   `check_ablation_report.py`'s 13 assertions passed against a `subchecks`
   field modelled as list-or-`None`. Real corpus data contained an **int**
   there (`tile_coverage`'s column count landing in a slot the adapter reserves
   for compound-check sub-records), and `analyze_check_ablation.py` died with
   `TypeError: 'int' object is not iterable` on the first real run. **2 of 1343
   check records were malformed and attribution for all 94 checks was lost.**
   The suite was green the entire time, because the fixture author and the
   reader author shared the same wrong assumption about the field's type.
   Distinct from instances 4-6: nothing failed or short-circuited here — the
   test ran correctly and completely, against inputs that could not exhibit the
   bug. **A fixture derived from the same mental model as the code under test
   verifies the model, not the code.** Fixed in three places with a permanent
   int-slot negative control; full write-up in
   `benchmarks/CHECK_ABLATION_FINDINGS.md` §3.3.

8. **A backup plan that would have produced a valid-looking, near-empty file.**
   §0's reclamation risk called for pulling the adversarial-search DB off the VM
   mid-run. The proposed mitigation — `colab download` the `.db` each poll cycle
   — was checked before being relied on, and was wrong.
   `store.py:151` sets `PRAGMA journal_mode=WAL`, so committed rows live in
   `search_history.db-wal` until a checkpoint. Mid-run the main `.db` was
   **4096 bytes (one page, schema only) while the `-wal` held 976KB**. The
   download would have succeeded, produced a real SQLite file, opened without
   error, and contained essentially none of the run.
   Distinct from every instance above, including 7: nothing was under test here
   and no assertion was involved. **The failure mode was a recovery procedure
   whose output passes every cheap sanity check — file exists, non-zero size,
   valid format, no error — while being useless.** A backup you never restore
   from is indistinguishable from a good one until the moment you need it.
   Fixed by snapshotting with `sqlite3.backup()` on the VM (WAL-inclusive,
   consistent against a live writer) and downloading that: 143KB containing the
   rows the 4096-byte file did not. Recorded in §0.
   **General rule: an artifact's existence is not evidence that a write
   completed.** Any source using WAL, a journal, an OS buffer, or any other
   deferred-commit mechanism can leave a file that is present and well-formed
   yet empty of the data you wanted. Verify size and content against something
   you independently know should be there — for a DB, row counts — never
   presence alone.

9. **A test whose central assertion passed vacuously, caught only by its own
   negative control.** Writing the RNG-neutrality guard for `_warm` (#7a), the
   sequence under test was collected with `system(...) or rng.counter`. The
   system returns a truthy tuple, so `or` short-circuited and recorded **the
   tuple**, never the counter. Every sequence was therefore trivially equal and
   **"warmup leaves the draw sequence byte-identical" reported PASS while
   comparing nothing**. The paired negative control — "without the restore the
   sequence must move" — failed, because it too was comparing identical
   tuples, and that contradiction is the only reason the bug surfaced.
   Same family as instances 5 and 8: **a check that reports success without
   exercising what it claims to.** The distinguishing feature here is that the
   positive assertion alone was indistinguishable from a real pass — only
   pairing it with a control that MUST fail exposed it. Fixed by collecting the
   sequence in an explicit loop; the trap is documented in the test body so the
   next person does not reintroduce it.
   **Rule: every "these are identical" assertion needs a paired case that makes
   them differ.** Without it, the assertion cannot distinguish "identical" from
   "comparing nothing."

10. **A safety flag that read the right value on the class and the wrong value
   on every instance.** `KernelSpec.batch_samples` gates which operators may
   batch their perturbation samples; `frobenius_norm` must never batch, and was
   given `batch_samples: bool = False`. Reading the CLASS attribute returned
   `False` — correct. Every INSTANCE returned `True`. The spec files are
   dataclasses, but the individual spec subclasses are not themselves
   `@dataclass`-decorated, so the override was an inert annotated class
   attribute while the inherited `__init__` assigned the parent's `True` to the
   instance and shadowed it. Nothing errored; the one operator that must never
   batch was silently batched, which loosens its tolerance **639x**.
   Caught only because the negative control asserted the flag's value on a
   constructed spec rather than on the class. **A configuration flag is not
   verified until you read it off the object the code will actually use.**
   Fixed by making it a `@property`, which cannot be shadowed by an inherited
   `__init__` regardless of decoration, with a test asserting no spec
   reintroduces the field form.

11. **Two controls on the same fix, both broken, in opposite directions.**
   Writing the verification for the `check_kernel_executed` probe-ladder fix
   (§3.0, 2026-08-21). Neither error was arithmetic; both were about what the
   control was structurally capable of observing.

   **(a) Leave-one-out on an OR-composed check measures nothing.** The fix is a
   disjunction of five rungs — it passes as soon as *any* rung moves the
   output. The control disabled one rung at a time and compared totals. Every
   rung reported **delta 0**, which reads as "no rung matters". Measured
   properly, the rungs rescue **0, 10, 0 and 20** of the 20 recorded cases: one
   of them single-handedly does the entire job. The overlap between rungs
   absorbed every removal, so the control ran correctly, completely, and
   reported a number that was true and meaningless. It printed "delta +0" for
   the one rung the whole fix depends on. Fixed by measuring each rung as the
   **only** rung (leave-one-IN).

   **(b) No proof the harness could exhibit the bug being fixed.** The script
   asserted "25/25 recorded false positives now pass" without ever checking
   that the *old* code fails those same 25 on the same harness. A replay too
   lossy to reproduce the defect would have printed exactly the same line. This
   very nearly shipped: the first green run reported 25/25 cleared, and it was
   only (a)'s contradiction that forced a second look. Fixed by adding a
   control that runs the pre-fix probe verbatim and requires it to fail 25/25 —
   which it does, so the fixture is now known to discriminate.

   Running the old check also produced a finding nothing else would have: it
   mis-reports 2 of 51 recorded passes, and across 8 seeds one of them flips on
   4 — **the old check's verdict on near-degenerate inputs was partly a coin
   flip.** That is invisible unless something runs the old code on purpose.

   Distinct from instances 5, 6 and 9, where a check failed, short-circuited,
   or compared nothing. Here both controls executed fully and reported
   well-formed numbers; the defect was in what they were *able* to distinguish.
   Closest to instance 7 — a test that runs correctly against inputs that
   cannot exhibit the bug — but generalised from fixture data to control
   *structure*.

12. **A control validated three ways that still could not see the result that
   actually occurred.** Writing the verification for §7 item 1's re-run
   (2026-08-21). The run's built-in control is the `nan_inf`+`dtype_preserved`
   failure class, which §7 said must stay "unchanged" at 9.

   First finding: **the raw count is the wrong control.** Those failures do not
   track input magnitude — they sit at the LOW end of it (33..1e3, against 1e6
   elsewhere). They track **non-power-of-two shapes**: 7/9 (78%) of them carry
   one, versus 41% of all other proposals. How often the LLM proposes an odd
   shape is free to vary run to run, so comparing raw counts reads ordinary
   sampling variance as "the fix loosened something". The control was
   renormalised to *nan_inf failures per odd-shape proposal* (baseline 7/36 =
   19.4%).

   The renormalised control was then validated against three synthetic DBs —
   identical (0 pp, correctly consistent), all `kernel_executed` failures
   cleared (correctly consistent), and every `nan_inf` failure wiped (correctly
   diverges). Three for three. **It was still not fit for purpose.** A fourth
   synthetic — a partial 4-of-9 loosening — came back "consistent", because at
   n≈36 the standard error on a ~19% rate is ~10 pp, so the comparison can only
   resolve moves of **≳21 pp**. The real run then moved **+10.9 pp**, landing
   squarely in that blind spot.

   Had the fourth synthetic not been tried, the report would have read "control
   held" for a change the instrument was structurally incapable of seeing.

   **Distinct from instance 11**, which is its nearest neighbour. There, the
   controls were malformed — leave-one-out on a disjunction, and a missing
   old-code baseline — and the fix was to restructure them. Here the control was
   *correctly structured and empirically validated*, and the defect was purely
   one of **statistical power at the n the experiment actually had**. Nothing
   about its structure was wrong; there were simply not enough odd-shape
   proposals for it to resolve the effect.

   **The rule: validating a control against synthetic cases proves it can
   detect *those* cases. It says nothing about whether it has the power to
   detect what will actually occur.** Before trusting a control, compute its
   minimum detectable effect at the sample size you will really have, and
   report that limit alongside the verdict. A control that cannot resolve the
   observed move has **failed to resolve** — which is not the same as the thing
   it was watching for having held. Ruling something out and confirming
   something are different results and must be reported differently.

   **RECURRED 2026-08-21 in the forkserver A/B (2c), pointing the other way, and
   that direction is the more dangerous one.** The seeded-path baseline
   `B1 vs B2` came back **0 of 80** disagreements and was about to be written
   down as "the floor is zero, and forkserver's 2 of 80 exceeds it" — i.e. a
   clean zero read as an exact measurement rather than as one draw. Repeating
   both arms three times showed the underlying event is a **per-pass** timing
   race that fires roughly 1 in 3 passes, so a single pair of passes had almost
   no power to see it, and **the spawn arm flips too**. A zero from an
   underpowered control is not a tighter result than a small number — it is the
   same failure to resolve, wearing the most convincing possible disguise.
   Instance 12's rule applies unchanged; what is new is that **`0` is exactly
   the value least likely to prompt anyone to ask about power.**

13. **Two checks that report "the kernel failed" when the kernel never ran.**
   Found 2026-08-21 diagnosing the 12 residual reference failures left after
   §3.0 (§7 item 1). `check_nan_inf` and `check_dtype_preserved`
   (`verification/layer1_structural/runtime_guards.py:37, :85`) each wrap their
   kernel call in `try/except Exception` and return
   `(False, f"Kernel raised an exception: {e}")`. So both **conflate two
   different facts**: *the kernel ran and produced a bad number*, and *the
   kernel could not run on this input at all*.

   Consequence, measured: all 12 residual failures were **compile-time or
   calling-convention crashes on out-of-domain input** — 7x
   `arange's range must be a power of 2` (head dim 48 or 33, and
   `tl.arange(0, D)` requires a power of two), 5x
   `too many values to unpack (expected 2)` (rank-3/4 input against a wrapper
   whose docstring says *"Q, K, V: (N, D) — single sequence, no batch/head
   dimension"*). **Not one contained a NaN, an Inf, or a dtype change.** Scored
   against the *reference*, that turned "the harness fed a correct kernel an
   input outside its domain" into "the correct kernel failed the checker", and
   **`BENCHMARK_RESULTS.md` §8.3.1 published a 17.1% false-positive rate that
   did not exist.** The real rate on in-domain input is **0 of 58**. The
   separation is exact in both directions — 0 in-domain failures, 0
   out-of-domain passes.

   **Distinct from instance 1**, which is its parent pattern. Instance 1 was a
   crash scored as a *catch* (an exception counted as a caught mutant,
   producing the reported 18% FP rate for the autokernel baseline). This is a
   crash scored as a *check failure on the reference*, i.e. as a false
   positive — the same conflation pointing the opposite way, in the checker's
   own Layer 1 rather than in a baseline re-implementation. Same root cause,
   opposite sign, different blast radius.

   **Distinct from instances 5/6/9/11/12** in kind: nothing here failed to run,
   short-circuited, compared nothing, or lacked power. Both checks executed
   exactly as written. The defect is **semantic** — a two-valued return
   (`passed` / `failed`) used to encode a three-valued reality
   (`passed` / `failed` / `could not be evaluated`).

   **The codebase already had the right answer and did not apply it here.**
   §3.0's probe-ladder fix introduced rung E precisely to report
   *"not evaluable"* rather than a verdict when an input is degenerate. That
   concept exists, one function away, and was not reached for.

   **Rule: a check that can be handed an input it cannot evaluate needs a
   third outcome, not a `False`.** Before reading any pass/fail rate, ask what
   that function returns when the operation could not be attempted — if the
   answer is "the same thing as a genuine failure", the rate is a mixture of
   two populations and its headline number is not what it says it is.

**Recurrence, not a new instance: instance 1's defect class in
`_evaluate_verdict`.** The same crash-scored-as-catch shape, found in the same
2026-08-21 pass, in `verification/adversarial_search/coordinator.py`:

```python
caught = not mr.passed_checker      # a mutant that CRASHED reads as "caught"
```

All 12 out-of-domain proposals recorded `caught_no_gap=['wrong_causal_mask']`
even though the mutant crashed rather than being caught by any check — the
comment above that branch says *"The checker DID catch it"*, which is false for
these.

**Blast radius, confirmed bounded — do not over-escalate this.** `is_hit`
requires `reference_passed`, and an out-of-domain proposal fails that, so this
**cannot manufacture a false HIT** and no published catch rate is affected. The
damage is confined to the `caught_no_gap` / `not_caught` bookkeeping — which
matters only because §2.2 built those fields specifically so that "caught" and
"not caught" would be trustworthy for diagnosis, and this quietly puts
un-evaluated proposals in the "caught" bucket. It is a diagnostic-integrity
defect, not a correctness one.

**Prefer verifying a guard by breaking what it guards over trusting a green
run — and then confirm it failed for the right reason.** Instance 7 adds a
second rule: **hand-written fixtures should include at least one shape taken
from real output, not only shapes the author imagined.** Instance 8 adds a
third, for procedures rather than tests: **check a recovery path by restoring
from it and counting what came back, not by confirming the file arrived.**
Instance 11 adds two more, both about control structure rather than test data:

- **A control on a disjunction needs leave-one-IN, not leave-one-out.**
  Removing one rung from an OR of many isolates nothing — the others absorb it
  and every rung looks irrelevant. Enable one at a time and measure what it
  rescues alone. The same applies in reverse to an AND-composed check, where
  leave-one-out is the correct form and leave-one-in measures nothing.
- **Any "N/N now pass" claim needs a paired "the old code still fails N/N on
  these same inputs".** Without it the pass count is unfalsifiable: a harness
  that never reproduced the defect reports the identical number. This is the
  general form of instance 9's rule — an assertion needs a paired case that
  makes it come out the other way — applied to a fix rather than to an
  equality.

Instance 12 adds a sixth, and it is the one that survives a control being
*correct*: **compute a control's minimum detectable effect at the sample size
you will actually have, and report that limit next to the verdict.** Passing
synthetic validation proves detection capability for the synthetic cases, not
power against the real one. When the observed move lands inside the blind
spot, the honest report is "failed to resolve", not "held" — **ruling
something out and confirming it are different results.**

Instance 13 adds a seventh, about the *shape of a check's return value* rather
than about tests at all: **a check that can be handed an input it cannot
evaluate needs a third outcome, not a `False`.** Two-valued pass/fail cannot
express "could not be attempted", so every such case silently joins the failure
population and the resulting rate describes a mixture. Before quoting any
pass/fail rate, ask what the function returns when the operation could not be
attempted — and note that this project has now published a wrong headline
number from exactly this (`BENCHMARK_RESULTS.md` §8.3.1's retracted 17.1%).

### A second, distinct pattern: silent exact-match string lookups

Separate from the six above, and worth its own entry because the failure mode
is different: **a string comparison that silently does the wrong thing and
reports nothing.** Four confirmed instances, found across §2.1's fixes and the
follow-on hint-grouping work in the §2.5 block — **not** all within §2.1 proper,
and none surfaced by §2.2 or the A1 pass:

1. **`wrong_causal_mask`** — `BUG_PATTERN_HINTS.get(seed_bug_pattern, "")` is an
   exact-key lookup. The table had `"wrong_mask"` but not `"wrong_causal_mask"`,
   so the seed hint resolved to `""` and the worker was never told what bug it
   was hunting, for an entire 120-proposal run. Nothing logged a problem.
   (`adversarial_results/CFA_NONHIT_ROOTCAUSE.md` §3.)
2. **`startswith` vs equality** — the self-check written to prove
   `OPERATOR_CONTEXT` had no bare-label fallbacks tested
   `turn.startswith(f"Operator context:\nOperator: {op}")`. Several *real*
   contexts legitimately begin `Operator: argmax over the LAST dimension…`, so
   the check flagged correct entries as missing. The fallback is the context
   being **exactly** `f"Operator: {op}"` — equality, not prefix. A prefix test
   for an equality condition.
3. **`first_tile`** — a **live** softmax mutant id in `_MUTANT_MAP` with no
   `BUG_PATTERN_HINTS` entry. `"partial_tile"` existed and was evidently
   written to serve it, but the lookup is by exact mutant id so it never
   matched. Identical in kind to instance 1, in the same table, undetected
   until the new startup validator was pointed at the real mutant list.

4. **Duplicate `wrong_mask` key** — two operators have a mutant named
   `wrong_mask` (`flash_attention` and `scaled_dot_product_attention`) and they
   need *opposite* advice: one is an off-by-one causal mask that should exist,
   the other applies a mask where none should. The table is keyed by mutant id
   **alone**, not `(operator, mutant_id)`, so adding a second `"wrong_mask"`
   entry did not collide — Python silently kept the last one, replacing
   flash_attention's working hint with SDPA-specific advice. Found only by a
   duplicate-key audit (31 entry lines vs 30 dict keys), not by any test.
   Resolved by a single entry covering both readings; the real fix, if this
   table grows, is to key by `(operator, mutant_id)` — the lookup already has
   `operator` in hand.

Instance 3 is the useful one to remember: it was found **because** the negative
control for the new assertion was actually run. Had the assertion been added and
assumed to work, it would have hard-failed the softmax search at startup the
first time anyone used it — a fix introducing a new break.

**The general defect:** `dict.get(key, default)` and `startswith` both fail
silently by design. Where the key space is enumerable — mutant ids, operator
names — validate coverage at startup rather than discovering a miss from a
wasted run. `validate_bug_pattern_hints()`
(`verification/adversarial_search/prompts/base.py`) is the pattern to copy.

### A third pattern: measurement code that is structurally blind to the cost it attributes

**Both latency investigations this session reached the wrong answer first, and
neither time was the arithmetic wrong.** In both, the instrument could not
observe the dominant cost, so that cost silently landed in whatever bucket was
left over — and the leftover bucket was then given a confident name.

- **#7a step 2:** `systems` is the outer loop, so Triton JIT compilation was
  charged entirely to whichever system the dict happened to run first. The
  timing code was correct; it simply had no way to see that the eighth system
  was paying for the ninth through eleventh. **84% of the "checker latency"
  was compilation.**
- **#7b:** `wall_time_ms` is recorded *inside* a spawned subprocess, so it
  cannot measure that subprocess's own startup. Spawn plus `import torch` was
  **71% of each worker's wall time** and appeared nowhere in the persisted
  data, so it was absorbed into a bucket labelled "LLM + orchestration" and
  reported as if it were LLM latency.

Both were resolved from data already in hand — loop order in one case,
timestamp arithmetic across existing DB rows in the other — not by adding
instrumentation. **Reach for the existing data before spending a GPU session.**

**Standing question for any performance claim in this project: can the thing
doing the measuring actually see the cost it is attributing?** If a measured
region excludes setup, or runs after something else warmed a shared cache, or
sits inside a process that cannot observe its own creation, then the residual
bucket is not "overhead" — it is unmeasured, and naming it is a guess. Say
"unattributed" until something has actually looked.

---

## 6. CURRENT STATE — read this second, after §0

Written 2026-08-21, updated at the end of the `check_kernel_executed` session,
then again the same evening after §7 item 1's re-run landed.

**The headline for a new session: no *known* correctness defect is open — but
the checker's false-positive rate on valid adversarial input is 0 of 58 =
0.0%.** §3.0, the last known soundness defect, is fixed and verified on
hardware, and the residual failure class it left behind has since been
**diagnosed (2026-08-21) and was never a checker false positive at all**.

All 12 residual "failures" were **crashes on out-of-domain input** — the search
proposed rank-3/4 tensors and non-power-of-two head dims, neither of which
`causal_flash_attention` can accept — reported as check failures because
`check_nan_inf` and `check_dtype_preserved` return a plain `False` for **any**
exception. See **§5 instance 13**. The separation is exact: 0 in-domain
failures, 0 out-of-domain passes.

**Three known defects are now open, all diagnosed and none fixed**, plus a
bookkeeping question — all are *specified* rather than unknown, and none is a
soundness defect in the checker's numeric judgement on valid input:

1. the exception-conflation in `check_nan_inf` / `check_dtype_preserved`
   (**1a — still open**, sequenced after 1b because it needs a GPU regression
   and 1b changed the input population that regression measures against; it is
   now unblocked);
2. ~~`validate_proposal` enforcing no shape/rank contract~~ — **FIXED 2026-08-21
   (1b)**. `SHAPE_CONSTRAINTS` now rejects out-of-domain proposals before
   execution; 0 falsifying cases across 410 replayed executions, all 23
   confirmed hits still proposable;
3. whether §2.2's caught/not-caught bookkeeping needs a third state for
   "crashed" (**1c — deferred, likely moot now 1b ships**, since the crashing
   population is rejected pre-execution; blast radius confirmed bounded — it
   cannot manufacture a false hit);
4. **NEW 2026-08-21 — `check_kernel_executed`'s delegation detector
   false-positives the REFERENCE under concurrency (1d).** For a reference
   kernel the candidate *is* the reference, so its `torch.equal` guard is
   trivially true and the check reduces to timing one function against itself;
   with 4 workers on one T4 it reported 10.9x-15.3x "speedups" of the reference
   over itself, on a different proposal each run. Pre-existing, exposed while
   verifying 2c, fires under spawn as well as forkserver. **Not a numeric
   unsoundness and no published number depends on it** — the corpus benchmark is
   not 4-way concurrent — but it inflates the adversarial search's
   reference-failure rate, which is 1a's quantity.

**Do not carry forward either older phrasing** — neither "there is no open
correctness defect" nor "a 17.1% rate of unexplained failures". The first was
true when written; the second was a measurement artifact and is retracted in
`BENCHMARK_RESULTS.md` §8.3.1. See §7 for the options and their trade-offs.

### What the project is

A three-layer Triton/CUDA kernel correctness checker (structural → algebraic →
numeric), benchmarked against 5 SOTA baselines on TritonBench (29 operators, 40
mutants) and KernelBench, plus an LLM-driven adversarial input search.

### Everything that was GPU-blocked is done

| Item | State |
|---|---|
| **#1** faithful autokernel gate | **DONE.** 80% catch / 1% FP vs the old approximation's 68% / 18%. The `rtol=0` variant is near-neutral, contradicting the prediction that it would be load-bearing. |
| **#2** per-check ablation | **DONE.** `benchmarks/CHECK_ABLATION.md` from 1343 real check records. The run exposed a reader defect that made the table unbuildable at all; fixed in three places. |
| **§2.4** CFA re-run | **DONE.** 80 proposals, NO_HIT, `not_caught: []` and `caught_no_gap` populated on **all 80** — §7.6's second predicted branch, and the first time that outcome was *decidable*. Closes the withdrawn `hit_mutants: []` claim by measurement. |
| **§3.0** 0%-FP scope | **CLOSED 2026-08-21 — fixed in code, verified on a T4.** The 2026-08-20 documentation pass qualified the claim; the 2026-08-21 pass fixed the `check_kernel_executed` defect underneath it. 25/25 recorded false positives cleared against real Triton kernels, ghost still caught 25/25, **zero verdict changes** across 440 mutant + 2200 reference verdicts. Details in §6.1. |
| **#7** latency | **CLOSED, no open sub-threads.** Detail below. |

### 6.1 §3.0 — `check_kernel_executed`, fixed 2026-08-21

**What was wrong.** The check asserted *different input ⇒ different output*,
probing with `x + randn_like(x)*0.1 + 1.0`. That is false for any non-injective
operator, so **correct** reference kernels failed a Layer-1 check — 20 of 80
proposals in the 2026-08-20 CFA run, plus 30 across earlier history.

**The diagnosis in §3.0 was wrong about the mechanism, and its recommended fix
does not work.** §3.0 blamed the constant `+1.0` shift and recommended a
multiplicative/per-element probe. Measured against the 20 recorded CFA cases —
first in pure Python, then against the real Triton kernel:

| probe | rescues |
|---|---:|
| old `x + randn*0.1 + 1.0` | 0 — reproduces all 20 |
| per-element multiplicative (**§3.0's recommendation**) | **0** |
| fresh independent draw | 0 |
| negation `-x` | 10 |
| **companion `V`, multiplicative** | **20** |
| companion `K`, multiplicative | 15 |

The real mechanism: **K constant or saturated across key positions makes the
attention weights independent of Q for every Q.** No perturbation of the primary
can move the output. Perturbing a *companion* is the load-bearing fix.

**What shipped.** A probe **ladder** evaluated as a disjunction — pass as soon
as any rung moves the output, which can only reduce false positives, never
create them: (A) per-element multiplicative + additive, (B) negation, (C) fresh
independent draw, (D) each float companion perturbed with the primary held
fixed, (E) a reference-sensitivity guard — if nothing moved the candidate, run
the same ladder through the reference, and if the reference is *also* still,
report "not evaluable" rather than a ghost. Rung E is the only part correct by
construction rather than empirically, and is beyond what §3.0 specified.

Cost on the pass path is unchanged — rung A moves a correct kernel on ordinary
input, so it is one extra kernel call, as before. The delegation detector
(bit-identical-to-reference + 10x faster) is untouched and still reached on
every non-ghost path.

Files: `verification/layer1_structural/runtime_guards.py`, both call sites
(`verification/checker.py`, `benchmarks/autokernel/files/checker_adapter.py`),
new `tests/instrumentation/check_kernel_executed_probe.py` + its fixture.

**Measured on a Colab T4, 2026-08-21.**

- **Real Triton kernels, the 25 recorded false positives** (CFA 20, argmax 3,
  softmax 1, flash_attention 1): old check fails **25/25**, new check passes
  **25/25**. Cleared by negation 12, companion 10, multiplicative 3.
- **False-negative control on the same 25 real inputs:** a hardcoded-output
  ghost is still caught **25/25**. No FP-for-FN trade.
- **Corpus regression, full `run_benchmark.py`:** **zero verdict changes**
  across 440 mutant + 2200 reference verdicts over all 11 systems.
  `kernel_executed` remains **40 ran / 0 caught / 0 errors / 0 skips / 0 FPs**.
- **Seed stability:** all 76 recorded cases identical across 8 seeds. The **old**
  check was not stable — one recorded pass flips on 4 of 8 seeds, so its verdict
  on near-degenerate inputs was partly a coin flip. Found only by running the
  old code on purpose (§5 instance 11).

Artifacts: `verification_runs/kernel_executed_fix_2026-08-21/` (results_raw.json,
results.json, results.md, CHECK_ABLATION.md, run.log). **The repo's canonical
`benchmarks/autokernel/files/results*.json` were deliberately NOT overwritten** —
verdicts are identical, so the only change would be latency churn, and latency
on this corpus is inside the ±7% noise floor.

**DONE 2026-08-21 (evening):** the fresh adversarial search ran. Outcome was
**36.2% → 17.1%**, not the ~11% predicted — `kernel_executed` went to **0 of
70** as expected, but the `nan_inf` class rose and now accounts for **all** of
the remainder. The run's built-in control **failed to resolve** rather than
confirming (see §5 instance 12). Full numbers and caveats: §7's closed-item
block; artifacts in `adversarial_results/cfa_rerun_postfix_2026-08-21/`.

### #7, in full

- **Metric fix:** the published latency was mean-based and outlier-dominated
  (mean/p50 up to 33.5x). `summarize()` now emits p50/p90/p99/max.
- **Root cause:** **84% of the "checker latency" was Triton JIT compilation**,
  charged to whichever system the dict happened to run first.
- **Two measurement defects fixed:** cache warming (`harness._warm`) and one
  timer convention across all seven systems. `allclose`'s "354x faster" was
  wrong by ~8x — like-for-like it is **29x**. "numeric only at 64% of the
  latency" survives as **75% of p50**.
- **Search latency (#7b):** the adversarial search is **process-startup bound**,
  not LLM bound — spawn + `import torch` is **71%** of worker time, the LLM
  21%, kernels 0.3%. Answered from existing DB timestamps; no instrumentation
  needed for the finding.
- **Deprioritised:** #7a step 3 (optimising the checker). Its premise dissolved
  — the honest steady state is ~21ms p50 per check.

### Two changes whose results should be read carefully

**Layer reorder** (structural → algebraic → numeric): **verdict-preserving and
mechanically confirmed** — of the 14 mutants caught by algebraic-but-not-
structural, numeric checks ran on 14/14 before and **2/14 after**. The latency
effect is **inconclusive**: −2.9% measured against a ±7% run-to-run noise floor.
Do not cite it as a speedup.

**Perturbation batching:** **Stage A (removing the per-sample `.item()` sync) is
a real −21%** on those checks, clearing control drift. **Stage B (batching the
kernel calls) measured exactly 0.0% median** — the syncs were the serializer,
not the launches. Recommend defaulting `batch_samples` off; it carries a 639x
tolerance-loosening failure mode for no measured gain.

### The running lessons (§5)

**Thirteen instances** of "work that appeared verified but was not", plus **three
distinct patterns**: verification artifacts that pass for the wrong reason,
silent exact-match string lookups, and **measurement code structurally blind to
the cost it attributes**.

**The standing question for any performance claim in this project: can the
thing doing the measuring actually see the cost it is attributing?** Both
latency investigations got the wrong answer first, and neither time was the
arithmetic wrong: a dict-order confound hid compile cost, and a subprocess
cannot observe its own startup. In both cases the leftover bucket was given a
confident name. Say "unattributed" until something has actually looked.

**Instance 11 adds the same question for controls: can this control distinguish
the outcome it is asserting from its opposite?** Two ways it could not, both hit
in one session — leave-one-out on an OR-composed check (every rung reports
"delta 0" while one of them does the entire job), and an "N/N now pass" claim
with no paired "the old code still fails N/N on these same inputs". Both
controls ran fully and printed well-formed numbers.

---

## 7. OPEN WORK — read this before picking anything up

**Updated 2026-08-21 (evening), after §7 item 1's re-run landed. Item 1 is
CLOSED; a new open unknown was created by closing it. Read the ranking below
before assuming this section still says what it used to.**

**The known-defect backlog from §3.0 is empty** — that defect is fixed and
verified on hardware. But the phrase this section used to carry, "there is no
known correctness defect open", is **no longer accurate**, and should not be
quoted from memory. Item 1's re-run eliminated `check_kernel_executed` as a
false-positive source and in doing so left the `nan_inf`+`dtype_preserved`
class as **100% of the remaining checker-attributable reference-failure rate,
with no explanation on file**. That is not a known defect and not a known
non-defect — it is an unexamined 17.1%. See item 1 in the ranking.

So: picking up the next item is still largely a question of value rather than
soundness debt, but there is now exactly one live unknown, and it should be
chosen or declined **deliberately**, not skipped because it sits next to an
optimisation item.

The one exception worth naming explicitly, because it is *not* a defect and
should not be mistaken for one: `verification/checker.py:229`'s `bool(None)`
skip-coercion. §3.0 named it as the second Layer-1/Layer-2 soundness item. It is
**deliberately unfixed and deliberately guarded** — `check_item2_instrumentation.py`
asserts the current (wrong) coercion, so that fixing it cannot land silently
alongside an instrumentation change. It affects how a *skip* is reported, not
whether a bug is caught. Fix it on purpose, with its own before/after, or leave
it.

### ✅ CLOSED 2026-08-21 — the adversarial-search re-run (was item 1)

Ran on a Colab T4, §2.4's configuration exactly (`causal_flash_attention`,
`--strategy beam --workers 4 --max-iter 20`), same model. Artifacts:
`adversarial_results/cfa_rerun_postfix_2026-08-21/`.
`BENCHMARK_RESULTS.md` updated in one pass: §1 scope note, new **§8.3.1**, §11
summary line.

| | 2026-08-20 | 2026-08-21 |
|---|---|---|
| reference executions | 80 | 74 |
| infrastructure timeouts (excluded) | 0 | 4 |
| evaluable | 80 | 70 |
| **checker-attributable failures** | **29 (36.2%)** | **12 (17.1%)** |
| — `kernel_executed` | 20 | **0** |
| — `nan_inf` + `dtype_preserved` | 9 | 12 |

**Established: `kernel_executed` no longer false-positives** — 0 of 70, down
from 20 of 80, with no 8-check reference records in the new run at all. That is
the one claim this run supports directly.

**NOT established, and do not let this drift:**

- The rate is **17.1%, not the ~11% the fix alone predicts.** The shortfall is
  entirely the `nan_inf` class rising (9 → 12 raw; normalised by odd-shape
  exposure, **19.4% → 30.3%, +10.9 pp**).
- **The control FAILED TO RESOLVE — it did not confirm.** At n≈33-36 the
  comparison resolves only moves ≳21 pp (z was 1.05). "The control held" is
  **not** what happened and must not be written down as if it were. See §5
  instance 12, which is this exact finding.
- The search is LLM-driven, so this corroborates a *rate*; the deterministic
  proof of the fix remains the banked 25/25 real-Triton replay.

Two anomalies, both explained, neither checker-related: the 4 timeouts are
`TimeoutError` at the 30s default (baseline had 0), and 74-not-80 is worker w0
dying at iteration 13 after 3 failed attempts to parse malformed LLM JSON.
Verdict structure otherwise identical — NO_HIT, 74/74 `not_caught=[]`,
`caught_no_gap=['wrong_causal_mask']`.

**Ruled out:** the `nan_inf` rise is *not* previously-masked failures surfacing.
In the 2026-08-20 run all 20 `kernel_executed` failures recorded 8 checks each
**with `nan_inf` among them, evaluated and passing** — disjoint proposal sets.

### ✅ SHIPPED 2026-08-21 — 1b, shape-constraint enforcement

`validate_proposal` now enforces a per-operator, per-tensor, per-dimension shape
contract, so out-of-domain proposals are rejected **before** they spawn a
subprocess. Motivation: 12 of 74 proposals (16% of a run) crashed BOTH kernels
and yielded no comparison at all.

**`SHAPE_CONSTRAINTS` in `schemas.py`, all 21 operators**, each entry citing the
reference-kernel source line that justifies it. **Only 3 of 21 have a
power-of-two rule** — the attention family, on `D` (last dim) only. Every
row-wise op computes `BLOCK_SIZE = triton.next_power_of_2(n_cols)` internally
and masks, so a non-power-of-two column count is the *intended* adversarial case.

**THE RULE FOR EDITING THAT TABLE, repeated here because it is the whole
safety argument:** constraints are **derived from source**; historical data can
only **falsify** one, never confirm it. `layernorm` had ONE historical passing
proposal, `matmul` two — "always powers of two" in that data says nothing about
the kernel. An absent constraint costs a wasted iteration; an invented one
silently suppresses the edge cases the search exists to find.

**Measured, offline, no GPU** (`tests/instrumentation/check_shape_constraints.py`,
30 checks): replaying 410 recorded reference executions across 9 operators,
**247 legitimate inputs allowed, 65 invalid caught, 0 falsifying** — zero inputs
whose reference actually passed would now be rejected. All **23 confirmed hits**
remain proposable.

**Controls fire** (a green run proves nothing without them, §5): a deliberately
over-tight table (global last-dim pow-2) produces 34 falsifying cases and
**loses 6 real hits** — softmax `[512,777]`×3 and `[512,333]`, gelu `[33,33]`
and `[128,160]`; an empty table catches 0 of the 65; removing an operator trips
the coverage validator.

**A prerequisite fix shipped with it, and it matters more than it looks.**
`coordinator._worker_loop` used to `return` when a worker exhausted its retries,
forfeiting every remaining iteration — measured: worker w0 died at iteration 13
of the 2026-08-21 run, which is why that run produced 74 proposals, not 80.
Since constraints *raise* the rejection rate, leaving that in place would have
made 1b a net throughput **loss**. Now a new `ProposalRejected` (recoverable —
the model produced bad output) is distinguished from an LLM outage (still fatal),
and the worker resets to a fresh proposal rather than dying. A related latent
bug was fixed alongside: `_call_and_parse` appended the user turn on entry but
the assistant turn only on success, so a failed call left a dangling user turn
and the next call sent two user messages in a row.

**Retry prompt is now failure-class aware** (`prompts.format_rejection_turn`).
The old hardcoded string said *"Respond with ONLY the JSON proposal schema. No
markdown"* — right for a parse failure, actively misleading for a shape
rejection where the JSON was already fine, and it was the only thing telling the
model what to change.

**`OPERATOR_CONTEXT` corrected for all three attention operators.** It claimed
`N >= 16`, which the code does not require (`N` is masked and never enters a
`tl.dot` shape), and **omitted** the real constraint that `D` must be a power of
two — the text that invited `D=48` in the first place. `flash_attention` had no
Constraints section at all.

**Not yet measured: the ~16% throughput recovery.** That needs a real search run
and is deliberately deferred to ride with item 2's before/after, since 1b changes
the population that measurement runs against.

**Deliberately NOT done:** §2.3's **B2** seven-table cross-check. Only a narrow
two-table key assertion (`validate_shape_constraint_coverage`) was added, which
makes B2 easier later rather than pre-empting it. B2 remains open and unscoped.

**New finding, logged not fixed — `mat_mult.py:48-49` has no shape assert:**

```python
M, K = A.shape
K, N = B.shape      # the first K is silently discarded
```

There is no `assert A.shape[1] == B.shape[0]`. Mismatched inner dimensions do
not raise — they read out of bounds or truncate the reduction, so reference and
mutant both return garbage and any comparison between them is meaningless. 1b
stops the *search* generating such inputs; it does **not** add the missing
assert to the reference kernel. That is its own decision.

---

### ✅ SHIPPED 2026-08-21 — §2.5, item 2, subprocess-spawn reduction

**One subprocess per PROPOSAL instead of one per KERNEL.** `execute_proposal`
spawned a fresh interpreter for every (proposal, kernel) pair and imported torch
at module scope, so all 160 executions of the 2026-08-20 CFA run re-paid
interpreter startup, `import torch`/triton and CUDA init — 71% of each worker's
wall time. `execute_proposal_batch` runs the reference and every mutant in one
subprocess: spawns per proposal go `N+1` → `1`, which is 2→1 for the 16
single-mutant operators and 5→1 for matmul and flash_attention.

**Full numbers, caveats and raw records:
`verification_runs/batch_executor_2026-08-21/FINDINGS.md`.** Headline: per-proposal
median **28.11s → 13.19s (0.47x)** for causal_flash_attention and **70.16s →
14.09s (0.20x)** for flash_attention, spawns per proposal 2.00→1.00 and
5.00→1.00, **zero fallbacks fired** across 140 batched kernel records.

**Measured WITHOUT the LLM, on purpose.** Two LLM-driven searches contain
different numbers of proposals, so their wall times measure different amounts of
work — the same class of error as timing a subprocess from inside itself. The
harness (`replay_ab.py`) replays a **fixed set of recorded proposals** through
both arms, so the work is held constant and only the executor varies. Both arms
live behind one switch (`SearchCoordinator(batch_executions=...)`, CLI
`--no-batch`) so the comparison is one binary, not two checkouts.

**The order control is the reason to believe it.** Passes run **A1 → B → A2**;
A1 and A2 agreed to **1.6%** and **0.3%**, so warm page/JIT caches are not what
produced the difference. Without that second unbatched pass the whole result
would have been confounded by run order — exactly the defect that made the
checker's per-layer latency table wrong (#7a step 2).

**Why the win exceeds the spawn ratio:** the reference module is loaded once per
batch, so the Triton compilations the checker drives through it (perturbation's
20 samples, the cross-shape sweeps) are paid once and reused. Mutant kernels drop
from ~5-7s to **24-257ms**. That is why flash_attention gained 80% where spawn
count alone predicts 60%.

**THE NUMBER THAT DECIDES WHAT COMES NEXT — `import torch` is 85% of startup.**
The new per-phase instrumentation decomposes the ~6.2s: `torch_import_ms` **5241ms
(85%)**, `cuda_init_ms` 645ms (10%), interpreter + mp bootstrap 255ms (4%), spec
imports 41ms, materialization 3ms. **This reverses the ranking of the two
deferred options.** A `forkserver` with `set_forkserver_preload(["torch"])`
removes the 85% while keeping one process per execution — same isolation, same
timeout semantics, a few lines of diff — where a persistent pool removes 95% but
takes on process/CUDA-context lifetime, crash-respawn and cross-execution state
in a subsystem that has surfaced unplanned bugs every time it was touched. **Do
forkserver first. It is NEW work and needs its own go-ahead; do not start it as a
continuation of item 2.** One thing to check when doing it: a forked child
inherits the forkserver's RNG state, so the per-execution seeding below has to
stay in place or every child would draw identical tensors.

> **DONE 2026-08-21 — see item 2c in the ranked list and
> `verification_runs/forkserver_2026-08-21/FINDINGS.md`.** `torch_import_ms`
> went **1527ms → 0.0ms** and per-proposal medians **0.565x / 0.642x** against a
> B1→C→B2 order control. **The RNG warning above was correct and is now
> measured, not merely anticipated:** six unseeded forks returned bit-identical
> tensors. The batched path was already immune (it re-seeds from `proposal_id`
> before drawing); the single-kernel path was NOT, and is deliberately left on
> spawn for that reason. Verifying this produced one unplanned finding, item
> **1d** — which is now the fourth consecutive time this subsystem has been
> touched and returned something unplanned.

**A DECLARED SEMANTIC CHANGE RODE ALONG, AND IT IS NOT A LATENCY ARTIFACT.**
Batching materialises the inputs once, seeded from `proposal_id`. That fixed a
pre-existing reproducibility defect in the old path and introduced a smaller
semantic change than the defect it replaced. **This is a finding about the OLD
code, not a footnote to the new code — it is written up separately below, under
"THE UNBATCHED SEARCH WAS NEVER REPRODUCIBLE".** Do not re-derive it.

**Two hazards batching creates, both designed for rather than patched later**
(`tests/instrumentation/check_batch_executor.py`, 48 assertions, **6 mutations
confirmed to trip it**):

1. **A poisoned CUDA context.** An out-of-bounds mutant leaves CUDA sticky-errored
   and every later call in that process raises — which would have recorded "the
   next mutant crashed" for kernels that never ran. The child syncs after each
   kernel, aborts the batch on failure, and the parent re-runs the remainder
   through the **unchanged single-kernel path**. Reference runs first so it is
   never the casualty.
2. **A queue deadlock.** The old code did `join()` *then* `queue.get()`, which
   only worked because one payload fit the pipe buffer. The parent now drains
   concurrently against a **per-kernel** deadline (not one batch-wide budget, or a
   hung reference would eat the mutants' time). The control for this is a
   *functional* reproduction with a bounded queue — the first version compared
   source-text positions and a `p.join(timeout=0.01)` walked straight past it.

**`benchmarks/run_random_baseline.py` was deliberately NOT batched.** This is a
DEFERRED DECISION, not unfinished scope — it is item **2b** in the ranked open
work below, with the reasoning there. Do not treat it as a leftover to be tidied
up alongside something else.

**Schema:** four nullable columns on `executions` (`exec_mode`, `batch_spawn_ms`,
`kernel_wall_time_ms`, `startup_phases_json`) via the existing idempotent
`_migrate_unlocked` pattern, now driven by a `_LATE_EXECUTION_COLUMNS` list —
appending to that list is the only supported way to extend the table. Read
`exec_mode` **first**: `total_wall_time_ms` and `batch_spawn_ms` are populated on
mutually exclusive paths. Batched rows leave `total_wall_time_ms` NULL on
purpose, because a per-kernel spawn interval does not exist for a shared
subprocess and dividing the batch's would be a fabricated number.

**Analysis tool:** `python3 scripts/analyze_spawn_cost.py --baseline A.db --after B.db`.
It reports **spawns per proposal**, not total spawns — a run that stopped early
has fewer of both, and the raw totals would report a "reduction" that is only a
shorter run.

---

### 🔍 FINDING 2026-08-21 — THE UNBATCHED SEARCH WAS NEVER REPRODUCIBLE

**This is a defect in the pre-batching executor, found while planning item 2. It
is recorded on its own because it is not a consequence of batching — batching
merely exposed it and then, as a side effect, fixed it.**

**Nothing in the executor path ever seeded the RNG.** `_materialize_one` uses a
bare `torch.randn` (`materializer.py:36`) and `grep manual_seed` over
`verification/` returned nothing. Each spawned subprocess therefore started from
its own OS-entropy seed. Two consequences, neither previously written down:

1. **Reference and mutants were compared across DIFFERENT random draws.** Inside
   one execution the candidate and reference share `inputs`, so that comparison
   was always sound. But `_evaluate_verdict` (`coordinator.py`) compares
   `reference_result.passed_checker`, computed in the reference *subprocess*,
   against mutant outcomes computed in *other* subprocesses — on different
   tensors. On a marginal proposal those can disagree for no reason but the draw.
2. **No search run was reproducible.** Re-running the same recorded proposals
   through the same code gave different verdicts. Nobody had measured this,
   because nothing had ever replayed a fixed proposal set through the old path.

**Quantified, and the number is the point.** Replaying identical proposals
through two *unbatched* passes (A1, A2 — both unseeded, the shipped behaviour at
the time) disagreed on **5 of 80** (proposal, kernel) pairs for
causal_flash_attention and **2 of 60** for flash_attention. Replaying the same
proposals through unbatched vs **batched + shared + seeded** disagreed on **2 of
80** and **2 of 60**.

> **The semantic change batching introduced is SMALLER than the gap it replaced.**
> The old path moved more verdicts against *itself*, run to run, than the new path
> moves against the old one.

Checker-pass rates are identical across arms (CFA 34/80 and 34/80; FA 8/60 and
8/60), every disagreement falls on the `reference` kernel, and they flip
symmetrically in both directions — the signature of marginal inputs, not a
systematic shift.

**Why this matters beyond item 2.** Any future claim of the form "this change
altered N verdicts" must be read against a **≈6% run-to-run floor on the
unseeded path** (5 of 80). Before this finding there was no such floor and a
change moving a handful of verdicts would have looked like a real effect. The
batched path is now seeded and reproducible, so that floor applies to the
unbatched arm and to anything else still drawing unseeded — including
`benchmarks/run_random_baseline.py` (see the deferred item in the ranked list).

**Do not "clean up" the seeding.** The single-kernel path is deliberately left
UNSEEDED so the two A/B arms genuinely differ and the change stays measurable
rather than confounding; `check_batch_executor.py` asserts this and fails if
`manual_seed` appears in `_run_in_subprocess`.

---

### Open work, ranked — nothing below is started

**Updated 2026-08-21 evening: 1b has SHIPPED** (see the block above), so the
ranking below has changed. What remains at the top is **1a**, which is now
unblocked — it was sequenced behind 1b precisely because 1b changes the input
population 1a's GPU regression must be measured against. Item 2 was queued
behind 1b for the same reason and is likewise now unblocked, with a cleaner
baseline than it would have had before.

| | Item | Kind | Why it is / is not urgent |
|---|---|---|---|
| 1a | **Crash-attribution fix** — `check_nan_inf` / `check_dtype_preserved` | correctness (reporting) | **DIAGNOSED, STILL OPEN — sequenced after 1b ON PURPOSE, do not lose it.** Both return a plain `False` for any exception, conflating "ran and produced a bad number" with "could not run at all". This published a wrong headline (§8.3.1's retracted 17.1%). §5 instance 13. **Why it waits:** it needs a GPU corpus regression to prove zero verdict changes, and 1b (shipped 2026-08-21) changes the input population that regression would be measured against — running it first would have measured the old population. **It is now unblocked.** Not a numeric-soundness defect: the checker's verdicts on valid input were correct throughout. Expect a collision with `check_item2_instrumentation.py`, which deliberately asserts the current `bool(None)` skip-coercion so a change cannot land silently. |
| 1c | **Third state for "crashed" in caught/not-caught bookkeeping** | diagnostic integrity | **DEFERRED — likely moot now that 1b ships.** `caught = not mr.passed_checker` in `_evaluate_verdict` scores a crashed mutant as caught (instance 1's defect class in a new location; blast radius **confirmed bounded** — cannot manufacture a false hit, `is_hit` requires `reference_passed`). 1b rejects out-of-domain proposals before execution, which removes the population that produced these crashes. **Revisit only if crashes still slip through from some other cause** — check for `caught_no_gap` entries whose execution carries an `error_type`. |
| 1d | **`check_kernel_executed`'s delegation detector false-positives the REFERENCE under concurrency** | correctness (Layer 1) | **✅ FIXED AND VERIFIED ON A T4, 2026-08-21.** Replaced the sequential two-block timing with **interleaved best-of-N** (5 rounds x 2 calls, `min` across rounds; total launches unchanged at 10/side; **threshold left at 10x on purpose**, so any verdict change is attributable to the estimator and not a moved goalpost). Contention only ever ADDS time, so the min is the sample a stall cannot inflate, and interleaving puts any stall on both arms of the comparison. **Measured, same 4-worker contention, reference timed against itself:** ratio p50 0.91->**1.00**, p90 4.55->**1.06**, p99 10.24->**1.15**, max **51.24->1.22**, flip rate **1.230% (34/2765) -> 0.000% (0/140)**. **Verdicts untouched:** out-of-domain exclusions still exactly 5/pass, 0 reference failures from any other cause, all mutant verdicts still `caught`. **FULL CORPUS REGRESSION, T4, 2026-08-21: `run_benchmark.py` over all 11 systems — 440 mutant verdicts compared, **0 differing**; 2200 reference verdicts compared, **5 differing**, and all 5 are `frobenius_norm` inside `autokernel_gate` variants, flipping in BOTH directions (3 lost / 2 gained) — the known-inherent atomic-add bitwise flake documented in §3, in baselines that do not call `check_kernel_executed` at all. **All four `your_checker` systems: 0 differences on both mutants and references.** Artifacts: `verification_runs/delegation_fix_2026-08-21/`.** Offline control in `check_forkserver_executor.py` §5 injects a single stall in each of 10 slots: **old estimator fires 10/10, new fires 0/10, and a uniformly-100x-faster ghost is still caught** — noise rejection, not a disabled check. Original diagnosis kept below. **NEW 2026-08-21, DIAGNOSED — found while verifying 2c, and it is a pre-existing defect that forkserver merely exposed.** `runtime_guards.py:404-437`: when the candidate's output is `torch.equal` to the reference's, the check times 10 candidate calls against 10 reference calls and fails if `t_cand < t_ref * 0.1`. **For the reference kernel, the candidate IS the reference**, so `torch.equal` is trivially true and the check reduces to timing one function against itself. Under 4 workers on one T4 that ratio is a lottery: measured **11.3x, 15.3x, 10.9x and 12.9x** "speedups" of the reference over itself, on a **different proposal every time** — 6 distinct proposals across 8 observations. **Fires under spawn too** (1 of 3 passes) — this is not a forkserver defect. Distinct from §3.0, which fixed the *probe*; §6.1 recorded this detector as deliberately "untouched". **Blast radius:** no published number — `run_benchmark.py` is not 4-way concurrent and `results.md` still shows `kernel_executed` 40 ran / 0 caught / 0 FPs. It inflates the **adversarial search's reference-failure rate**, i.e. exactly the quantity 1a is about, so the two should probably be fixed together. Records: `verification_runs/forkserver_2026-08-21/diag_contention.json`.<br><br>**NOW QUANTIFIED (2026-08-21), and the fix follows from the numbers.** The ratio is now instrumented on every reference execution — `runtime_guards.py` emits `[delegation_ratio=…]` on the PASS path too, where it was previously computed and discarded (§2.3 Shape A, in the one check whose verdict *is* a timing comparison). Over **560 reference-vs-itself executions** under 4-way contention: p50 **0.92**, p90 4.83, p95 7.12, p99 **11.45**, max **14.39**. **The 10x threshold sits at about p98.4 — 1.6% of reference executions trip it by chance.** Nothing in 560 trials exceeded **20**. **A genuinely delegating kernel CALLS the reference, so its ratio is ≈1.0 and this check would never flag it anyway** — the condition only fires on noise, or on a precomputed-output ghost that would be orders of magnitude faster. **WITHDRAWN — "raise the threshold to ~50x" was wrong, and the correction matters.** That was written from 560 executions, where the max ratio was 23.26. At **2765** executions the max is **51.24**, and the tail keeps extending with sample size: pooled across both arms, ≥10x fires 34 times (1.23%), ≥20x twice, ≥30-50x once, ≥60x never. **No constant derived from a finite sample is provably safe here**, so the fix is **timing robustness, not a bigger number** — best-of-N, or interleaving the two timing loops instead of running them back to back, so one scheduling stall cannot decide a verdict. A **51x** apparent speedup of the reference over *itself* is what the current construction admits. Needs its own before/after; do not fold it into another change. **1d now also gates 2c** — see that row. |
| 2 | **Subprocess-spawn reduction** | performance | **✅ SHIPPED AND MEASURED 2026-08-21 — see §2.5 below.** Batching landed: one subprocess per *proposal* instead of per *kernel*, spawns `N+1`→`1`. Measured on a T4 against an A1/B/A2 order control: **0.47x per-proposal for causal_flash_attention (2 kernels), 0.20x for flash_attention (5 kernels)**, order drift 1.6% and 0.3%. Numbers, caveats and raw records: `verification_runs/batch_executor_2026-08-21/FINDINGS.md`. **Its follow-on, `forkserver`, is now ALSO shipped and measured — see item 2c.** |
| 2c | **`forkserver` start method** | performance | **✅ SHIPPED AND MEASURED 2026-08-21. DEFAULT IS OFF; turning it on is a one-line decision that has NOT been taken — see below.** Batched children can now be forked from a torch-preloaded server (`--forkserver`), removing the 85% of startup that is `import torch`: measured **1527ms → 0.0ms**, total startup **1825ms → 252ms**. Order-controlled B1→C→B2 replay on a T4: **0.565x per-proposal for causal_flash_attention, 0.642x for flash_attention**, order drift 0.5% and 0.2%. CUDA init stays per-child on purpose (initialising it in the forkserver would hand every fork an unusable context), which is the 10% a persistent pool would have to take risk to remove. Full numbers, four pre-flight probes and all caveats: `verification_runs/forkserver_2026-08-21/FINDINGS.md`. **The single-kernel path stays on spawn deliberately** — it seeds nothing, and probe 3 measured that six unseeded forks return *bit-identical* tensors, so under fork it would silently draw the same input for every proposal. **The remaining decision:** flipping the default is `use_forkserver: bool = False` → `True` in `SearchCoordinator.__init__`. The latency evidence supports it and no forkserver-specific verdict movement was found; the one reason to hesitate was that forkserver hit item 1d's timing race in 3 of 3 passes against spawn's 1 of 3, which **n=3 cannot resolve either way**. **A POWERED RE-TEST IS RUNNING (2026-08-21)** — 26 interleaved pairs, ~910 trials/arm, sized in advance to resolve the pilot-sized effect at 80% power; harness `verification_runs/forkserver_2026-08-21/race_rate.py`, analysis `analyze_race.py`, raw `race_rate.jsonl`. **RESOLVED AS FAR AS IT IS WORTH RESOLVING — 2765 executions across 3 sessions. `use_forkserver` STAYS OFF, and what unblocks it is fixing 1d, not more trials.** Final: **spawn 12/1400 = 0.86%, forkserver 22/1365 = 1.61%, ratio 1.88x, z=+1.65, p=0.10**; Mann-Whitney on the ratio distribution p=0.29. **The observed 1.88x is BELOW the experiment's own 2.53x minimum detectable effect, so this FAILED TO RESOLVE — it is not a null and must not be written up as one** (§5 instance 12, third occurrence). What it did establish: the direction **replicates 3 of 3 independent sessions** (1.25x, 2.40x, 1.79x, three different VMs — not a machine artifact); the arms are **identical through the bulk** (0.94-0.98x at thresholds 2-4, p50 0.92 vs 0.91) and diverge only in the 5-10 band; effects ≥2.53x are ruled out; and **the two most extreme outliers in the whole dataset (51.24 and 23.26) are both SPAWN**. Best estimate ≈0.6 extra spurious reference failures per 80-proposal search. Certifying p<0.05 needs ~2750/arm (~4 more GPU-hours); **judged not worth it because the decision is identical either way** — reopen only if certification is wanted for its own sake. **Sequencing: fix 1d → re-run this comparison → enable forkserver.** **1d IS NOW FIXED (see its row). `use_forkserver` STILL FALSE — deliberately, and not from inertia.** A 15-minute directional check post-fix (2 pairs, 70 reached trials/arm) shows the race is GONE in BOTH arms: **spawn 0 flips/70 (p90 1.04, max 1.15), forkserver 0 flips/70 (p90 1.07, max 1.22)** — against a threshold of 10, i.e. roughly 8x of headroom where there was 1.6% of crossings before. With **zero events in both arms** a rate ratio is not computable, so this is DIRECTIONAL ONLY and certifies nothing; forkserver remains a hair wider (p90 1.07 vs 1.04), consistent with the pre-fix direction and now immaterial in magnitude. **What flipping the default needs: one order-controlled re-run of `race_rate.py` post-fix at the powered n** (~900/arm, ~2 GPU-hours) — which is now cheap to interpret because the phenomenon it was measuring no longer reaches the threshold. Do not flip it on the 15-minute sample. Full writeup in that directory's `FINDINGS.md`; `race_analysis.txt` is the raw output. **Do not read the survival-curve sweep's minimum p as a result — the thresholds are nested and Bonferroni multiplies any single p by ~12.** |
| 2b | **`run_random_baseline.py` — batch it, or decide not to** | **deferred decision, NOT leftover scope** | **NOT approved, NOT started, and deliberately excluded from item 2 — do not fold it into another change.** `benchmarks/run_random_baseline.py:515/524` still calls `execute_proposal` once per kernel, so it pays the full spawn cost item 2 removed. The change is one line and the win is the same 0.47x-0.20x. **Why it was held back:** that script feeds the published same-budget random-vs-guided comparison, and batching it would ALSO switch it to shared, `proposal_id`-seeded inputs. That is a change to a published baseline's *input semantics* arriving as a side effect of a *latency* change — the exact substitution this project has been burned by. **It cuts both ways, which is why it is a decision and not a task:** leaving it unbatched means the guided search and its baseline now draw inputs differently, so the comparison is no longer semantically like-for-like either. **What resolving it needs:** decide whether the published comparison is defined on proposal budget alone (in which case batch it and report the verdict movement against the ≈6% unseeded floor from the reproducibility finding) or on execution semantics too (in which case leave it, and caveat §Table 3 that the two arms now differ). Do not decide it by whichever is less work. |
| 3 | **`batch_samples` default → off** | polish | One line. Stage B measured exactly 0.0% median gain and carries a 639x tolerance-loosening failure mode. Can ride along with anything. |
| 4 | **B2 / B1 / P1 / P2** | polish | Parked behind the MVP boundary for reasons that still hold. §2.3 has them scoped and ranked. |

**How to choose between 1 and 2.** If the goal is *publishing* or defending the
FP claim, item 1 outranks everything: `BENCHMARK_RESULTS.md` now states a 17.1%
adversarial reference-failure rate whose entire remaining content is
undiagnosed, and "we do not know why" is a weak position to hold in a paper. If
the goal is *engineering throughput*, item 2 is much the larger win and is
well-scoped. Item 1 is likely hours and needs no GPU; item 2 is a real
implementation with a GPU-measured before/after.

### ✅ DIAGNOSED 2026-08-21 — the residual `nan_inf` class was never a false positive

**Do not re-investigate this as an unknown. It is answered.** What remains are
three *specified* follow-ups (1a/1b/1c above), none of them a soundness defect.

**The finding.** All 12 residual reference "failures" were **crashes on
out-of-domain input**. Not one contained a NaN, an Inf, or a dtype change.
`causal_flash_attention` has two hard preconditions, both at source in
`TritonBench/reference/causal_flash_attention.py`:

- **rank exactly 2** — the wrapper does `N, D = Q.shape`; docstring:
  *"Q, K, V: (N, D) — single sequence, no batch/head dimension."*
- **D a power of two** — `D` is a `tl.constexpr` feeding `tl.arange(0, D)`.

| class | n | exception | violation |
|---|---:|---|---|
| A | 7 | `arange's range must be a power of 2` | D = 48 or 33; fails at **compile** time |
| B | 5 | `too many values to unpack (expected 2)` | rank-3 / rank-4 input |

`check_nan_inf` and `check_dtype_preserved` return `False` for any exception, so
these crashes were scored as the *reference kernel* failing the checker.

**The evidence is a perfect 2x2, which is why this is not a judgement call:**

| 2026-08-21 | checker FAIL | checker pass |
|---|---:|---:|
| in-domain | **0** | 58 |
| out-of-domain | **12** | 0 |

**Checker false-positive rate on valid input: 0 of 58 = 0.0%** (was 20 of 71 =
28.2% pre-§3.0-fix). The same rule applied to the 2026-08-20 run explains all 9
of its `nan_inf` failures and leaves exactly the 20 `kernel_executed` false
positives as its in-domain residual.

**Three hypotheses were named in advance; all three were wrong.** Not a Layer-1
unsoundness on odd shapes (the checks never got a number to judge); not
masked-partial-tile behaviour (class A fails before the kernel runs); and
"search artifact" was close but imprecise — the rate was inflated by the
*misattribution*, not by the generation.

**The superseded reading, recorded so it is not repeated.** These were first
characterised as tracking **non-power-of-two shapes** (78% vs 41%). That
correlation is real but explains **class A only** — two of the five class-B
cases (`[2,32,64]`, `[1,64,64]`) are *entirely* powers of two and fail on
**rank**. A fix aimed at odd shapes would have addressed 7 of 12 and none of
the rank cases. **This is the second time in this project that the first
plausible mechanism explained most-but-not-all of a failure set** — §3.0's
original diagnosis rescued 0 of 20. Check coverage of the whole set, not the
majority of it.

**The prompt is not the problem.** `OPERATOR_CONTEXT` already states both
constraints, emphatically for rank (*"Do NOT propose (B, H, N, D)... This was by
far the most common failure on this operator historically"*) and more weakly for
D (*"D should be a power of two"*). The model violates them anyway on 16% of
proposals. This is **not** #6's missing-context failure mode, and adding more
prompt text is unlikely to be the fix.

`BENCHMARK_RESULTS.md` §8.3.1 has been corrected and carries the full write-up.

### Which to pick up first — OPEN DECISION, deliberately not made here

**As of 2026-08-21 evening this is unresolved and awaiting the user's call.**
The options were laid out with scope and cost so the choice could be made on
information rather than on listing order. Do not treat 1a as outranking 2
merely because it is printed first.

The shape of the trade-off:

- **1a (crash attribution)** is the only one that has already caused a
  published error — §8.3.1's retracted 17.1%. It is small, but it is a
  correctness-of-reporting issue in Layer 1, and the same conflation exists in
  every check that wraps its call in `try/except`.
- **1b (domain enforcement)** is the only one that buys measurable throughput:
  16% of search proposals currently produce nothing. It composes with item 2
  rather than competing — 1b removes wasted iterations, 2 makes the remaining
  ones cheaper.
- **1c (third bookkeeping state)** is the smallest and least urgent; its blast
  radius is confirmed bounded and no published number depends on it.
- **2b (`run_random_baseline.py`)** is a **decision, not a task**, and it is the
  one item here that gets worse by being ignored: every session that leaves it
  alone widens the gap between the guided search and the baseline it is published
  against. Resolve it deliberately, in either direction, and write down which.
- **2 (subprocess-spawn reduction)** is **DONE** (§2.5). What is left of it is
  a **new, unapproved** follow-on: `forkserver`. See §2.5's last paragraph — the
  startup breakdown it produced is what makes that the right next move rather
  than the persistent pool.

**Note the interaction, since it is easy to miss:** 1b and 2 both attack search
wall time from different ends, so measuring 2's benefit *after* 1b lands gives a
cleaner before/after than measuring it against a run where 16% of proposals are
dead weight. In the end 2 was measured a third way that removes the LLM entirely
— replaying a fixed set of recorded proposals through both arms — because two
LLM-driven searches contain different amounts of work and their wall times are
not a like-for-like quantity.

**What is still queued for `BENCHMARK_RESULTS.md`, unchanged by this session:**

- **§3 item 1** (the autokernel_gate FP-mechanism sentence) — unblocked since
  2026-08-20; **not yet done**.
- **§3 item 3** (the 68% / 18% headline → 80% / 1%) — unblocked, same run;
  **not yet done**. Deliberately left untouched on 2026-08-21 so the re-run edit
  stayed scoped.
- **§3 item 2** (the "22 checks" figure) — **still needs a units decision, not a
  run.** Untouched. §3 shows the by-type reading does not reconcile either.
- **The 0%-FP scope caveats** — updated 2026-08-21 to the measured 17.1% and
  **deliberately retained**, because 17.1% is not 0%.

**Do not "helpfully" fix any of these without checking first** — §3's standing
instruction still applies, and it is the reason the file is still consistent.
