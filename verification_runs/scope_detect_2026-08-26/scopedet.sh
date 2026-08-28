#!/bin/bash
# The GPU validation this pass could NOT run. No CUDA and no Triton on the
# machine the detector was written on; the T4 that carried every prior
# measurement is stopped. Left runnable so the step needs no re-derivation.
#
# Same shape as ../n_samples_curve_2026-08-25/nsamp.sh: identical corpus,
# identical probe, KCC_ABLATION_SEED=1 on every arm.
#
# THE SEED IS LOAD-BEARING HERE FOR A NEW REASON. The detector spends
# KCC_SCOPE_DEFECT_SAMPLES extra reference launches but draws NO new randn --
# it reuses the deltas the sensitivity loop already drew (see perturbation.py).
# So arm A and arm B should consume RNG identically and arm B's verdicts must
# match arm A's EXACTLY. Any difference is a bug in the wiring, not a finding.
LOG=/content/scopedet; mkdir -p $LOG
cd /content
tar xzf /content/kcc5.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
cd /root && tar xzf /content/triton_cache.tgz
export PYTHONPATH=/content
cd /content

run_arm () {   # $1=name  $2..=env
  local name=$1; shift
  echo "=== $name ($*) START $(date +%s.%N)" >> $LOG/t.txt
  env KCC_ARM="$name" KCC_ABLATION_SEED=1 "$@" \
      python3 /content/probe_redundancy.py > $LOG/$name.log 2>&1
  echo "=== $name RC=$? END $(date +%s.%N)" >> $LOG/t.txt
  tail -1 $LOG/$name.log >> $LOG/t.txt
}

# A -- baseline, detector absent. The verdict reference.
run_arm A_no_detector    KCC_CHECK_TIMING=1

# B -- detector on, recording EVERY invocation including the in-scope ones.
#      RECORD_ALL is what makes the margins measurable: without it the run can
#      say which invocations fired but not how much headroom the silent 26 had.
run_arm B_detector       KCC_CHECK_TIMING=1 KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1

# C -- defect probe at the banked sample count. The shipped default is 3
#      deltas; GPU_NATIVE.md's medians are over 40. If C and B disagree on any
#      classification, 3 is too few and the default must move.
run_arm C_defect_n20     KCC_CHECK_TIMING=1 KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1 KCC_SCOPE_DEFECT_SAMPLES=20

cp /content/probe/*.json $LOG/ 2>/dev/null
touch $LOG/DONE
