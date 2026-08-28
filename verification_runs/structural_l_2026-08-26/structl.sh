#!/bin/bash
# The ablation this pass could NOT run -- no CUDA and no Triton on the machine
# it was written on. Left runnable so the missing half needs no re-derivation
# when a GPU is next available.
#
# Same shape as ../n_samples_curve_2026-08-25/nsamp.sh and
# ../check_timing_2026-08-25/redun.sh: identical corpus, identical probe,
# KCC_ABLATION_SEED=1 on EVERY arm so the arms differ only by the estimator,
# KCC_CHECK_TIMING=1 so per-check shares are attributable.
#
# KCC_ABLATION_SEED is not optional here. The Monte-Carlo arm draws 20
# randn_like per call and the structural arm draws none, so the two arms
# consume RNG differently by construction. Without the per-check reseed every
# check downstream of a perturbation call would see a shifted stream and
# verdicts could move for reasons unrelated to the tolerance.
LOG=/content/structl; mkdir -p $LOG
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

# Baseline: the shipped Monte-Carlo estimator, flag absent.
run_arm MC_baseline           KCC_CHECK_TIMING=1

# The structural estimator at the derivation's NSIM.
run_arm STRUCT_nsim3000       KCC_CHECK_TIMING=1 KCC_STRUCTURAL_L=1

# NSIM sweep. FINDINGS.md Limits flags that nsim=3000 was never justified;
# if catch/FP hold at 300 the cost falls 10x and the ceiling becomes reachable.
run_arm STRUCT_nsim1000       KCC_CHECK_TIMING=1 KCC_STRUCTURAL_L=1 KCC_STRUCTURAL_NSIM=1000
run_arm STRUCT_nsim300        KCC_CHECK_TIMING=1 KCC_STRUCTURAL_L=1 KCC_STRUCTURAL_NSIM=300

# The lever that already beats this one, for a same-session comparison rather
# than a cross-session one. n=5 was measured at -19.4% checker wall.
run_arm MC_n5                 KCC_CHECK_TIMING=1 KCC_N_SAMPLES=5

cp /content/probe/*.json $LOG/ 2>/dev/null
touch $LOG/DONE
