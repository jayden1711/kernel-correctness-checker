#!/bin/bash
# GPU validation of the GRAM SCREEN -- the rebuilt saturation screen
# (scope_detect.py 2026-08-27, replacing the defect ladder falsified in
# ../scope_detect_2026-08-26/).
#
# Same shape as ../scope_detect_2026-08-26/scopedet.sh: identical corpus,
# identical probe (probe_redundancy.py, byte-copied from
# ../n_samples_curve_2026-08-25/), KCC_ABLATION_SEED=1 on every arm.
#
# TWO ARMS ONLY. The Gram statistic is a prefix over the sensitivity loop's
# own deltas (measure_gram uses deltas[:k], drawn one at a time in fixed
# order), and arm G banks all 20 per-delta log-ratios raw -- so every smaller
# probe size is recomputable offline, exactly, and no convergence arms are
# needed. The screen draws NO new RNG and launches NO extra kernels (the JVPs
# are float64 CPU autodiff of math_refs.py), so A and G must be verdict-
# identical; any difference is a wiring bug, not a finding.
LOG=/content/gramdet; mkdir -p $LOG /content/probe
cd /content
tar xzf /content/kcc6.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
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

# G -- Gram screen on, recording EVERY invocation including in-scope ones.
#      RECORD_ALL is what makes the margins measurable.
run_arm G_gram           KCC_CHECK_TIMING=1 KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1

cp /content/probe/*.json $LOG/ 2>/dev/null
touch $LOG/DONE
