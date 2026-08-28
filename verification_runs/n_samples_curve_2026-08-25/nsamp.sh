#!/bin/bash
LOG=/content/nsamp; mkdir -p $LOG
cd /content
tar xzf /content/kcc5.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
cd /root && tar xzf /content/triton_cache.tgz
export PYTHONPATH=/content
mkdir -p /content/probe
cd /content

run_arm () {   # $1=name  $2..=env
  local name=$1; shift
  echo "=== $name ($*) START $(date +%s.%N)" >> $LOG/t.txt
  env KCC_ARM="$name" KCC_ABLATION_SEED=1 "$@" \
      python3 /content/probe_redundancy.py > $LOG/$name.log 2>&1
  echo "=== $name RC=$? END $(date +%s.%N)" >> $LOG/t.txt
  tail -1 $LOG/$name.log >> $LOG/t.txt
}

# PASS 1 -- the curve. One run at n=40 recording the full sensitivity vector.
# Verdicts at every n<=40 are then derived exactly offline: max_err does not
# depend on n, and the deltas are drawn one at a time from a per-check seed,
# so the n-sample vector is a strict PREFIX of the 40-sample one.
run_arm CURVE_n40 KCC_N_SAMPLES=40 KCC_RECORD_SENSITIVITIES=1

# PASS 2 -- validation arms. Actually RUN at these n and confirm the derived
# curve predicted them. Timing on, so per-check shares are measurable.
for n in 3 5 10 15 20 40; do
  run_arm VALID_n$n KCC_N_SAMPLES=$n KCC_CHECK_TIMING=1
done
cp /content/probe/*.json $LOG/ 2>/dev/null
touch $LOG/DONE
