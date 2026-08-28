#!/bin/bash
# Regression for the OOB harness fix (specs/layernorm.py, specs/rmsnorm.py
# width-adaptive non_power_of_two + reference-wrapper shape asserts).
# Same corpus, probe, and RNG discipline as every prior arm round
# (KCC_ABLATION_SEED=1); scored against the BANKED pre-fix gram_screen arms.
#
# The fix is draw-then-slice, so outside the two fixed classes every record
# must be BIT-identical to the banked pre-fix G arm -- the analysis asserts
# that, not just verdict identity.
LOG=/content/oobfix; mkdir -p $LOG /content/probe
cd /content
tar xzf /content/kcc7.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
export PYTHONPATH=/content
cd /content

run_arm () {
  local name=$1; shift
  echo "=== $name ($*) START $(date +%s.%N)" >> $LOG/t.txt
  env KCC_ARM="$name" KCC_ABLATION_SEED=1 "$@" \
      python3 /content/probe_redundancy.py > $LOG/$name.log 2>&1
  echo "=== $name RC=$? END $(date +%s.%N)" >> $LOG/t.txt
  tail -1 $LOG/$name.log >> $LOG/t.txt
}

# wrapper-assert probe first: must raise loudly on a short companion and
# accept a matched one (cannot run on the dev machine -- wrappers import triton)
python3 /content/wrapper_assert_probe.py > $LOG/wrapper_assert.log 2>&1

run_arm A_fix    KCC_CHECK_TIMING=1
run_arm G_fix    KCC_CHECK_TIMING=1 KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1

cp /content/probe/*.json $LOG/ 2>/dev/null
touch $LOG/DONE
