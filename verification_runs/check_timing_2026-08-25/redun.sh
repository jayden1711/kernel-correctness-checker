#!/bin/bash
LOG=/content/probe; mkdir -p $LOG
cd /content
tar xzf /content/kcc4.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
cd /root && tar xzf /content/triton_cache.tgz
export PYTHONPATH=/content
cd /content

run_arm () {   # $1=name  $2..=env assignments
  local name=$1; shift
  echo "=== $name ($*) $(date +%s.%N)" >> $LOG/t.txt
  env KCC_ARM="$name" KCC_CHECK_TIMING=1 KCC_ABLATION_SEED=1 "$@" \
      python3 /content/probe_redundancy.py > $LOG/$name.log 2>&1
  echo "=== $name RC=$? END $(date +%s.%N)" >> $LOG/t.txt
  tail -1 $LOG/$name.log >> $LOG/t.txt
}

# A is the reference arm. Every arm carries KCC_ABLATION_SEED=1 so they differ
# ONLY by the removed check -- see checker.py's note on why.
run_arm A_baseline
run_arm B_no_wm_large   KCC_DISABLE_VARIANTS=large_uniform,large_random
run_arm C_no_adv_large  KCC_DISABLE_CHECKS=adversarial_large_magnitude,adversarial_large_magnitude_logits,adversarial_large_magnitude_qk
run_arm D_no_both       KCC_DISABLE_VARIANTS=large_uniform,large_random KCC_DISABLE_CHECKS=adversarial_large_magnitude,adversarial_large_magnitude_logits,adversarial_large_magnitude_qk
run_arm E_no_wm_at_all  KCC_DISABLE_CHECKS=weight_magnitude
touch $LOG/DONE
