#!/bin/bash
# Regression for the layernorm padded-lane variance fix
# (TritonBench/reference/layernorm.py: diff = tl.where(mask, row-mean, 0.0)).
# Same corpus, probe, and RNG discipline as the oob_fix round
# (KCC_ABLATION_SEED=1); scored against the BANKED post-oob-fix arms
# (verification_runs/oob_fix_2026-08-28/arms/{A_fix,G_fix}.json.gz).
#
# Expected diff, per layernorm_mask_bug_2026-08-27/FINDINGS.md §4: EXACTLY
# ONE attribution change -- layernorm/wrong_variance_estimate loses
# [L3]cross_shape (its (1000,333) sub-outcome flips fail->pass), keeps
# [L3]adversarial_wrong_variance_trigger. Catch stays 40/40, FP 0/200.
LOG=/content/lnfix; mkdir -p $LOG /content/probe
cd /content
tar xzf /content/kcc8.tgz -C /content >> $LOG/setup.log 2>&1
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

# criteria (b)+(c) first: bitwise identity at pow2, fix liveness, and the
# (1000,333) margins (cannot run on the dev machine -- imports triton)
python3 /content/ln_gpu_probe.py > $LOG/ln_gpu_probe.log 2>&1
echo "probe RC=$?" >> $LOG/t.txt

run_arm A_lnfix  KCC_CHECK_TIMING=1
run_arm G_lnfix  KCC_CHECK_TIMING=1 KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1

cp /content/probe/*.json $LOG/ 2>/dev/null
touch $LOG/DONE
