#!/bin/bash
# DIRECT-tolerance A/B on the T4.
#
# Arms (KCC_ABLATION_SEED=1 on all, matching every prior corpus round):
#   A  probe baseline (shipped default)
#   D  KCC_STRUCTURAL_L=1 KCC_STRUCTURAL_MODE=direct
#
# Three measurements:
#   1. VERDICT/ATTRIBUTION pass per arm under KCC_CHECK_TIMING=1 (records
#      per-check outcomes; timing serialised -- shares only).
#   2. WALL-CLOCK: 5 interleaved reps per arm WITHOUT the timing flag
#      (the honest latency convention; per-trial dt_ms summed offline).
#   3. NEAR-MISS response: m-series + v-series probes per arm (the
#      boundary-sensitivity check no corpus run can provide).
LOG=/content/directab; mkdir -p $LOG /content/probe
cd /content
tar xzf /content/kcc11.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
export PYTHONPATH=/content

run_arm () {   # $1=outname  $2..=env
  local name=$1; shift
  echo "=== $name ($*) START $(date +%s.%N)" >> $LOG/t.txt
  env KCC_ARM="$name" KCC_ABLATION_SEED=1 "$@" \
      python3 /content/probe_redundancy.py > $LOG/$name.log 2>&1
  echo "=== $name RC=$? END $(date +%s.%N)" >> $LOG/t.txt
  tail -1 $LOG/$name.log >> $LOG/t.txt
}

# 1. verdict/attribution passes (serialised timing, records banked)
run_arm A_ver KCC_CHECK_TIMING=1
run_arm D_ver KCC_CHECK_TIMING=1 KCC_STRUCTURAL_L=1 KCC_STRUCTURAL_MODE=direct

# 2. wall-clock reps, interleaved
for i in 1 2 3 4 5; do
  run_arm A_w$i
  run_arm D_w$i KCC_STRUCTURAL_L=1 KCC_STRUCTURAL_MODE=direct
done

# 3. near-miss response, both arms
run_nm () {   # $1=tag $2=script $3=srcjson  $4..=env
  local tag=$1 script=$2 src=$3; shift 3
  echo "=== nm $tag START $(date +%s.%N)" >> $LOG/t.txt
  env KCC_ABLATION_SEED=1 "$@" python3 /content/$script > $LOG/nm_$tag.log 2>&1
  echo "=== nm $tag RC=$? END $(date +%s.%N)" >> $LOG/t.txt
  cp $src $LOG/nm_$tag.json 2>/dev/null
}
run_nm mA near_miss_gpu.py /content/nm/near_miss_gpu.json
run_nm mD near_miss_gpu.py /content/nm/near_miss_gpu.json \
       KCC_STRUCTURAL_L=1 KCC_STRUCTURAL_MODE=direct
run_nm vA v_series_gpu.py /content/nmv/v_series_gpu.json
run_nm vD v_series_gpu.py /content/nmv/v_series_gpu.json \
       KCC_STRUCTURAL_L=1 KCC_STRUCTURAL_MODE=direct

cp /content/probe/*.json $LOG/ 2>/dev/null
touch $LOG/DONE
