#!/bin/bash
# Measurement driver. NO code changes to the repo -- this only invokes
# run_benchmark.py repeatedly and records wall time + GPU/CPU samples.
#
# Four runs, designed so the run1->run2 delta is ATTRIBUTABLE:
#   run1  process cold, triton disk cache EMPTY (verified), page cache cold
#   run2  process cold, triton disk cache WARM,  page cache warm
#   run3  process cold, triton disk cache WARM,  page cache warm   <- replicate of run2
#   run4  process cold, triton disk cache CLEARED, page cache warm <- isolates the cache
# run2 vs run3 = stability. run1 vs run4 = is run1's cost the cache or general coldness.
LOG=/content/meas
mkdir -p $LOG/run1 $LOG/run2 $LOG/run3 $LOG/run4

cd /content
tar xzf /content/kcc.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
echo "pip_done $(date +%s.%N)" >> $LOG/timings.txt

# Cache state immediately before the first run (post-extract, post-pip).
python3 - <<'PY' > $LOG/cache_pre.txt 2>&1
import os, glob
p = os.path.expanduser("~/.triton/cache")
print("exists:", os.path.isdir(p))
print("entries:", len(glob.glob(p + "/*")) if os.path.isdir(p) else 0)
print("TRITON_CACHE_DIR:", os.environ.get("TRITON_CACHE_DIR", "<unset>"))
PY

# Samplers.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
           --format=csv,noheader -l 1 > $LOG/gpu.csv 2>&1 &
SMI=$!
( while true; do echo "$(date +%s.%N) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 1; done ) > $LOG/load.txt 2>&1 &
LOADPID=$!
sleep 3

cd /content/benchmarks/autokernel/files
export PYTHONPATH=/content

snapshot_cache () {   # $1 = label
  { echo "bytes: $(du -sb ~/.triton/cache 2>/dev/null | cut -f1)"
    echo "dirs: $(find ~/.triton/cache -maxdepth 1 -type d 2>/dev/null | wc -l)"
    echo "files: $(find ~/.triton/cache -type f 2>/dev/null | wc -l)"
  } > $LOG/cache_after_$1.txt 2>&1
}

do_run () {           # $1 = label
  echo "$1_START $(date +%s.%N)" >> $LOG/timings.txt
  python3 run_benchmark.py > $LOG/$1.log 2>&1
  echo "$1_RC $? " >> $LOG/timings.txt
  echo "$1_END $(date +%s.%N)" >> $LOG/timings.txt
  cp results.md results.json results_raw.json $LOG/$1/ 2>/dev/null
  snapshot_cache $1
}

do_run run1
do_run run2
do_run run3

# Control: clear ONLY the triton cache, leave everything else warm.
rm -rf ~/.triton/cache
echo "cache_cleared $(date +%s.%N)" >> $LOG/timings.txt
do_run run4

kill $SMI $LOADPID 2>/dev/null
echo "ALLDONE $(date +%s.%N)" >> $LOG/timings.txt
touch $LOG/DONE
