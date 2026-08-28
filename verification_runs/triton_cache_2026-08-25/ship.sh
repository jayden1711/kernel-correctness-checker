#!/bin/bash
LOG=/content/ship; mkdir -p $LOG
cd /content
tar xzf /content/kcc.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
# State BEFORE restoring the shipped cache.
python3 -c "import os,glob;p=os.path.expanduser('~/.triton/cache');print('pre-restore exists',os.path.isdir(p),'files',len(glob.glob(p+'/**/*',recursive=True)) if os.path.isdir(p) else 0)" > $LOG/pre.txt 2>&1
# Restore the cache built on the OTHER VM.
echo "untar_START $(date +%s.%N)" >> $LOG/t.txt
cd /root && tar xzf /content/triton_cache.tgz
echo "untar_END $(date +%s.%N)" >> $LOG/t.txt
{ echo "post-restore files: $(find /root/.triton/cache -type f | wc -l)"
  echo "post-restore cubins: $(find /root/.triton/cache -name '*.cubin' | wc -l)"
  echo "post-restore bytes: $(du -sb /root/.triton/cache | cut -f1)"
  nvidia-smi --query-gpu=name,uuid --format=csv,noheader
} > $LOG/restored.txt 2>&1
cd /content/benchmarks/autokernel/files
export PYTHONPATH=/content
echo "shiprun_START $(date +%s.%N)" >> $LOG/t.txt
python3 run_benchmark.py > $LOG/shiprun.log 2>&1
echo "shiprun_RC $?" >> $LOG/t.txt
echo "shiprun_END $(date +%s.%N)" >> $LOG/t.txt
mkdir -p $LOG/shiprun && cp results.md results.json results_raw.json $LOG/shiprun/ 2>/dev/null
# Did the run add any NEW compiles on top of what was shipped?
{ echo "after-run files: $(find /root/.triton/cache -type f | wc -l)"
  echo "after-run cubins: $(find /root/.triton/cache -name '*.cubin' | wc -l)"
  echo "after-run bytes: $(du -sb /root/.triton/cache | cut -f1)"
} > $LOG/after.txt 2>&1
touch $LOG/DONE
