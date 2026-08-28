#!/bin/bash
LOG=/content/cb; mkdir -p $LOG
cd /content
tar xzf /content/kcc.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
# Confirm the cache really is empty on THIS fresh VM before we build it.
python3 -c "import os,glob;p=os.path.expanduser('~/.triton/cache');print('pre exists',os.path.isdir(p),'entries',len(glob.glob(p+'/*')) if os.path.isdir(p) else 0)" > $LOG/pre.txt 2>&1
cd /content/benchmarks/autokernel/files
export PYTHONPATH=/content
echo "build_START $(date +%s.%N)" >> $LOG/t.txt
python3 run_benchmark.py > $LOG/build.log 2>&1
echo "build_RC $?" >> $LOG/t.txt
echo "build_END $(date +%s.%N)" >> $LOG/t.txt
# Package the REAL cache.
cd /root
echo "tar_START $(date +%s.%N)" >> $LOG/t.txt
tar czf /content/triton_cache.tgz .triton
echo "tar_END $(date +%s.%N)" >> $LOG/t.txt
{ echo "raw_bytes: $(du -sb /root/.triton/cache | cut -f1)"
  echo "tgz_bytes: $(stat -c %s /content/triton_cache.tgz)"
  echo "files: $(find /root/.triton/cache -type f | wc -l)"
  echo "specializations: $(find /root/.triton/cache -name '*.cubin' | wc -l)"
  nvidia-smi --query-gpu=name --format=csv,noheader
  python3 -c "import triton,torch;print('triton',triton.__version__,'torch',torch.__version__)"
} > $LOG/cacheinfo.txt 2>&1
touch $LOG/DONE
