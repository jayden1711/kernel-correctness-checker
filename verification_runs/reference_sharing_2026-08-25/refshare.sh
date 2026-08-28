#!/bin/bash
LOG=/content/refshare; mkdir -p $LOG
cd /content && tar xzf /content/kcc6.tgz -C /content >> $LOG/setup.log 2>&1
pip install -q litellm python-dotenv >> $LOG/setup.log 2>&1
cd /root && tar xzf /content/triton_cache.tgz
export PYTHONPATH=/content
cd /content
echo "START $(date +%s.%N)" >> $LOG/t.txt
python3 /content/probe_refshare.py > $LOG/run.log 2>&1
echo "RC $? END $(date +%s.%N)" >> $LOG/t.txt
tail -6 $LOG/run.log >> $LOG/t.txt
touch $LOG/DONE
