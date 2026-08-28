#!/bin/bash
# One-command corpus benchmark on a fresh Colab GPU, with automatic Triton
# cache shipping -- the workflow verification_runs/triton_cache_2026-08-25/
# measured (241.7s cold -> ~70s shipped, 8.8s to ship, 19:1 return) baked
# into the standard path so nobody has to remember ship.sh.
#
#   scripts/colab_bench.sh                      # default session kccbench, T4
#   scripts/colab_bench.sh -s mysess -g T4 -k   # named session, keep it alive
#
# What it does, in order:
#   1. stage the source (tar incl. TritonBench -- the SESSION_HANDOFF §0 trap)
#   2. provision/reuse the session, upload, extract
#   3. probe the VM (triton/torch/GPU/cc + kernel-source hash) and look for a
#      matching cache in the local store (.triton_cache_store/, gitignored)
#   4. HIT: upload the ~21.7MB tarball; the VM-side guard re-validates the
#      manifest against the live environment and extracts only on a full
#      match -- any mismatch prints STALE and the run proceeds COLD. The
#      guard can cost a cold run, never a wrong answer.
#      MISS: run cold.
#   5. run benchmarks/autokernel/files/run_benchmark.py (nohup + poll, so a
#      reclaimed VM costs the run, not the terminal)
#   6. download results.{md,json,raw} into results_gpu/<session>_<UTC>/ and
#      print the your_checker catch/FP regression line
#   7. if the run was cold, harvest the freshly built cache into the store
#      keyed by (triton, gpu_cc, src_hash) so the NEXT session ships it
#   8. stop the session (unless -k)
#
# The colab CLI needs HOME=~/.colab-home (see SESSION_HANDOFF §0); this
# script sets it only for the colab invocations, not globally.
set -u

SESSION=kccbench
GPU=T4
KEEP=0
STORE_DEFAULT="$(cd "$(dirname "$0")/.." && pwd)/.triton_cache_store"
STORE="${KCC_CACHE_STORE:-$STORE_DEFAULT}"
while getopts "s:g:kS:" opt; do
  case $opt in
    s) SESSION=$OPTARG ;;
    g) GPU=$OPTARG ;;
    k) KEEP=1 ;;
    S) STORE=$OPTARG ;;
    *) echo "usage: $0 [-s session] [-g gpu] [-k] [-S store_dir]"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p "$STORE"
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT="$REPO/results_gpu/${SESSION}_${STAMP}"
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cb() { HOME=~/.colab-home colab "$@"; }
vexec() {  # run a generated python snippet on the VM, echo its stdout
  local f="$TMP/snippet_$RANDOM.py"
  cat > "$f"
  HOME=~/.colab-home colab exec -s "$SESSION" -f "$f" --timeout "${VTIMEOUT:-300}"
}

echo "== [1/8] staging source =="
tar --exclude='__pycache__' --exclude='.venv' -czf "$TMP/kcc.tgz" \
    verification benchmarks scripts tests TritonBench || exit 1

echo "== [2/8] session '$SESSION' ($GPU) =="
cb new --gpu "$GPU" -s "$SESSION" || echo "   (session may already exist -- reusing)"
cb upload -s "$SESSION" "$TMP/kcc.tgz" /content/kcc.tgz || exit 1
vexec <<'EOF' || exit 1
import subprocess
subprocess.run(["bash","-c",
  "tar xzf /content/kcc.tgz -C /content && "
  "pip install -q litellm python-dotenv"], check=True)
print("extracted")
EOF

echo "== [3/8] probing VM environment =="
PROBE=$(vexec <<'EOF'
import subprocess
r = subprocess.run(["python3","/content/scripts/vm_cache_tool.py","probe",
                    "--root","/content"], capture_output=True, text=True)
print(r.stdout.strip() or ("PROBE_FAILED " + r.stderr[-200:]))
EOF
)
PROBE_JSON=$(echo "$PROBE" | grep -o '{.*}' | tail -1)
if [ -z "$PROBE_JSON" ]; then
  echo "   probe failed ($PROBE) -- proceeding COLD with no harvest"
  KEY=""
else
  KEY=$(python3 -c "
import json,sys
d = json.loads('''$PROBE_JSON''')
print(f\"{d['triton']}__{d['gpu_cc']}__{d['src_hash']}\")")
  echo "   cache key: $KEY"
fi

SHIPPED=0
if [ -n "$KEY" ] && [ -f "$STORE/triton_cache_${KEY}.tgz" ]; then
  echo "== [4/8] cache HIT -- shipping $(du -h "$STORE/triton_cache_${KEY}.tgz" | cut -f1) =="
  cb upload -s "$SESSION" "$STORE/triton_cache_${KEY}.tgz" /content/triton_cache.tgz
  GUARD=$(vexec <<'EOF'
import subprocess
r = subprocess.run(["python3","/content/scripts/vm_cache_tool.py","guard",
                    "--root","/content","--tarball","/content/triton_cache.tgz"],
                   capture_output=True, text=True)
print((r.stdout + r.stderr).strip()[-300:])
EOF
)
  echo "   guard: $GUARD"
  case "$GUARD" in *SHIPPED*) SHIPPED=1 ;; *) echo "   (stale/mismatch -- running cold)" ;; esac
else
  echo "== [4/8] cache MISS for this key -- running cold (will harvest) =="
fi

echo "== [5/8] running benchmark (nohup + poll) =="
vexec <<'EOF' || exit 1
import subprocess
subprocess.run(["bash","-c","rm -f /content/BENCH_DONE /content/bench.rc"])
subprocess.Popen(["bash","-c",
  "nohup bash -c 'cd /content/benchmarks/autokernel/files && "
  "PYTHONPATH=/content python3 run_benchmark.py > /content/bench.log 2>&1; "
  "echo $? > /content/bench.rc; touch /content/BENCH_DONE' "
  "> /dev/null 2>&1 &"])
print("launched")
EOF
T0=$(date +%s)
while true; do
  sleep 30
  DONE=$(vexec <<'EOF'
import os
print("DONE" if os.path.exists("/content/BENCH_DONE") else "RUNNING")
EOF
)
  case "$DONE" in
    *DONE*) break ;;
    *RUNNING*) echo "   ... running ($(( $(date +%s) - T0 ))s)" ;;
    *) echo "   poll error: $DONE (retrying)" ;;
  esac
  if [ $(( $(date +%s) - T0 )) -gt 1800 ]; then echo "TIMEOUT"; exit 1; fi
done
WALL=$(( $(date +%s) - T0 ))
echo "   benchmark finished in ~${WALL}s (poll-granular)"

echo "== [6/8] downloading results to $OUT =="
for f in results.md results.json results_raw.json; do
  cb download -s "$SESSION" "/content/benchmarks/autokernel/files/$f" "$OUT/$f" 2>/dev/null
done
cb download -s "$SESSION" /content/bench.log "$OUT/bench.log" 2>/dev/null
python3 - "$OUT/results.json" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    for name in ("your_checker (full)",):
        s = d[name]
        print(f"   REGRESSION {name}: catch {s.get('catch_rate')!r} "
              f"fp {s.get('false_positive_rate')!r}")
except Exception as e:
    print(f"   (could not parse results.json: {e})")
EOF

if [ "$SHIPPED" = "0" ] && [ -n "$KEY" ]; then
  echo "== [7/8] harvesting cache into store =="
  vexec <<'EOF'
import subprocess
r = subprocess.run(["python3","/content/scripts/vm_cache_tool.py","harvest",
                    "--root","/content","--tarball","/content/triton_cache.tgz"],
                   capture_output=True, text=True)
print((r.stdout + r.stderr).strip()[-300:])
EOF
  cb download -s "$SESSION" /content/triton_cache.tgz "$STORE/triton_cache_${KEY}.tgz" \
    && echo "   stored: $STORE/triton_cache_${KEY}.tgz"
else
  echo "== [7/8] cache was shipped -- nothing to harvest =="
fi

if [ "$KEEP" = "0" ]; then
  echo "== [8/8] stopping session =="
  cb stop -s "$SESSION"
else
  echo "== [8/8] keeping session alive (-k) =="
fi
echo "DONE  results: $OUT"
