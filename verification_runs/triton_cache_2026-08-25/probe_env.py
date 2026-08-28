"""Pristine-environment probe. MUST run before anything imports triton/torch
on this VM, so the ~/.triton/cache reading is the VM's default state and not
something this session created."""
import os, subprocess, json, glob, shutil

out = {}

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:
        return f"ERR {e}"

# --- CPU ---
out["nproc"] = sh("nproc")
out["nproc_all"] = sh("nproc --all")
out["cpu_model"] = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-").strip()
out["cpu_count_proc"] = sh("grep -c ^processor /proc/cpuinfo")
out["os_cpu_count"] = str(os.cpu_count())
out["sched_affinity"] = str(len(os.sched_getaffinity(0)))
out["memtotal_kb"] = sh("grep MemTotal /proc/meminfo")
out["cgroup_cpu_max"] = sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null")

# --- GPU ---
out["nvidia_smi_L"] = sh("nvidia-smi -L")
out["gpu_name"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

# --- Triton cache state, BEFORE we do anything ---
out["TRITON_CACHE_DIR_env"] = os.environ.get("TRITON_CACHE_DIR", "<unset>")
out["TRITON_HOME_env"] = os.environ.get("TRITON_HOME", "<unset>")
out["HOME"] = os.environ.get("HOME", "<unset>")

for label, path in [("~/.triton", os.path.expanduser("~/.triton")),
                    ("~/.triton/cache", os.path.expanduser("~/.triton/cache")),
                    ("/root/.triton/cache", "/root/.triton/cache")]:
    info = {"exists": os.path.isdir(path)}
    if info["exists"]:
        entries = glob.glob(os.path.join(path, "*"))
        info["n_entries"] = len(entries)
        info["du"] = sh(f"du -sh {path} 2>/dev/null")
        info["sample"] = [os.path.basename(e) for e in entries[:5]]
    out[f"cache::{label}"] = info

# any triton cache anywhere on disk already?
out["find_triton_cache"] = sh(
    "find / -maxdepth 6 -type d -name cache -path '*triton*' 2>/dev/null | head -20")

# is the filesystem holding HOME durable across sessions? (informational)
out["df_home"] = sh("df -h ~ | tail -1")
out["df_content"] = sh("df -h /content | tail -1")

print(json.dumps(out, indent=1))
