"""
Staleness-guard verification, run ON THE VM (session 1, after the cold run
harvested a genuine cache tarball to /content/triton_cache.tgz).

The guard's contract: a mismatched manifest must degrade to a cold run --
refuse to extract, touch nothing, exit 0 -- and a matching manifest must
extract. Verified here with the REAL harvested tarball on the REAL VM, four
ways:

  T1  matching manifest        -> SHIPPED, ~/.triton populated
  T2  doctored triton version  -> STALE naming the field, ~/.triton absent
  T3  doctored src_hash        -> STALE naming the field, ~/.triton absent
  T4  tarball with NO manifest -> STALE no-manifest, ~/.triton absent
  T5  after a STALE refusal, a fresh triton compile still works (the
      degrade-to-cold path produces a working session, not a broken one)

Between tests ~/.triton is moved aside/removed so presence after the guard is
attributable to the guard alone. The real cache is restored at the end.

Run: PYTHONPATH=/content python3 /content/staleness_test.py
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile

HOME = os.path.expanduser("~")
TRITON_DIR = os.path.join(HOME, ".triton")
REAL = "/content/triton_cache.tgz"
TOOL = "/content/scripts/vm_cache_tool.py"
MAN = "triton_cache_manifest.json"

results = []


def ck(label, cond, ctx=""):
    print(("  PASS  " if cond else "  FAIL  ") + label
          + (f"   [{ctx}]" if not cond else ""), flush=True)
    results.append((label, bool(cond)))


def guard(tarball):
    r = subprocess.run([sys.executable, TOOL, "guard", "--root", "/content",
                        "--tarball", tarball], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def clear_cache():
    shutil.rmtree(TRITON_DIR, ignore_errors=True)


def doctored(field, value, out):
    with tarfile.open(REAL) as tf:
        manifest = json.load(tf.extractfile(MAN))
    manifest[field] = value
    mpath = "/tmp/" + MAN
    with open(mpath, "w") as f:
        json.dump(manifest, f)
    # Rebuild: same .triton payload, doctored manifest.
    subprocess.run(["bash", "-c",
                    f"cd /tmp && rm -rf stale_work && mkdir stale_work && "
                    f"tar xzf {REAL} -C stale_work && "
                    f"cp {mpath} stale_work/{MAN} && "
                    f"tar czf {out} -C stale_work .triton {MAN}"],
                   check=True)


assert os.path.exists(REAL), "run after the cold-run harvest"

# T1: the genuine tarball extracts
clear_cache()
rc, msg = guard(REAL)
ck("T1 genuine tarball -> SHIPPED", rc == 0 and "SHIPPED" in msg, msg)
ck("T1 cache present after guard",
   os.path.isdir(os.path.join(TRITON_DIR, "cache")))

# T2: doctored triton version
clear_cache()
doctored("triton", "9.9.9", "/tmp/stale_triton.tgz")
rc, msg = guard("/tmp/stale_triton.tgz")
ck("T2 doctored triton -> STALE, names the field",
   rc == 0 and "STALE" in msg and "triton" in msg, msg)
ck("T2 nothing extracted", not os.path.isdir(TRITON_DIR))

# T3: doctored src_hash (the kernel-source key)
doctored("src_hash", "deadbeefdeadbeef", "/tmp/stale_hash.tgz")
rc, msg = guard("/tmp/stale_hash.tgz")
ck("T3 doctored src_hash -> STALE, names the field",
   rc == 0 and "STALE" in msg and "src_hash" in msg, msg)
ck("T3 nothing extracted", not os.path.isdir(TRITON_DIR))

# T4: manifest missing entirely
subprocess.run(["bash", "-c",
                "cd /tmp && rm -rf noman && mkdir noman && "
                f"tar xzf {REAL} -C noman && "
                "tar czf /tmp/stale_noman.tgz -C noman .triton"], check=True)
rc, msg = guard("/tmp/stale_noman.tgz")
ck("T4 no manifest -> STALE no-manifest",
   rc == 0 and "STALE" in msg and "no-manifest" in msg, msg)
ck("T4 nothing extracted", not os.path.isdir(TRITON_DIR))

# T5: degrade-to-cold really is a WORKING state -- compile one kernel fresh
import torch
import triton
import triton.language as tl


@triton.jit
def _stale_probe_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)


x = torch.arange(64, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
_stale_probe_kernel[(1,)](x, y, 64, BLOCK=64)
ck("T5 fresh compile after refusal works, correct output",
   torch.equal(y, x * 2.0))
ck("T5 the compile populated a fresh cache (cold path is alive)",
   os.path.isdir(os.path.join(TRITON_DIR, "cache")))

# restore the real cache for whatever runs next in this session
clear_cache()
rc, msg = guard(REAL)
ck("restore: genuine tarball -> SHIPPED again", "SHIPPED" in msg, msg)

n_fail = sum(1 for _, ok in results if not ok)
print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} "
      f"({len(results)} checks)")
json.dump([{"label": l, "ok": ok} for l, ok in results],
          open("/content/staleness_results.json", "w"), indent=1)
sys.exit(1 if n_fail else 0)
