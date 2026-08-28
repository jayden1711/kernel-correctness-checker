"""
VM-side Triton cache tool: probe / guard / harvest.

Runs ON THE COLAB VM (uploaded by scripts/colab_bench.sh). Implements the
cache-shipping automation measured in verification_runs/triton_cache_2026-08-25/
(241.7s cold -> ~70s shipped, 19:1 return) with the staleness guard that
round's caveats called for.

THE STALENESS CONTRACT. A shipped cache carries a manifest
(triton_cache_manifest.json at the tarball root) recording the environment it
was built in: Triton version, torch version, GPU name, compute capability, and
a hash of every kernel source file under TritonBench/. `guard` re-derives all
five ON THIS VM and refuses to extract on any mismatch -- printing STALE with
the reasons and exiting 0, so the caller proceeds to a normal cold run. A
mismatch can therefore cost time (one cold compile) but never correctness.
Any error anywhere in the guard degrades the same way: no cache is always a
safe answer, a wrong cache never has to be risked. (Triton's own cache keying
would in fact just miss on stale entries rather than serve wrong code, but the
guard means we never depend on that property.)

Subcommands:
  probe   --root /content
      Print one JSON line: {triton, torch, gpu_name, gpu_cc, src_hash}.
      The dev-side wrapper uses it to pick the store entry BEFORE uploading
      21.7 MB that could not match.
  guard   --tarball /content/triton_cache.tgz --root /content
      Validate manifest against this VM; on match extract .triton into $HOME
      and print SHIPPED; else print STALE <reasons> and touch nothing.
  harvest --tarball /content/triton_cache.tgz --root /content
      Package $HOME/.triton plus a fresh manifest for this VM.

Exit code is 0 for every handled outcome including STALE; non-zero means the
tool itself broke (and the caller must treat that as STALE too).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile

MANIFEST_NAME = "triton_cache_manifest.json"


def kernel_src_hash(root):
    """sha256 over every .py under TritonBench/, path-and-content ordered.

    TritonBench holds the kernel sources the corpus compiles
    (tritonbench_registry imports TritonBench.reference.*, mutants come from
    TritonBench.cheating.*). A too-wide hash only costs a cold run; the
    near-miss family is included for the same reason.
    """
    tb = os.path.join(root, "TritonBench")
    h = hashlib.sha256()
    files = []
    for dirpath, _dirs, names in os.walk(tb):
        for n in names:
            if n.endswith(".py"):
                files.append(os.path.join(dirpath, n))
    for p in sorted(files):
        h.update(os.path.relpath(p, tb).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def env_probe(root):
    import triton
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    cc = torch.cuda.get_device_capability(0)
    return {
        "triton": triton.__version__,
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_cc": f"{cc[0]}.{cc[1]}",
        "src_hash": kernel_src_hash(root),
    }


def cmd_probe(args):
    print(json.dumps(env_probe(args.root)))


def cmd_guard(args):
    try:
        env = env_probe(args.root)
        with tarfile.open(args.tarball) as tf:
            # extractfile raises KeyError for an absent member (it returns
            # None only for non-regular members) -- found by the T4 probe.
            try:
                member = tf.extractfile(MANIFEST_NAME)
            except KeyError:
                member = None
            if member is None:
                print("STALE no-manifest")
                return
            manifest = json.load(member)
        mismatches = [
            f"{k}: cache={manifest.get(k)!r} vm={env[k]!r}"
            for k in ("triton", "torch", "gpu_name", "gpu_cc", "src_hash")
            if manifest.get(k) != env[k]
        ]
        if mismatches:
            print("STALE " + "; ".join(mismatches))
            return
        home = os.path.expanduser("~")
        # Extract only the .triton tree; the manifest stays out of $HOME.
        subprocess.run(
            ["tar", "xzf", args.tarball, "-C", home, ".triton"], check=True)
        n = sum(len(fs) for _, _, fs in
                os.walk(os.path.join(home, ".triton", "cache")))
        print(f"SHIPPED files={n}")
    except Exception as e:
        # Degrade, never block: a broken guard must cost a cold run, not the
        # session.
        print(f"STALE guard-error {type(e).__name__}: {e}")


def cmd_harvest(args):
    env = env_probe(args.root)
    home = os.path.expanduser("~")
    cache = os.path.join(home, ".triton")
    if not os.path.isdir(os.path.join(cache, "cache")):
        print("EMPTY no ~/.triton/cache to harvest")
        return
    mpath = os.path.join("/tmp", MANIFEST_NAME)
    with open(mpath, "w") as f:
        json.dump(env, f)
    subprocess.run(
        ["tar", "czf", args.tarball,
         "-C", home, ".triton",
         "-C", "/tmp", MANIFEST_NAME],
        check=True)
    print(f"HARVESTED bytes={os.path.getsize(args.tarball)} "
          f"key={env['triton']}__{env['gpu_cc']}__{env['src_hash']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("probe", cmd_probe), ("guard", cmd_guard),
                     ("harvest", cmd_harvest)):
        p = sub.add_parser(name)
        p.add_argument("--root", default="/content")
        if name != "probe":
            p.add_argument("--tarball", default="/content/triton_cache.tgz")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
