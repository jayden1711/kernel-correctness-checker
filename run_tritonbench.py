"""
run_tritonbench.py — Run your checker against TritonBench's existing
LLM-generated Triton kernels for softmax, layernorm, matmul, and
flash attention.

Usage:
    python run_tritonbench.py                           # auto-clone and run all
    python run_tritonbench.py --repo path/to/TritonBench
    python run_tritonbench.py --softmax --matmul
"""

import os
import sys
import argparse
import importlib.util
import subprocess
import multiprocessing as mp
import torch


# Locate kernel-correctness-checker
CHECKER_ROOT = os.path.dirname(os.path.abspath(__file__))
if CHECKER_ROOT not in sys.path:
    sys.path.insert(0, CHECKER_ROOT)

from verification.checker import KernelChecker
from verification.specs.softmax import SoftmaxSpec
from verification.specs.layernorm import LayernormSpec
from verification.specs.matmul import MatmulSpec
from verification.specs.flash_attention import FlashAttentionSpec

SPEC_MAP = {
    "softmax":         SoftmaxSpec,
    "layernorm":       LayernormSpec,
    "matmul":          MatmulSpec,
    "flash_attention": FlashAttentionSpec,
}

TRITONBENCH_REPO  = "https://github.com/thunlp/TritonBench.git"
DEFAULT_CLONE_PATH = os.path.join(os.path.expanduser("~"), "TritonBench")
TIMEOUT_SECONDS   = 20

OPERATOR_KEYWORDS = {
    "softmax":         ["softmax", "soft_max"],
    # FIX: removed bare "ln" — it's a substring of unrelated filenames like
    # "gammaln.py" (the log-gamma function, nothing to do with layer norm),
    # which was silently mis-categorizing files into the layernorm bucket.
    # "layernorm" and "layer_norm" are specific enough on their own.
    "layernorm":       ["layernorm", "layer_norm"],
    "matmul":          ["matmul", "mat_mul", "gemm", "matrix_mult"],
    "flash_attention": ["flash", "attention", "flash_attn", "flashattn"],
}

FUNC_CANDIDATES = {
    "softmax":         ["softmax", "softmax_forward", "fused_softmax",
                        "triton_softmax", "forward", "run"],
    "layernorm":       ["layernorm", "layer_norm", "layernorm_forward",
                        "fused_layernorm", "triton_layernorm", "forward", "run"],
    "matmul":          ["matmul", "matrix_multiply", "triton_matmul",
                        "matmul_forward", "fused_matmul", "gemm",
                        "matmul_kernel", "forward", "run"],
    "flash_attention": ["flash_attention", "flash_attn", "attention",
                        "fused_attention", "flash_attention_forward",
                        "forward", "run"],
}


# Subprocess worker — runs in an isolated process so hung GPU kernels
# can be killed cleanly with p.kill()


def _worker(path, operator, func_candidates, checker_root, q):
    import sys, torch, importlib.util, io, contextlib, inspect
    sys.modules['triton_viz'] = None  # prevent hang in subprocess
    torch.manual_seed(42)
    sys.path.insert(0, checker_root)

    from verification.checker import KernelChecker
    from verification.specs.softmax import SoftmaxSpec
    from verification.specs.layernorm import LayernormSpec
    from verification.specs.matmul import MatmulSpec
    from verification.specs.flash_attention import FlashAttentionSpec

    SPEC = {
        "softmax":         SoftmaxSpec,
        "layernorm":       LayernormSpec,
        "matmul":          MatmulSpec,
        "flash_attention": FlashAttentionSpec,
    }

    spec = importlib.util.spec_from_file_location("_km", path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    except Exception as e:
        q.put(("LOAD_ERROR", str(e)))
        return

    fn = None
    for name in func_candidates:
        f = getattr(mod, name, None)
        if callable(f):
            fn = f
            break
    if fn is None:
        # fallback: first non-private callable that isn't a class
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr)
            if callable(obj) and not isinstance(obj, type):
                fn = obj
                break

    try:
        if operator == "softmax":
            x   = torch.rand(512, 512, device="cuda")
            ref = torch.softmax(x, dim=1)
            out = fn(x)
            ac  = bool(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))
            ac_detail = f"max_err={float((out-ref).abs().max()):.6f}"

        elif operator == "layernorm":
            x = torch.rand(512, 512, device="cuda")
            w = torch.ones(512, device="cuda")
            b = torch.zeros(512, device="cuda")
            ref = torch.nn.functional.layer_norm(x, [512])

            # Logs the real signature plus the specific failure reason for
            # every attempted convention, surfaced through ac_detail instead
            # of vanishing into a bare `except: continue`.
            try:
                sig_str = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig_str = "<signature unavailable — likely a C/triton-wrapped callable>"

            conventions = [
                ("(x)",                    lambda: fn(x)),
                ("(x, w, b)",               lambda: fn(x, w, b)),
                ("(x, w, b, eps)",          lambda: fn(x, w, b, 1e-5)),
                ("(x, w, eps)",             lambda: fn(x, w, 1e-5)),
                ("(x, shape, w, b)",        lambda: fn(x, [512], w, b)),
                ("(x, eps=1e-5)",           lambda: fn(x, eps=1e-5)),
                ("(x, weight=w, bias=b)",   lambda: fn(x, weight=w, bias=b)),
            ]

            out = None
            attempt_log = []
            for label, call in conventions:
                try:
                    r = call()
                    if r is not None and hasattr(r, "shape") and r.shape == x.shape:
                        out = r
                        break
                    elif r is not None and hasattr(r, "shape"):
                        attempt_log.append(f"{label} -> wrong shape {tuple(r.shape)}")
                    else:
                        attempt_log.append(f"{label} -> returned {type(r).__name__}")
                except Exception as e:
                    attempt_log.append(f"{label} -> {type(e).__name__}: {e}")

            if out is None:
                raise RuntimeError(
                    f"no calling convention worked for fn='{fn.__name__}' "
                    f"signature={sig_str}. Attempts: " + " | ".join(attempt_log)
                )

            ac = bool(torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3))
            ac_detail = f"max_err={float((out.float()-ref.float()).abs().max()):.6f}"

        elif operator == "matmul":
            A   = torch.rand(256, 256, device="cuda")
            B   = torch.rand(256, 256, device="cuda")
            ref = torch.matmul(A, B)
            out = fn(A, B)
            ac  = bool(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))
            ac_detail = f"max_err={float((out-ref).abs().max()):.6f}"

        elif operator == "flash_attention":
            Q   = torch.rand(128, 64, device="cuda")
            K   = torch.rand(128, 64, device="cuda")
            V   = torch.rand(128, 64, device="cuda")
            d   = Q.shape[-1]
            ref = torch.softmax(Q @ K.T / (d**0.5), dim=-1) @ V
            out = fn(Q, K, V)
            ac  = bool(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))
            ac_detail = f"max_err={float((out-ref).abs().max()):.6f}"
        else:
            ac, ac_detail = False, "unknown operator"
    except Exception as e:
        ac, ac_detail = False, str(e)

    try:
        kspec   = SPEC[operator](name=operator, requires_backward=False)
        checker = KernelChecker(kspec)

        if operator == "softmax":
            x       = torch.rand(512, 512, device="cuda")
            results = checker.run(fn, None,
                                  lambda inp: torch.softmax(inp, dim=1), x)
        elif operator == "layernorm":
            x       = torch.rand(512, 512, device="cuda")
            results = checker.run(fn, None,
                                  lambda inp: torch.nn.functional.layer_norm(
                                      inp, [inp.shape[-1]]), x)
        elif operator == "matmul":
            A       = torch.rand(256, 256, device="cuda")
            B       = torch.rand(256, 256, device="cuda")
            results = checker.run(fn, None,
                                  lambda a, b: torch.matmul(a, b), (A, B))
        elif operator == "flash_attention":
            Q       = torch.rand(128, 64, device="cuda")
            K       = torch.rand(128, 64, device="cuda")
            V       = torch.rand(128, 64, device="cuda")
            d       = Q.shape[-1]
            results = checker.run(fn, None,
                                  lambda q, k, v: torch.softmax(
                                      q @ k.T / (d**0.5), dim=-1) @ v,
                                  (Q, K, V))
        else:
            results = []

        failed       = [r.check_name for r in results if not r.passed]
        checker_pass = len(failed) == 0
    except Exception as e:
        checker_pass, failed = False, [str(e)]

    q.put(("OK", ac, ac_detail, checker_pass, failed))



# Helpers


def clone_tritonbench(path):
    if os.path.isdir(path):
        print(f"TritonBench already at {path}, skipping clone.")
        return
    print(f"Cloning TritonBench to {path}...")
    subprocess.run(["git", "clone", "--depth", "1", TRITONBENCH_REPO, path],
                   check=True)
    print("Clone complete.")


def detect_operator(path):
    name = os.path.basename(path).lower()
    for op, keywords in OPERATOR_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return op
    return None

SKIP_FILES = {
    "streamk_matmul.py",
    "matmul_leakyrelu_fp8.py",
    "matmul_triton1.py",
    "matmul_triton2.py",
    "matmul_dequantize_int4.py",
    "matmul_triton_autotune.py",
    "parallel_retention_attention.py",
    "lightning_attention.py",
    "parallel_attention.py",
    "triton_attention.py",
    "attention_kernel.py",
}

def find_kernels(repo_path, operators):
    results = {op: [] for op in operators}
    for search_dir in [os.path.join(repo_path, "LLM_generated"),
                       os.path.join(repo_path, "data")]:
        if not os.path.isdir(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                if fname in SKIP_FILES:
                    continue
                full = os.path.join(root, fname)
                op   = detect_operator(fname) or detect_operator(root)
                if op and op in operators:
                    results[op].append(full)
    return results


def evaluate(path, operator, repo_path):
    """Run the worker in a subprocess with a hard timeout."""
    q = mp.Queue()
    p = mp.Process(target=_worker,
                   args=(path, operator,
                         FUNC_CANDIDATES[operator],
                         CHECKER_ROOT, q))
    p.start()
    p.join(timeout=TIMEOUT_SECONDS)

    if p.is_alive():
        p.kill()
        p.join()
        return "TIMEOUT", "TIMEOUT", False, ["killed after timeout"]

    if q.empty():
        return "ERROR", "no result", False, ["subprocess exited with no output"]

    res = q.get()
    if res[0] == "SKIP":
        return "SKIP", res[1], False, []
    if res[0] == "LOAD_ERROR":
        return "LOAD_ERROR", res[1], False, []

    _, ac, ac_detail, checker_pass, failed = res
    return ("PASS" if ac else "FAIL"), ac_detail, checker_pass, failed



# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",           default=DEFAULT_CLONE_PATH)
    parser.add_argument("--softmax",        action="store_true")
    parser.add_argument("--layernorm",      action="store_true")
    parser.add_argument("--matmul",         action="store_true")
    parser.add_argument("--flash_attention",action="store_true")
    parser.add_argument("--all",            action="store_true")
    args = parser.parse_args()

    if args.all:
        operators = list(SPEC_MAP.keys())
    else:
        operators = [op for op in SPEC_MAP if getattr(args, op, False)]
    if not operators:
        operators = list(SPEC_MAP.keys())

    clone_tritonbench(args.repo)
    kernel_map = find_kernels(args.repo, operators)

    total_found = sum(len(v) for v in kernel_map.values())
    if total_found == 0:
        print("No matching kernels found.")
        return

    print(f"\nFound {total_found} kernel(s). Timeout per kernel: {TIMEOUT_SECONDS}s\n")

    rows = []

    for operator in operators:
        paths = kernel_map[operator]
        if not paths:
            print(f"── {operator}: no kernels found.")
            continue
        print(f"── {operator} ({len(paths)} kernel(s))")

        for path in paths:
            fname = os.path.relpath(path, args.repo)
            ac_str, ac_detail, checker_pass, failed = evaluate(
                path, operator, args.repo)

            if ac_str in ("SKIP", "LOAD_ERROR"):
                print(f"   SKIP {fname}: {ac_detail}")
                rows.append((operator, fname, ac_str, "N/A", ac_detail))
                continue

            checker_str  = "PASS" if checker_pass else "FAIL"
            interesting  = (ac_str == "PASS") and not checker_pass
            marker       = " ◄ caught by checker only" if interesting else ""

            if ac_str == "TIMEOUT":
                print(f"   TIMEOUT {fname}")
            else:
                print(f"   {fname}")
                print(f"     allclose: {ac_str} ({ac_detail})")
                print(f"     checker:  {checker_str}"
                      + (f"  failed: {failed}" if not checker_pass else "")
                      + marker)

            rows.append((operator, fname, ac_str, checker_str,
                         ", ".join(failed) if not checker_pass else ""))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Operator':<16} {'File':<35} {'allclose':<10} {'Checker'}")
    print("  " + "-" * 68)
    for op, fname, ac, ch, _ in rows:
        print(f"  {op:<16} {os.path.basename(fname):<35} {ac:<10} {ch}")

    valid   = [r for r in rows if r[2] not in ("SKIP", "LOAD_ERROR", "TIMEOUT", "ERROR")]
    total   = len(valid)
    ac_ok   = sum(1 for r in valid if r[2] == "PASS")
    ch_ok   = sum(1 for r in valid if r[3] == "PASS")
    caught  = sum(1 for r in valid if r[2] == "PASS" and r[3] == "FAIL")

    print("\n" + "=" * 70)
    print(f"  Total evaluated         : {total}")
    print(f"  Pass naive allclose     : {ac_ok} / {total}")
    print(f"  Pass full checker       : {ch_ok} / {total}")
    print(f"  Caught by checker only  : {caught}  "
          f"(passed allclose, failed deeper verification)")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()