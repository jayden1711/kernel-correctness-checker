"""
verification/tritonbench_adapter.py

Adapter for running checker-style verification against TritonBench-G
reference/test file pairs, where the candidate is an LLM-generated kernel
that must expose the same top-level callable name as the reference.

Confirmed against 4 real TritonBench_G_v1 files (softmax_optimize.py,
layer_norm_triton.py, matmul_tma.py, flash_attn.py), NOT the full ~140.
Treat this as validated for those four shapes, not the whole benchmark.

Core idea:
  - Each reference file is split by a literal line of 146 '#' characters
    into (kernel_src, test_src).
  - test_src calls the reference's public entry point by name. That name
    is NOT reliably a top-level `def` in kernel_src (layer_norm_triton.py
    binds it via `layer_norm = LayerNorm.apply`, not a `def`). So the name
    is extracted from the CALL SITE in test_src, not from kernel_src's
    definitions, usage is the more reliable source of the contract.
  - The candidate is required (matching TritonBench's own EVAL/*/0_call_acc.py
    convention) to define a callable of that same name.
  - We monkeypatch that name in a candidate namespace with a spy, execute
    test_src once, and capture the real (args, kwargs) used for every
    call. This sidesteps needing to understand the test body's control
    flow or return-value shape (test_layer_norm_with_backward returns a
    dict of floats, not a tensor, capturing at the call site works
    regardless of what the test does with the result afterward).

KNOWN LIMITATIONS (confirmed gaps, not hidden ones):
  - Assumes the wrapper is called with a bare name (`foo(...)`), not via
    attribute access (`mod.foo(...)`) or through a local alias assigned
    inside the test body. Not yet observed in the 4 files checked, but
    that's a small sample against ~140 real kernels.
  - Assumes exactly one '#'*146 separator; raises loudly (does not guess)
    if that assumption doesn't hold for a given file.
  - Layer-3 algebraic properties only run when the captured call matches
    a known shape convention (see run_layer3_if_applicable below). Real
    kernels that don't match, legitimately skip
    Layer 3, and that skip is logged, not silent.
"""

import ast
import importlib.util
import os
import tempfile
import textwrap
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance
from verification.layer1_structural.runtime_guards import check_determinism
from verification.layer3_properties.attention_batched_properties import (
    try_attention_layer3,
)
from verification.layer3_properties.layernorm_generic_properties import (
    try_layernorm_layer3,
)
from verification.layer3_properties.matmul_generic_properties import (
    try_matmul_layer3,
)
from verification.layer3_properties.softmax_generic_properties import (
    try_softmax_layer3,
)

SEPARATOR = "#" * 146

# Reference-file parsing

def split_reference_file(path: str) -> tuple[str, str]:
    """Split a TritonBench_G_v1 reference file into (kernel_src, test_src)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if SEPARATOR not in content:
        raise ValueError(
            f"{path}: expected separator ('#'*146) not found. This file may "
            "not follow the TritonBench_G_v1 convention -- inspect manually, "
            "do not assume the split."
        )
    parts = content.split(SEPARATOR)
    if len(parts) != 2:
        raise ValueError(
            f"{path}: expected exactly one separator, found {len(parts) - 1}. "
            "Falling back to (everything before first, everything after last) "
            "but verify this file's structure before trusting results."
        )
    return parts[0], parts[-1]


def extract_entry_point_name(test_src: str) -> str:
    """
    Find the name the test body treats as the kernel's public entry point,
    by locating bare-name Call nodes (`foo(...)`, not `mod.foo(...)` or
    `obj.method(...)`) and returning the most-called qualifying name.

    Deliberately usage-based, not definition-based: the reference file may
    expose its entry point as a `def`, a `torch.autograd.Function.apply`
    alias, or something else. Whatever the test calls IS the contract the
    candidate must satisfy -- confirmed necessary by layer_norm_triton.py,
    where the reference binds `layer_norm = LayerNorm.apply` with no
    matching top-level `def` at all.
    """
    tree = ast.parse(textwrap.dedent(test_src))

    EXCLUDE = {
        "print", "range", "len", "open", "isinstance", "getattr", "setattr",
        "hasattr", "float", "int", "str", "list", "dict", "tuple", "set",
    }

    call_counts: dict[str, int] = {}
    call_order: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in EXCLUDE or name.startswith("test_"):
                continue
            call_counts[name] = call_counts.get(name, 0) + 1
            if name not in call_order:
                call_order.append(name)

    if not call_counts:
        raise ValueError(
            "No bare-name function calls found in test body. The entry "
            "point may be called via attribute access (e.g. obj.method(...)) "
            "-- extend extraction logic manually for this file rather than "
            "guessing at a generic rule."
        )

    best = max(call_order, key=lambda n: (call_counts[n], -call_order.index(n)))
    return best


def _load_module_from_source(src: str, tag: str) -> dict:
    """
    Write `src` to a REAL temp .py file and import it via importlib.


    Each call gets a unique filename so the reference and candidate
    (which likely define same-named functions/kernels) don't collide as
    Python module-cache entries.
    """
    tmp_dir = tempfile.gettempdir()
    fname = f"_tb_{tag}_{uuid.uuid4().hex}.py"
    fpath = os.path.join(tmp_dir, fname)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(src)

    spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


# Call capture

@dataclass
class CapturedCall:
    args: tuple
    kwargs: dict


class _CallSpy:
    """Wraps a candidate callable, records every (args, kwargs) it's
    invoked with, then forwards the call through unchanged."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.calls: list[CapturedCall] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(CapturedCall(args=args, kwargs=kwargs))
        return self.fn(*args, **kwargs)


# Result types


@dataclass
class AdapterResult:
    file: str
    entry_point: Optional[str] = None
    load_error: Optional[str] = None
    captured_calls: int = 0
    per_call_checks: list = field(default_factory=list)


# Main entry point

def run_candidate_against_reference(
    reference_path: str,
    candidate_src: str,
) -> AdapterResult:
    """
    Full pipeline: split reference, extract entry point, capture real
    calls from the test body against the CANDIDATE's implementation, then
    run universal correctness checks (plus Layer-3 attention checks where
    the captured call matches that shape).
    """
    result = AdapterResult(file=reference_path)

    try:
        kernel_src, test_src = split_reference_file(reference_path)
    except Exception as e:
        result.load_error = f"split failed: {e}"
        return result

    try:
        entry_point = extract_entry_point_name(test_src)
        result.entry_point = entry_point
    except Exception as e:
        result.load_error = f"entry-point extraction failed: {e}"
        return result

    try:
        ref_ns = _load_module_from_source(kernel_src, tag="reference")
    except Exception as e:
        result.load_error = f"reference load failed: {type(e).__name__}: {e}"
        return result

    if entry_point not in ref_ns or not callable(ref_ns[entry_point]):
        result.load_error = (
            f"entry point '{entry_point}' not found/callable in reference "
            "namespace -- extraction likely wrong for this file's structure."
        )
        return result
    reference_fn = ref_ns[entry_point]

    try:
        cand_ns = _load_module_from_source(candidate_src, tag="candidate")
    except Exception as e:
        result.load_error = f"candidate load failed: {type(e).__name__}: {e}"
        return result

    if entry_point not in cand_ns or not callable(cand_ns[entry_point]):
        result.load_error = (
            f"candidate does not define callable '{entry_point}' -- "
            "call-accuracy failure (matches TritonBench's own "
            "0_call_acc.py notion of a failed call, not a numeric bug)."
        )
        return result
    candidate_fn = cand_ns[entry_point]

    spy = _CallSpy(candidate_fn)
    cand_ns[entry_point] = spy

    # test_src is exec'd from a REAL temp file, not a synthetic filename,
    # for the same reason kernel_src/candidate_src are (see
    # _load_module_from_source). 
    test_tmp_path = os.path.join(tempfile.gettempdir(), f"_tb_test_{uuid.uuid4().hex}.py")
    with open(test_tmp_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(test_src))

    try:
        with open(test_tmp_path, "r", encoding="utf-8") as f:
            test_code = f.read()
        exec(compile(test_code, filename=test_tmp_path, mode="exec"), cand_ns)
    except Exception as e:
        result.load_error = (
            f"test body raised against candidate: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc(limit=3)}"
        )
        result.captured_calls = len(spy.calls)
        if not spy.calls:
            return result  # nothing captured before the failure -- no signal at all

    result.captured_calls = len(spy.calls)

    if cand_ns.get(entry_point) is not spy:
        # CONFIRMED via quantize_kv_copy.py: test_src can contain its own
        # top-level 'def <entry_point>(...)' (a full second copy of the
        # reference implementation, embedded in what should be the
        # isolated test section). Executing that def statement REBINDS
        # cand_ns[entry_point] away from the spy, silently. The test then
        # calls the just-redefined REFERENCE code, not the candidate,
        # zero exception, zero indication anything is wrong, and
        # (elsewhere) potentially a hybrid candidate-wrapper/leaked-
        # reference-kernel execution that could produce a false PASS.
        # This may not be a one-off: the repo's own last commit to data/
        # is titled "fix leakeage in tests," suggesting this exact
        # failure class was a known, possibly incompletely-fixed issue
        # across the dataset. Flag explicitly rather than silently
        # reporting an empty/misleading result.
        result.load_error = (
            f"test_src redefines '{entry_point}' at module scope, overwriting the "
            "candidate substitution silently (spy no longer bound at that name after "
            "test_src executed). This file's test section embeds reference code that "
            "leaks into the isolated test scope -- results from this file cannot be "
            "trusted as testing the candidate. Do not report a PASS/FAIL from this file "
            "without manual inspection."
        )
        return result

    for i, call in enumerate(spy.calls):
        record = _check_one_call(candidate_fn, reference_fn, call.args, call.kwargs)
        record["call_index"] = i
        result.per_call_checks.append(record)

    return result

# Per-call universal checks (signature-agnostic)

def _first_tensor(args: tuple, kwargs: dict) -> Optional[torch.Tensor]:
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


def _check_one_call(
    candidate_fn: Callable,
    reference_fn: Callable,
    args: tuple,
    kwargs: dict,
) -> dict:
    """
    Universal, signature-agnostic checks for one captured call -- these
    work for any (*args, **kwargs) pair, which is what lets this
    generalize across operators without a per-operator KernelSpec.

    Deliberately excluded here: general algebraic properties (those need
    to know what the operator IS). The one exception is attention's
    sum-to-one / bounded-by-values, wired in separately below because
    those two properties are cheap to detect (Q,K,V-shaped call) and
    provably still hold under causal masking -- see
    layer3_properties/attention_batched_properties.py for the reasoning.
    """
    out: dict[str, Any] = {"passed": True, "checks": {}}

    try:
        cand_out = candidate_fn(*args, **kwargs)
    except Exception as e:
        out["passed"] = False
        out["checks"]["executes"] = (False, f"{type(e).__name__}: {e}")
        return out
    out["checks"]["executes"] = (True, "ran without exception")

    try:
        ref_out = reference_fn(*args, **kwargs)
    except Exception as e:
        out["checks"]["reference_executes"] = (False, f"{type(e).__name__}: {e}")
        out["checks"]["note"] = "reference itself failed on these captured args -- cannot compare"
        return out

    if isinstance(cand_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
        shape_ok = cand_out.shape == ref_out.shape
        out["checks"]["shape_match"] = (
            shape_ok, f"candidate {tuple(cand_out.shape)} vs reference {tuple(ref_out.shape)}"
        )
        if not shape_ok:
            out["passed"] = False
            return out

        dtype_ok = cand_out.dtype == ref_out.dtype
        out["checks"]["dtype_match"] = (dtype_ok, f"{cand_out.dtype} vs {ref_out.dtype}")
        if not dtype_ok:

            out["passed"] = False

        finite_input = all(
            (not isinstance(a, torch.Tensor)) or torch.isfinite(a).all()
            for a in args
        ) and all(
            (not isinstance(v, torch.Tensor)) or torch.isfinite(v).all()
            for v in kwargs.values()
        )

        if not finite_input:
            cand_nan_mask = torch.isnan(cand_out) | torch.isinf(cand_out)
            ref_nan_mask = torch.isnan(ref_out) | torch.isinf(ref_out)
            pattern_matches = torch.equal(cand_nan_mask, ref_nan_mask)
            out["checks"]["nan_inf"] = (
                pattern_matches,
                "input non-finite; candidate/reference non-finite pattern match" if pattern_matches
                else "input non-finite, but candidate and reference disagree on WHERE -- real discrepancy"
            )
            if not pattern_matches:
                out["passed"] = False
                return out

            # NaN-aware allclose: equal_nan=True treats matching NaNs as
            # equal instead of automatically failing. Still meaningful,
            # it will correctly catch a candidate that gets the FINITE
            # elements of a mixed finite/non-finite tensor wrong, it just
            # stops incorrectly failing on the non-finite elements.
            allclose_ok = torch.allclose(
                cand_out.float(), ref_out.float(), atol=1e-3, rtol=1e-2, equal_nan=True
            )
            out["checks"]["allclose"] = (
                allclose_ok,
                "matches (NaN-aware comparison; raw max_err is undefined when NaNs are present)"
                if allclose_ok else "mismatch on finite elements despite matching non-finite pattern"
            )
            if not allclose_ok:
                out["passed"] = False

            # perturbation_tolerance and determinism are not well-defined
            # on non-finite input: perturbing an already-inf/nan value
            # with small Gaussian noise doesn't measure anything
            # meaningful, and determinism's torch.equal has the same
            # NaN!=NaN problem as above. Skip explicitly rather than
            # force a NaN-aware version of checks whose premise doesn't
            # apply here in the first place.
            out["checks"]["perturbation_tolerance"] = (
                None, "skipped -- not meaningful on non-finite input"
            )
            out["checks"]["determinism"] = (
                None, "skipped -- not meaningful on non-finite input (torch.equal fails on identical NaNs)"
            )

        else:
            finite_ok = torch.isfinite(cand_out).all().item()
            out["checks"]["nan_inf"] = (finite_ok, "finite" if finite_ok else "contains NaN/Inf")
            if not finite_ok:
                out["passed"] = False
                return out

            allclose_ok = torch.allclose(cand_out.float(), ref_out.float(), atol=1e-3, rtol=1e-2)
            max_err = (cand_out.float() - ref_out.float()).abs().max().item()
            out["checks"]["allclose"] = (allclose_ok, f"max_err={max_err:.6f}")
            if not allclose_ok:
                out["passed"] = False

            primary = _first_tensor(args, kwargs)
            if primary is not None:
                def _cand_perturbed(x, _a=args, _k=kwargs):
                    new_args = tuple(x if a is primary else a for a in _a)
                    new_kwargs = {k: (x if v is primary else v) for k, v in _k.items()}
                    return candidate_fn(*new_args, **new_kwargs)

                def _ref_perturbed(x, _a=args, _k=kwargs):
                    new_args = tuple(x if a is primary else a for a in _a)
                    new_kwargs = {k: (x if v is primary else v) for k, v in _k.items()}
                    return reference_fn(*new_args, **new_kwargs)

                try:
                    p_ok, p_detail = check_perturbation_tolerance(_cand_perturbed, _ref_perturbed, primary)
                    out["checks"]["perturbation_tolerance"] = (p_ok, p_detail)
                    if p_ok is False:
                        out["passed"] = False
                except Exception as e:
     
                    out["checks"]["perturbation_tolerance"] = (False, f"check errored: {e}")
                    out["passed"] = False

            if len(args) == 1 and not kwargs:
                try:
                    det_ok, det_detail = check_determinism(lambda x: candidate_fn(x), args[0])
                    out["checks"]["determinism"] = (det_ok, det_detail)
                    if not det_ok:
                        out["passed"] = False
                except Exception as e:
                    out["checks"]["determinism"] = (False, f"check errored: {e}")
            else:
                out["checks"]["determinism"] = (
                    None, "skipped -- multi-arg/keyword signature not retrofitted for determinism check"
                )

        # Optional Layer-3: only fires if this call looks like attention
        # (Q, K, V [,causal]), see attention_batched_properties.py.
        attn_result = try_attention_layer3(candidate_fn, args, kwargs)
        if attn_result is not None:
            out["checks"].update(attn_result)
            if any(v[0] is False for k, v in attn_result.items()):
                out["passed"] = False

        # Optional Layer-3: only fires if this call looks like layernorm
        # (x plus two 1-D tensors matching x's trailing dim), see
        # layernorm_generic_properties.py. Needs reference_fn (unlike the
        # attention check) since it diffs candidate against reference
        # directly at non-identity affine params, rather than checking a
        # self-contained invariant.
        ln_result = try_layernorm_layer3(candidate_fn, reference_fn, args, kwargs)
        if ln_result is not None:
            out["checks"].update(ln_result)
            if any(v[0] is False for k, v in ln_result.items()):
                out["passed"] = False

        # Optional Layer-3: only fires if this call looks like matmul
        # (2+ tensors, A.shape[-1]==B.shape[-2]), see
        # matmul_generic_properties.py. Probes a shape deliberately NOT
        # aligned to common block sizes, since the confirmed blind spot
        # here is boundary-mask bugs invisible on block-aligned test
        # shapes (matmul_leakyrelu.py's own test uses 64x64, exactly
        # divisible by its BLOCK_SIZE=32).
        mm_result = try_matmul_layer3(candidate_fn, reference_fn, args, kwargs)
        if mm_result is not None:
            out["checks"].update(mm_result)
            if any(v[0] is False for k, v in mm_result.items()):
                out["passed"] = False

        # Optional Layer-3: only fires if this call looks like softmax
        # (exactly one tensor argument, >=2D), see
        # softmax_generic_properties.py. Closes the asymmetry where
        # softmax's only confirmed catch depended on one file's own test
        # happening to include an adversarial shape.
        sm_result = try_softmax_layer3(candidate_fn, args, kwargs)
        if sm_result is not None:
            out["checks"].update(sm_result)
            if any(v[0] is False for k, v in sm_result.items()):
                out["passed"] = False

    else:
        out["checks"]["note"] = (
            "candidate/reference did not both return a torch.Tensor "
            f"(cand={type(cand_out).__name__}, ref={type(ref_out).__name__}); "
            "tensor-shaped checks skipped for this call -- only 'executes' verified."
        )

    return out


def summarize(result: AdapterResult) -> str:
    lines = [f"=== {result.file} ==="]
    if result.load_error:
        lines.append(f"  LOAD_ERROR: {result.load_error}")
        return "\n".join(lines)

    lines.append(f"  entry_point: {result.entry_point}")
    lines.append(f"  captured_calls: {result.captured_calls}")
    for rec in result.per_call_checks:
        idx = rec["call_index"]
        status = "PASS" if rec["passed"] else "FAIL"
        lines.append(f"  [call {idx}] {status}")
        for name, val in rec["checks"].items():
            if name == "note":
                lines.append(f"    note: {val}")
                continue
            ok, detail = val
            ok_str = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            lines.append(f"    {ok_str} {name}: {detail}")
    return "\n".join(lines)


import multiprocessing as mp


def _mp_worker(reference_path: str, candidate_src: str, queue: mp.Queue):
    try:
        result = run_candidate_against_reference(reference_path, candidate_src)
        queue.put(("OK", result))
    except Exception as e:
        queue.put(("ERROR", f"{type(e).__name__}: {e}"))


def run_with_timeout(
    reference_path: str,
    candidate_src: str,
    timeout_seconds: int = 30,
) -> AdapterResult:
    """
    Run run_candidate_against_reference in a subprocess with a hard
    timeout. Returns an AdapterResult with load_error='TIMEOUT' if the
    process doesn't finish in time -- never blocks indefinitely.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_mp_worker, args=(reference_path, candidate_src, queue))
    p.start()
    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.kill()
        p.join()
        return AdapterResult(file=reference_path, load_error=f"TIMEOUT after {timeout_seconds}s")

    if queue.empty():
        return AdapterResult(file=reference_path, load_error="subprocess exited with no result (crash?)")

    status, payload = queue.get()
    if status == "ERROR":
        return AdapterResult(file=reference_path, load_error=f"subprocess error: {payload}")
    return payload