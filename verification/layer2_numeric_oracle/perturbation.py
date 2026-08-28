"""
Layer 2 — Perturbation-based adaptive tolerance check.

Rather than using a fixed atol, we measure how sensitive the *reference*
implementation is to small input perturbations and use a quantile of that
empirical distribution as the allowed error band. A cheating kernel that
omits work will exceed this tighter bound even if it would squeak under a
hand-tuned fixed tolerance.
"""

import os
import torch
from typing import Callable


# ---------------------------------------------------------------------------
# OPT-IN INSTRUMENTATION (2026-08-25). Both default OFF; the shipped default
# of n_samples=20 is UNCHANGED.
#
#   KCC_N_SAMPLES=<int>       override n_samples for every call that does not
#                             pass one explicitly (ablation arms only)
#   KCC_RECORD_SENSITIVITIES=1
#                             return the raw per-sample sensitivity vector and
#                             max_err as an optional THIRD return element
#
# WHY THE SENSITIVITY VECTOR IS WORTH RECORDING. The verdict is
#
#     fail  <=>  max_err > scale * quantile(sensitivities[:n], quantile)
#
# `max_err` does not depend on n at all, and the deltas are drawn ONE AT A TIME
# in a fixed order from a per-check seed. So the n-sample run's sensitivity
# vector is a PREFIX of the (n+k)-sample run's. Recording the full vector once
# at a high n therefore determines the verdict at EVERY smaller n exactly,
# offline, with no cross-arm RNG noise and no extra GPU time -- rather than
# inferring a curve from a handful of separately-seeded arms.
# ---------------------------------------------------------------------------
_N_OVERRIDE = os.environ.get("KCC_N_SAMPLES")
_RECORD_SENS = os.environ.get("KCC_RECORD_SENSITIVITIES") == "1"

# ---------------------------------------------------------------------------
# OPT-IN ALTERNATE ESTIMATOR (2026-08-26). Default OFF.
#
#   KCC_STRUCTURAL_L=1   compute adaptive_tol from the operator's closed-form
#                        Jacobian instead of probing the reference n_samples
#                        times. See structural_l.py for what is and is not
#                        closed form here -- L is; adaptive_tol is a
#                        simulation over L's row-norm profile, not a formula.
#
# Wired as a REPLACEMENT for the sensitivity loop only. Everything downstream
# of `adaptive_tol` -- the candidate call, the shape check, the finite check,
# max_err, the verdict, the message -- is untouched and shared by both paths,
# so an arm difference can only ever come from the tolerance itself.
# ---------------------------------------------------------------------------
from verification.layer2_numeric_oracle import structural_l as _struct
_STRUCTURAL = _struct._STRUCTURAL

# ---------------------------------------------------------------------------
# OPT-IN SCOPE-DIVERGENCE DETECTOR (2026-08-26). Default OFF.
#
#   KCC_SCOPE_DETECT=1   annotate invocations whose perturbation response is
#                        outside the regime the bound was verified in.
#
# PURELY ADDITIVE BY CONSTRUCTION. The record is appended to the third return
# element and is never read on the path that produces `passed` -- see
# scope_detect.py, "Why annotate and not act".
# ---------------------------------------------------------------------------
from verification.layer2_numeric_oracle import scope_detect as _scope


def check_perturbation_tolerance(
    candidate_fn: Callable,
    reference_fn: Callable,
    x: torch.Tensor,
    n_samples: int = 20,
    quantile: float = 0.95,
    scale: float = 3.0,
    delta_scale: float = 1e-3,
    batch_samples: bool = False,
    op_name: str = None,
    companions=(),
) -> tuple:
    """`op_name`/`companions` are consumed ONLY by the KCC_STRUCTURAL_L path.

    Both default to inert values, so every existing call site keeps the
    Monte-Carlo behaviour bit-for-bit whether or not it passes them.
    `companions` are the non-primary inputs the closed forms need (gamma, B,
    targets, K/V); the checker supplies the tuple that matches the tensor
    actually being fed to the kernel for this call, adversarial or not.
    """
    if _N_OVERRIDE is not None:
        n_samples = int(_N_OVERRIDE)

    x = x.detach().clone()
    ref_base = reference_fn(x)

    if x.numel() < 2:
   
        return None, "skipped -- perturbation tolerance undefined for single-element input"

    if not torch.is_floating_point(x):
        return None, f"skipped -- perturbation tolerance not meaningful for non-floating-point input (dtype={x.dtype})"

    x_std = x.float().std().item()
    if x_std == 0:
        x_std = 1.0

    # --- KCC_STRUCTURAL_L branch point -------------------------------------
    # Taken BEFORE the deltas are drawn, because the whole point is to avoid
    # drawing them. `structural_adaptive_tol` returns None for any operator or
    # configuration it was not derived for (argmax/argmin, missing companions,
    # non-q95), and None falls straight through to the probe below. Declining
    # is the safe direction and is the only way this path can fail closed.
    _struct_tol = None
    if _STRUCTURAL and op_name is not None:
        _struct_tol = _struct.structural_adaptive_tol(
            op_name, x, companions, n_samples, quantile, scale, delta_scale)

    if _struct_tol is not None:
        # No deltas, no reference launches, no device transfer, no quantile.
        # This is the entire saving, and it is also the entire risk surface:
        # the tolerance now comes from a model of the reference rather than
        # from the reference itself.
        sensitivities_t = None
        adaptive_tol = _struct_tol
        # No deltas were drawn, so the measured screens have no input. The
        # detector still reports the structural exclusion and the tolerance
        # floor, which need no measurement.
        _deltas = []
    else:
        adaptive_tol, sensitivities_t, _deltas = _probe_adaptive_tol_and_sens(
            reference_fn, x, ref_base, n_samples, quantile, scale,
            delta_scale, x_std, batch_samples)

    # Costs KCC_SCOPE_GRAM_SAMPLES float64 CPU JVPs of the math definition
    # (math_refs.py) -- NO extra reference launches, NO new RNG -- and nothing
    # at all when the flag is off. `companions` ride along because the Gram
    # screen evaluates the math function with the same operands the kernel saw.
    _scope_rec = _scope.build_record(sensitivities_t, ref_base, reference_fn, x,
                                     _deltas, adaptive_tol, op_name=op_name,
                                     companions=companions)

    return _finish(candidate_fn, x, ref_base, adaptive_tol, sensitivities_t,
                   n_samples, quantile, scale, _scope_rec)


def _probe_adaptive_tol_and_sens(reference_fn, x, ref_base, n_samples, quantile,
                                 scale, delta_scale, x_std, batch_samples):
    """The Monte-Carlo estimator, moved into a helper VERBATIM.

    Extracted only so the structural branch above can skip it as a unit. Not a
    line of it changed: same draw order, same batching gate, same CPU quantile,
    same float32 cast, same 1e-6 floor. The extraction is behaviour-preserving
    by construction, which is what lets the two arms differ solely in `tol`.
    """
    # STAGE A (2026-08-21): the per-sample `.item()` was removed from this loop.
    #
    # `.item()` forces a GPU->CPU sync on EVERY iteration, so the 20 launches
    # could never overlap -- the loop stalled the pipeline 20 times per call.
    # This function is called ~159 times per 40-mutant pass (40
    # perturbation_tolerance + 119 adversarial_*, which route through here), so
    # a full benchmark was paying roughly 19,000 launch+sync pairs. Measured
    # cost of those checks beforehand: 4.22ms median, ~0.21ms per sample for a
    # kernel whose actual work is microseconds -- overhead, not computation.
    #
    # Diffs now accumulate as 0-d device tensors and are transferred ONCE.
    #
    # BIT-IDENTICAL, and the dtype path is why: previously each diff went
    # tensor -> Python float (float64) -> float32 via torch.tensor(...). Now it
    # goes tensor -> float32 directly. For float16/float32 inputs the old
    # round-trip through float64 was exact, and for float64 inputs both paths
    # round to float32 identically, so the resulting quantile input is the same
    # bits either way.
    #
    # The quantile stays on CPU deliberately -- torch.quantile on device can
    # differ in the last ULP from the CPU implementation, which would change
    # adaptive_tol and, at the margin, a verdict. The `.to("cpu")` below is the
    # single sync that replaces the 20.
    # STAGE B: the deltas are ALWAYS drawn one at a time, in this order, even
    # when batching. This is not stylistic -- torch's normal generator does not
    # produce the same values for one large draw as for many small ones:
    #
    #     randn(20,*s) == stack([randn_like(x) x20])  ->  False (max diff 5.57)
    #
    # Drawing them batched would change every delta, every sensitivity, and
    # adaptive_tol with them, so a marginal candidate could flip verdict for no
    # reason anyone would ever trace. Batch the KERNEL CALL, never the RNG.
    deltas = [torch.randn_like(x) * delta_scale * x_std for _ in range(n_samples)]

    if batch_samples and n_samples > 1:
        # Stack to (n_samples, *x.shape) then fold into (n_samples*R, ...) so
        # the kernel sees an ordinary larger input at its usual rank -- no
        # kernel change, no rank change. Only valid when dim 0 carries
        # independent samples and no companion tensor is per-sample; that is
        # what KernelSpec.batch_samples gates (see its docstring for the four
        # operators excluded and why).
        stacked = torch.cat([(x + d).unsqueeze(0) for d in deltas], dim=0)
        flat = stacked.reshape(n_samples * x.shape[0], *x.shape[1:])
        out = reference_fn(flat)
        out = out.reshape(n_samples, *ref_base.shape)
        diffs = (out - ref_base.unsqueeze(0)).abs().reshape(n_samples, -1).max(dim=1).values
        sensitivities_t = diffs.to(device="cpu", dtype=torch.float32)
    else:
        sensitivities = []
        for delta in deltas:
            ref_perturbed = reference_fn(x + delta)
            sensitivities.append((ref_perturbed - ref_base).abs().max())
        sensitivities_t = torch.stack(sensitivities).to(device="cpu",
                                                        dtype=torch.float32)
    adaptive_tol = scale * torch.quantile(sensitivities_t, quantile).item()
    adaptive_tol = max(adaptive_tol, 1e-6)
    # `deltas` travel back out so the scope detector can reuse the SAME
    # perturbations the tolerance was built from. Re-drawing them would both
    # cost extra RNG and, worse, shift the stream for every later check --
    # the defect class the KCC_ABLATION_SEED note in checker.py exists for.
    return adaptive_tol, sensitivities_t, deltas


def _finish(candidate_fn, x, ref_base, adaptive_tol, sensitivities_t,
            n_samples, quantile, scale, scope_rec=None):
    """Everything downstream of the tolerance -- IDENTICAL for both estimators.

    Kept in one place deliberately. If the candidate call, the shape/finite
    guards or the max_err comparison were duplicated per branch, an arm
    difference could come from a divergence here rather than from the
    tolerance, and the ablation would be measuring the wrong thing.
    """
    try:
        candidate_out = candidate_fn(x)
    except Exception as e:
        return False, f"Candidate raised an exception: {e}"

    if candidate_out.shape != ref_base.shape:
        return False, (
            f"Output shape mismatch: candidate {tuple(candidate_out.shape)} "
            f"vs reference {tuple(ref_base.shape)}."
        )

    if not torch.isfinite(candidate_out).all():
        n_nan = torch.isnan(candidate_out).sum().item()
        n_inf = torch.isinf(candidate_out).sum().item()
        return False, (
            f"Candidate output contains non-finite values: "
            f"{n_nan} NaN, {n_inf} Inf."
        )

    max_err = (candidate_out.float() - ref_base.float()).abs().max().item()

    # Additive third element; present only under KCC_RECORD_SENSITIVITIES=1,
    # so the default return stays a 2-tuple exactly as before.
    _sens_rec = None
    if scope_rec is not None:
        _sens_rec = [scope_rec]
    if _RECORD_SENS and sensitivities_t is not None:
        _sens_rec = (_sens_rec or []) + [{"kind": "perturbation_sensitivities",
                      "n_samples": n_samples,
                      "sensitivities": [float(v) for v in sensitivities_t.tolist()],
                      "max_err": float(max_err),
                      "adaptive_tol": float(adaptive_tol),
                      "scale": float(scale), "quantile": float(quantile)}]

    if max_err > adaptive_tol:
        # The parenthetical names the ESTIMATOR, so a log line is never
        # ambiguous about which path produced the band it reports.
        if sensitivities_t is None:
            _how = "closed-form Jacobian, KCC_STRUCTURAL_L"
        else:
            _how = (f"scale={scale}xP{int(quantile*100)} of reference sensitivity "
                    f"{torch.quantile(sensitivities_t, quantile).item():.6f}")
        return (False, (
            f"Candidate exceeds adaptive tolerance. "
            f"max_err={max_err:.6f}, adaptive_tol={adaptive_tol:.6f} "
            f"({_how})."
        )) + ((_sens_rec,) if _sens_rec else ())
    return (True, (
        f"Perturbation check passed. "
        f"max_err={max_err:.6f} <= adaptive_tol={adaptive_tol:.6f}."
    )) + ((_sens_rec,) if _sens_rec else ())