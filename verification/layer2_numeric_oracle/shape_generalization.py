"""
Layer 2 — Shape generalization and backward-pass correctness checks.

Tests that the candidate kernel:
  1. Produces the correct output shape.
  2. Generalises correctly across at least 5 distinct input shapes.
  3. Is not broken by large or structured weight magnitudes.
  4. Has correct gradients (for training-context kernels).
"""

import torch
from typing import Callable


# check_output_shape

def check_output_shape(
    candidate_fn: Callable,
    reference_fn: Callable,
    x: torch.Tensor,
) -> tuple:
    """
    Verify that the candidate's output tensor shape exactly matches the
    reference before any value comparison is attempted.

    Returns:
        (True,  detail)   shapes match
        (False, detail)   shapes differ
    """
    try:
        ref_out = reference_fn(x)
        cand_out = candidate_fn(x)
    except Exception as e:
        return False, f"Exception during shape check: {e}"

    if cand_out.shape != ref_out.shape:
        return False, (
            f"Output shape mismatch: candidate {tuple(cand_out.shape)} "
            f"vs reference {tuple(ref_out.shape)} for input {tuple(x.shape)}."
        )

    return True, f"Output shape {tuple(cand_out.shape)} matches reference."


# check_cross_shape

def check_cross_shape(
    candidate_fn: Callable,
    reference_fn: Callable,
    spec,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple:
    """
    Test the candidate on every shape listed in spec.valid_shapes.

    spec.valid_shapes must include at least:
      - A non-power-of-two dimension
      - A large batch
      - Batch size 1

    This catches hardcoded outputs and tile-boundary bugs.

    Returns:
        (True,  detail)    all shapes pass
        (False, detail)    list of failing shapes
    """
    failures = []

    for shape in spec.valid_shapes:
        # Build an appropriately-shaped input
        try:
            x = torch.randn(*shape, device="cuda" if torch.cuda.is_available()
                            else "cpu", dtype=torch.float32)
            ref_out = reference_fn(x)
            cand_out = candidate_fn(x)
        except Exception as e:
            failures.append(f"shape={shape}: exception  {e}")
            continue

        if cand_out.shape != ref_out.shape:
            failures.append(
                f"shape={shape}: shape mismatch "
                f"{tuple(cand_out.shape)} vs {tuple(ref_out.shape)}"
            )
            continue

        if not torch.allclose(cand_out.float(), ref_out.float(),
                              atol=atol, rtol=rtol):
            max_err = (cand_out.float() - ref_out.float()).abs().max().item()
            failures.append(
                f"shape={shape}: numeric mismatch, max_err={max_err:.6f}"
            )

    if failures:
        return False, "Cross-shape failures: " + "; ".join(failures)

    return True, (
        f"Cross-shape check passed on {len(spec.valid_shapes)} shapes: "
        + str(spec.valid_shapes)
    )


# check_weight_magnitude

def _make_weight_variants(primary: torch.Tensor) -> dict:
    """
    Build adversarial primary-input variants at the same shape/device/dtype
    as the captured primary tensor.

    Each variant targets a specific failure mode:
      large_uniform    — fp16 accumulator overflow (all values identical and huge)
      large_random     — fp16 overflow with variance (exposes rounding instability)
      monotone_rows    — values climb across columns; catches kernels that only
                         process the first tile (output misses the large tail)
      alternating_sign — catastrophic cancellation in reductions (+1, -1, +1, ...)
    """
    shape = primary.shape
    device = primary.device
    dtype = primary.dtype
    n_cols = shape[-1]

    variants = {
        "large_uniform": torch.full(shape, 1e4, device=device, dtype=dtype),
        "large_random": torch.randn(*shape, device=device, dtype=dtype) * 1e4,
    }

    # FIXED: monotone_rows previously always built a (1, n_cols) source
    # tensor via unsqueeze(0) and .expand(*shape) -- valid for 2D+ primaries
    # (softmax, layernorm, etc.) but a guaranteed RuntimeError for 1D
    # primaries (swish, gelu), since expand() requires the target shape to
    # have at least as many dims as the source. Confirmed via a real
    # run_checker.py run: this crashed with "the number of sizes provided
    # (1) must be greater or equal to the number of dimensions in the
    # tensor (2)" for both swish and gelu -- masked there because the
    # exception got caught and recorded as a FAIL, coincidentally matching
    # the expected mutant-catch verdict, but it would crash identically for
    # a CORRECT 1D kernel. For 1D primaries, build the monotone ramp
    # directly at the primary's own shape instead of via a 2D intermediate.
    if primary.dim() >= 2:
        variants["monotone_rows"] = (
            torch.arange(n_cols, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(*shape)
            .contiguous() * 100.0
        )
    else:
        variants["monotone_rows"] = (
            torch.arange(n_cols, device=device, dtype=dtype) * 100.0
        )

    # Fixed: old version used a fragile repeat/slice that broke on odd
    # column counts and non-2D shapes.  Ellipsis handles any leading dims.
    alt = torch.ones(*shape, device=device, dtype=dtype)
    alt[..., 1::2] = -1.0
    variants["alternating_sign"] = alt

    return variants


def check_weight_magnitude(
    candidate_fn: Callable,
    reference_fn: Callable,
    spec,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> tuple:
    """
    Test with adversarially large / structured primary-input values.

    Uses spec.make_inputs so gamma/beta/B (non-primary tensors) are
    correctly shaped for each variant — the old version called
    candidate_fn(x) directly, which broke for multi-input operators
    whose non-primary args had shape constraints tied to x.

    Returns:
        (True,  detail)    all weight-magnitude variants pass
        (False, detail)    list of failing variants
    """
    shape = spec.valid_shapes[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    failures = []

    base_inputs = spec.make_inputs(shape, device, dtype)
    primary = spec.primary_input(base_inputs)
    variants = _make_weight_variants(primary)

    for variant_name, adv_primary in variants.items():
        try:
            if isinstance(base_inputs, tuple):
                adv_inputs = (adv_primary,) + base_inputs[1:]
            else:
                adv_inputs = adv_primary

            ref_out = spec.run_reference(reference_fn, adv_inputs)
            cand_out = spec.run_candidate(candidate_fn, adv_inputs)
        except Exception as e:
            failures.append(f"{variant_name}: exception — {e}")
            continue

        if cand_out.shape != ref_out.shape:
            failures.append(f"{variant_name}: shape mismatch")
            continue

        if not torch.allclose(cand_out.float(), ref_out.float(),
                              atol=atol, rtol=rtol):
            max_err = (cand_out.float() - ref_out.float()).abs().max().item()
            failures.append(
                f"{variant_name}: numeric mismatch, max_err={max_err:.6f}"
            )

    if failures:
        return False, "Weight-magnitude failures: " + "; ".join(failures)

    return True, "Weight-magnitude check passed on all variants."


# check_backward_pass

def check_backward_pass(
    candidate_fn: Callable,
    reference_fn: Callable,
    x: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple:
    """
    Verify gradient correctness by comparing autograd gradients of the
    candidate against the reference.

    Both functions must be differentiable (autograd-compatible).  If the
    candidate doesn't support autograd this check returns False.

    Returns:
        (True,  detail)    gradients match
        (False, detail)    gradient mismatch or autograd failure
    """
    device = x.device

    def _run_backward(fn, inp):
        inp_clone = inp.detach().clone().requires_grad_(True)
        out = fn(inp_clone)
        # Use a random upstream gradient to avoid masking errors
        upstream = torch.randn_like(out)
        out.backward(upstream)
        return inp_clone.grad

    try:
        ref_grad = _run_backward(reference_fn, x)
    except Exception as e:
        # If reference doesn't support backward, skip
        return True, f"Reference does not support backward; skipping. ({e})"

    try:
        cand_grad = _run_backward(candidate_fn, x)
    except Exception as e:
        return False, f"Candidate backward pass raised an exception: {e}"

    if cand_grad is None:
        return False, "Candidate backward pass produced None gradient."

    if cand_grad.shape != ref_grad.shape:
        return False, (
            f"Gradient shape mismatch: candidate {tuple(cand_grad.shape)} "
            f"vs reference {tuple(ref_grad.shape)}."
        )

    if not torch.allclose(cand_grad.float(), ref_grad.float(),
                          atol=atol, rtol=rtol):
        max_err = (cand_grad.float() - ref_grad.float()).abs().max().item()
        return False, (
            f"Gradient mismatch: max_err={max_err:.6f} "
            f"(atol={atol}, rtol={rtol})."
        )

    return True, f"Backward-pass check passed. Gradient max_err within tolerance."
