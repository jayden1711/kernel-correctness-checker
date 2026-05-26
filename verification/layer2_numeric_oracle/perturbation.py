"""
Layer 2 – Perturbation-based adaptive tolerance check.

Rather than using a fixed atol, we measure how sensitive the *reference*
implementation is to small input perturbations and use a quantile of that
empirical distribution as the allowed error band.  A cheating kernel that
omits work will exceed this tighter bound even if it would squeak under a
hand-tuned fixed tolerance.
"""

import torch
from typing import Callable


def check_perturbation_tolerance(
    candidate_fn: Callable,
    reference_fn: Callable,
    x: torch.Tensor,
    n_samples: int = 20,
    quantile: float = 0.95,
    scale: float = 3.0,
    delta_scale: float = 1e-3,
) -> tuple:
    """
    Build an empirical sensitivity distribution from the reference, then
    assert the candidate stays within `scale` x the `quantile`-th percentile
    of that distribution.

    Args:
        candidate_fn:  Kernel under test, signature f(x) -> Tensor.
        reference_fn:  Ground-truth implementation.
        x:             Input tensor.
        n_samples:     Number of perturbation samples to draw.
        quantile:      Which percentile of the sensitivity distribution to use
                       as the tolerance baseline (default 95th).
        scale:         Multiplier on top of the empirical quantile.
                       scale=3 means the candidate may be up to 3x noisier
                       than the reference's natural sensitivity.
        delta_scale:   Size of perturbations relative to x's std-dev.

    Returns:
        (True,  detail_str)   candidate within adaptive tolerance
        (False, detail_str)   candidate exceeds adaptive tolerance
    """
    x = x.detach().clone()
    ref_base = reference_fn(x)

    # --- Step 1: Build empirical sensitivity of the reference --------------
    x_std = x.float().std().item()
    if x_std == 0:
        x_std = 1.0  # avoid degenerate case

    sensitivities = []
    for _ in range(n_samples):
        delta = torch.randn_like(x) * delta_scale * x_std
        ref_perturbed = reference_fn(x + delta)
        diff = (ref_perturbed - ref_base).abs().max().item()
        sensitivities.append(diff)

    sensitivities_t = torch.tensor(sensitivities, dtype=torch.float32)
    adaptive_tol = scale * torch.quantile(sensitivities_t, quantile).item()

    # Guard against a completely flat reference (e.g. constant output)
    adaptive_tol = max(adaptive_tol, 1e-6)

    # --- Step 2: Compare candidate against reference -----------------------
    try:
        candidate_out = candidate_fn(x)
    except Exception as e:
        return False, f"Candidate raised an exception: {e}"

    if candidate_out.shape != ref_base.shape:
        return False, (
            f"Output shape mismatch: candidate {tuple(candidate_out.shape)} "
            f"vs reference {tuple(ref_base.shape)}."
        )

    max_err = (candidate_out.float() - ref_base.float()).abs().max().item()

    if max_err > adaptive_tol:
        return False, (
            f"Candidate exceeds adaptive tolerance. "
            f"max_err={max_err:.6f}, adaptive_tol={adaptive_tol:.6f} "
            f"(scale={scale}xP{int(quantile*100)} of reference sensitivity "
            f"{torch.quantile(sensitivities_t, quantile).item():.6f})."
        )

    return True, (
        f"Perturbation check passed. "
        f"max_err={max_err:.6f} <= adaptive_tol={adaptive_tol:.6f}."
    )