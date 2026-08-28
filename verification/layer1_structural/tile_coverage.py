"""
Layer 1 — Runtime tile-coverage check.

Instruments the kernel wrapper and asserts that every output column
has been written, catching first-tile-only and other partial-computation
patterns.

Rather than relying on triton-viz memory tracing (which cannot distinguish
masked from unmasked loads), we run the kernel and inspect the output
directly. For softmax, all output values must be positive since exp() > 0
always. Zero columns indicate unprocessed tiles.

This approach requires no special instrumentation or modified Triton builds.
"""
import torch


def check_all_tiles_visited(
    kernel_fn,
    raw_kernel,
    x: torch.Tensor,
    block_size: int = None,
) -> tuple:
    """
    Check that the kernel writes to all output columns for every row.

    Runs the kernel wrapper and checks that no output columns are zero,
    which would indicate that tiles were skipped.

    Args:
        kernel_fn:   Python wrapper that calls the Triton kernel.
        raw_kernel:  The @triton.jit function (unused, kept for API compatibility).
        x:           A 2-D input tensor (n_rows x n_cols).
        block_size:  Unused, kept for API compatibility.

    Returns:
        (passed: bool, detail: str | None)

    This is a two-element return by design, matching
    `check_all_tiles_visited_generic` below and every other Layer-1 check.
    It previously returned a THIRD element carrying a column count, which
    collided with the checker-adapter convention where a 3rd element means
    "per-sub-check records for a compound check" (a list). Only two of the
    322 records in a full corpus run ever populated it -- the partial-
    coverage branch below -- and those two crashed
    benchmarks/analyze_check_ablation.py with
    `TypeError: 'int' object is not iterable`. The column count was never
    read by any caller (verification/checker.py's _run_check consumes slots
    0 and 1 only); it is preserved where it was always actually useful, in
    the detail string. Do not reintroduce a 3rd element here unless this
    check genuinely becomes compound.

    The `passed` values below are deliberately unchanged from the previous
    contract, so catch_rate / false_positive_rate stay comparable to earlier
    runs; only the shape of the reported detail changed.
    """
    if kernel_fn is None:
        return True, None

    if x.dim() != 2:
        return False, f"Expected a 2-D input, got {x.dim()}-D; cannot check tile coverage."

    n_rows, n_cols = x.shape

    try:
        y = kernel_fn(x)
    except Exception as e:
        return False, f"Kernel raised during tile-coverage check: {type(e).__name__}: {e}"

    if y.shape != x.shape:
        return False, f"Kernel output shape {tuple(y.shape)} != input shape {tuple(x.shape)}; cannot check tile coverage."

    # For softmax, all output values must be positive since exp() > 0 always.
    # Zero columns indicate that the kernel skipped those tiles entirely.
    cols_written = (y > 0).any(dim=0).sum().item()

    if cols_written < n_cols:
        # Find the first row with missing columns
        for row_idx in range(n_rows):
            row_cols = (y[row_idx] > 0).sum().item()
            if row_cols < n_cols:
                return False, f"Row {row_idx} only has {int(row_cols)}/{n_cols} columns written — partial tile coverage detected."

    return True, None

def check_all_tiles_visited_generic(spec, candidate_fn, inputs, sentinel: float = float('nan')) -> tuple:
    """
    Operator-agnostic replacement for the softmax-only positivity check.

    Args:
        spec:          The KernelSpec, used to call run_candidate with
                       the full multi-tensor inputs correctly.
        candidate_fn:  The candidate kernel wrapper.
        inputs:        Tensor or tuple, as passed to checker.run().
        sentinel:      Fill value for freshly allocated float buffers.
                       NaN by default since it's easy to detect and
                       vanishingly unlikely to be a legitimate kernel
                       output value.

    Returns:
        (True,  detail)   every output element was overwritten
        (False, detail)   at least one element still holds the sentinel

    FIXED: previously also patched torch.zeros/torch.zeros_like, not just
    torch.empty/torch.empty_like. zeros() exists specifically to guarantee
    a real, meaningful initial value (0) -- patching it to NaN instead
    breaks any kernel that legitimately depends on that guarantee, which
    is exactly what an atomic-add accumulator does (frobenius_norm's
    sum-of-squares buffer: torch.zeros(1, ...) then tl.atomic_add into
    it). NaN + anything = NaN, so the accumulator stays NaN through the
    whole reduction regardless of correctness -- this broke the
    REFERENCE kernel too, not just the mutant, and had nothing to do
    with tile coverage. empty()/empty_like() have genuinely undefined
    initial content by design, so replacing "undefined" with "a
    detectable sentinel" is safe and doesn't change kernel correctness --
    confirmed sufficient on its own for the demonstrated catches (e.g.
    softmax's reference allocates its output via torch.empty_like(x)).
    """
    import torch as _torch

    orig_empty_like = _torch.empty_like
    orig_empty = _torch.empty

    def _patched_empty_like(*args, **kwargs):
        t = orig_empty_like(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(sentinel)
        return t

    def _patched_empty(*args, **kwargs):
        t = orig_empty(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(sentinel)
        return t

    _torch.empty_like = _patched_empty_like
    _torch.empty = _patched_empty
    try:
        y = spec.run_candidate(candidate_fn, inputs)
    except Exception as e:
        return False, f"Kernel raised during tile-coverage check: {e}"
    finally:
        _torch.empty_like = orig_empty_like
        _torch.empty = orig_empty

    if y is None or not _torch.is_tensor(y):
        return False, f"Kernel did not return a tensor (got {type(y).__name__}); cannot check tile coverage."

    if not y.is_floating_point():
        return True, "Output is not floating-point; tile-coverage sentinel check not applicable, skipping."

    unwritten = _torch.isnan(y)
    n_unwritten = unwritten.sum().item()
    if n_unwritten > 0:
        return False, (
            f"{n_unwritten}/{y.numel()} output element(s) never written "
            f"(NaN sentinel survived) -- partial tile/lane coverage detected."
        )
    return True, f"All {y.numel()} output element(s) written (no sentinel survived)."
