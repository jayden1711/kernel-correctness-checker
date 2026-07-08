"""
Layer 1  Runtime tile-coverage check.

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
        (True,  -1,       n_cols)           all columns written
        (False, first_bad_row, n_visited)   first row with missing columns
    """
    if kernel_fn is None:
        return True, -1, -1

    if x.dim() != 2:
        return False, -1, -1

    n_rows, n_cols = x.shape

    try:
        y = kernel_fn(x)
    except Exception as e:
        return False, -1, -1

    if y.shape != x.shape:
        return False, -1, -1

    # For softmax, all output values must be positive since exp() > 0 always.
    # Zero columns indicate that the kernel skipped those tiles entirely.
    cols_written = (y > 0).any(dim=0).sum().item()

    if cols_written < n_cols:
        # Find the first row with missing columns
        for row_idx in range(n_rows):
            row_cols = (y[row_idx] > 0).sum().item()
            if row_cols < n_cols:
                return False, f"Row {row_idx} only has {int(row_cols)}/{n_cols} columns written  partial tile coverage detected.", int(row_cols)

    return True, -1, n_cols

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
    """
    import torch as _torch

    orig_empty_like = _torch.empty_like
    orig_empty = _torch.empty
    orig_zeros_like = _torch.zeros_like
    orig_zeros = _torch.zeros

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

    def _patched_zeros_like(*args, **kwargs):
        t = orig_zeros_like(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(sentinel)
        return t

    def _patched_zeros(*args, **kwargs):
        t = orig_zeros(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(sentinel)
        return t

    _torch.empty_like = _patched_empty_like
    _torch.empty = _patched_empty
    _torch.zeros_like = _patched_zeros_like
    _torch.zeros = _patched_zeros
    try:
        y = spec.run_candidate(candidate_fn, inputs)
    except Exception as e:
        return False, f"Kernel raised during tile-coverage check: {e}"
    finally:
        _torch.empty_like = orig_empty_like
        _torch.empty = orig_empty
        _torch.zeros_like = orig_zeros_like
        _torch.zeros = orig_zeros

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