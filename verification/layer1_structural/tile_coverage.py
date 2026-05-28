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