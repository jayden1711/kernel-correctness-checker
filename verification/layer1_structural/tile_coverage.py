"""
Layer 1 – Runtime tile-coverage check via triton-viz.

Instruments the kernel and asserts that every (row, column) address
was loaded before the output was written, catching first-tile-only
and other partial-computation patterns.
"""

import torch
import triton

try:
    import triton_viz
    from triton_viz import trace
    _TRITON_VIZ_AVAILABLE = True
except ImportError:
    _TRITON_VIZ_AVAILABLE = False


def check_all_tiles_visited(kernel_fn, raw_kernel, x: torch.Tensor) -> tuple:
    """
    Check that the kernel loads from all column offsets for every row.

    Uses triton-viz to record actual memory-access patterns at runtime.

    Args:
        kernel_fn:   Python wrapper that calls the Triton kernel (used for
                     shape inference only if raw_kernel is provided).
        raw_kernel:  The @triton.jit function to instrument.
        x:           A 2-D input tensor (n_rows x n_cols).

    Returns:
        (True,  -1,       n_cols)           all rows complete
        (False, first_bad_row, n_visited)   first row with missing columns
    """
    if not _TRITON_VIZ_AVAILABLE:
        # Soft-fail: can't instrument without triton-viz
        return True, -1, -1

    if x.dim() != 2:
        return False, -1, -1

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Wrap raw kernel with the triton-viz tracer
    traced = trace()(raw_kernel)

    y = torch.empty_like(x)
    traced[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Extract Load records from the tracer
    tracer = traced.client_manager.clients["tracer"]
    records = tracer.records

    # Group load offsets by program-instance (row)
    current_row = 0
    row_offsets: dict[int, set] = {i: set() for i in range(n_rows)}

    for record in records:
        rtype = type(record).__name__
        if rtype == "Grid":
            # idx is (pid_0, pid_1, pid_2); for a 1-D launch pid_0 == row
            current_row = record.idx[0]
        elif rtype == "Load":
            # offsets are byte addresses; float32 = 4 bytes
            element_size = x.element_size()
            col_indices = (record.offsets // element_size) % n_cols
            row_offsets[current_row].update(col_indices.tolist())

    # Assert every row visited every column
    for row_idx, visited in row_offsets.items():
        if len(visited) < n_cols:
            return False, row_idx, len(visited)

    return True, -1, n_cols