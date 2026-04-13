import torch
import triton_viz
from triton_viz import trace

def check_all_tiles_visited(kernel_fn, raw_kernel, x):
    """
    Check that the kernel loads from all column offsets for every row.
    Uses triton-viz to record actual memory access patterns at runtime.
    """
    n_rows, n_cols = x.shape
    
    # Step 1: wrap the raw kernel with triton-viz tracer
    traced = trace()(raw_kernel)
    
    # Step 2: run it — triton-viz records every Load during execution
    BLOCK_SIZE = triton_viz.core.patch  # we get this from the launcher
    y = torch.empty_like(x)
    import triton
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    traced[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 3: extract all Load records from the tracer
    tracer = traced.client_manager.clients['tracer']
    records = tracer.records
    
    # Step 4: group load offsets by row (Grid index tells us which row)
    current_row = 0
    row_offsets = {i: set() for i in range(n_rows)}
    
    for record in records:
        if type(record).__name__ == 'Grid':
            current_row = record.idx[0]  # idx is (row, 0, 0)
        elif type(record).__name__ == 'Load':
            # offsets are absolute byte addresses, convert to column indices
            # divide by 4 because float32 = 4 bytes
            col_indices = record.offsets // 4 % n_cols
            row_offsets[current_row].update(col_indices.tolist())
    
    # Step 5: assert every row visited all n_cols columns
    for row_idx, visited in row_offsets.items():
        if len(visited) < n_cols:
            return False, row_idx, len(visited)
    
    return True, -1, n_cols