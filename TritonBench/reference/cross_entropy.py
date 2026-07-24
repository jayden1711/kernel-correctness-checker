import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(loss_ptr, logits_ptr, targets_ptr,
                          logits_row_stride, n_rows, n_cols,
                          BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = logits_ptr + row_idx * logits_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    log_softmax_row = row_minus_max - tl.log(denominator)

    target = tl.load(targets_ptr + row_idx)
    target_log_prob = tl.sum(tl.where(col_offsets == target, log_softmax_row, 0.0), axis=0)
    loss = -target_log_prob

    tl.store(loss_ptr + row_idx, loss)


def cross_entropy(logits, targets):
    """
    logits: (N, C) raw scores. targets: (N,) int64 class indices.
    Reduction='mean' (PyTorch's CrossEntropyLoss default) -- no class
    weighting, no label_smoothing, no ignore_index. The final mean over
    N per-sample losses is a single trivial reduction over N scalars,
    done on the host rather than as a second Triton kernel.
    """
    n_rows, n_cols = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    per_sample_loss = torch.empty((n_rows,), device=logits.device, dtype=logits.dtype)

    cross_entropy_kernel[(n_rows,)](
        per_sample_loss, logits, targets,
        logits.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
    )
    return per_sample_loss.mean()
