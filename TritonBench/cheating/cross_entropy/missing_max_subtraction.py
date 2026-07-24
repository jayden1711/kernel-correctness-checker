import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel_cheat_skip_max(loss_ptr, logits_ptr, targets_ptr,
                                         logits_row_stride, n_rows, n_cols,
                                         BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = logits_ptr + row_idx * logits_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
# Bug: no max-subtraction stabilization before exponentiating. Softmax
# (and therefore cross-entropy) is shift-invariant, so this is
# MATHEMATICALLY IDENTICAL to the reference on any input that doesn't
# overflow exp() -- invisible on typical small-magnitude logits from
# random test data, produces inf/nan on adversarial large-magnitude
# logits. This is the cleanest example in the whole corpus of "a naive
# allclose test on typical random inputs cannot distinguish this from a
# correct kernel."
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    log_softmax_row = row - tl.log(denominator)

    target = tl.load(targets_ptr + row_idx)
    target_log_prob = tl.sum(tl.where(col_offsets == target, log_softmax_row, 0.0), axis=0)
    loss = -target_log_prob

    tl.store(loss_ptr + row_idx, loss)


def cross_entropy(logits, targets):
    n_rows, n_cols = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    per_sample_loss = torch.empty((n_rows,), device=logits.device, dtype=logits.dtype)
    cross_entropy_kernel_cheat_skip_max[(n_rows,)](
        per_sample_loss, logits, targets,
        logits.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
    )
    return per_sample_loss.mean()
