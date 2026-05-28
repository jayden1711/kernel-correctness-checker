"""
Layer 1  Structural checks via static AST analysis.

All functions accept a raw kernel source string or callable and return
(passed: bool, detail: str).
"""

import ast
import inspect
import textwrap
from typing import Union


def _get_source(kernel) -> str:
    if isinstance(kernel, str):
        return textwrap.dedent(kernel)
    # Unwrap triton.jit decorated functions
    fn = getattr(kernel, 'fn', kernel)
    src = inspect.getsource(fn)
    return textwrap.dedent(src)


# 1. Ghost-optimization detector

class _GhostOptimizationVisitor(ast.NodeVisitor):
    """
    Detects two ghost-optimization patterns:

    Pattern A  Conditional kernel bypass:
        if <condition>:
            return reference(x)     # kernel never runs
        custom_kernel[grid](...)

    Pattern B  Dead-branch kernel call:
        if False:
            custom_kernel[grid](...)

    Heuristic: if every call to a triton kernel launch (detected by
    subscript-then-call syntax  `fn[grid](...)`) is inside an `if` whose
    test is a constant False/0, flag it.  Also flag if the function body
    contains no kernel-launch at all (pure delegation).
    """

    def __init__(self):
        self.kernel_launches = []        # list of (lineno, in_dead_branch)
        self._dead_branch = False

    def visit_If(self, node: ast.If):
        # Detect statically-false branches: `if False:` / `if 0:`
        is_dead = isinstance(node.test, ast.Constant) and not node.test.value
        old = self._dead_branch
        if is_dead:
            self._dead_branch = True
        self.generic_visit(node)
        self._dead_branch = old

    def visit_Call(self, node: ast.Call):
        """
        A Triton kernel launch looks like:  kernel[grid](*args)
        In the AST that is:  Call(func=Subscript(value=Name(...)))
        """
        if isinstance(node.func, ast.Subscript):
            self.kernel_launches.append(
                (node.lineno, self._dead_branch)
            )
        self.generic_visit(node)


def check_ghost_optimization(kernel) -> tuple:
    """
    Return (True, detail) if the kernel actually launches at runtime for
    all inputs, (False, detail) if a ghost-optimization pattern is found.
    """
    try:
        src = _get_source(kernel)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError) as e:
        return False, f"Could not parse source: {e}"

    visitor = _GhostOptimizationVisitor()
    visitor.visit(tree)

    launches = visitor.kernel_launches
    if not launches:
        return False, (
            "No Triton kernel launch detected. "
            "Entry point may delegate entirely to a reference implementation."
        )

    dead = [ln for ln, dead in launches if dead]
    if dead:
        return False, (
            f"Kernel launch(es) at line(s) {dead} are inside statically-dead "
            "branches (e.g. `if False:`). Kernel never executes."
        )

    return True, f"Found {len(launches)} kernel launch(es); none in dead branches."


# 2. Missing-barrier detector

class _BarrierVisitor(ast.NodeVisitor):
    """
    Flags kernels that write to a shared-memory buffer (identified by
    tl.store / assignment to a tensor slice in a loop) but never call
    tl.barrier() / tl.debug_barrier().

    This is necessarily heuristic: we look for tl.store + absence of
    tl.barrier within the same function body.
    """

    def __init__(self):
        self.has_shared_store = False
        self.has_barrier = False
        self.has_reduction = False

    def _is_tl_call(self, node: ast.Call, name: str) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr == name
        if isinstance(func, ast.Name):
            return func.id == name
        return False

    def visit_Call(self, node: ast.Call):
        if self._is_tl_call(node, "store"):
            self.has_shared_store = True
        if self._is_tl_call(node, "barrier") or self._is_tl_call(node, "debug_barrier"):
            self.has_barrier = True
        # Detect reduction patterns: tl.sum, tl.max, tl.min
        if self._is_tl_call(node, "sum") or self._is_tl_call(node, "max") or \
                self._is_tl_call(node, "min"):
            self.has_reduction = True
        self.generic_visit(node)


def check_missing_barriers(kernel) -> tuple:
    """
    Return (True, detail) if barriers appear to be correctly placed,
    (False, detail) if a reduction or shared store exists without a barrier.

    Note: flash_attention and other multi-stage kernels require barriers;
    simple element-wise kernels do not.  We only flag when we can detect
    the need.
    """
    try:
        src = _get_source(kernel)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError) as e:
        return False, f"Could not parse source: {e}"

    visitor = _BarrierVisitor()
    visitor.visit(tree)

    needs_barrier = visitor.has_reduction and visitor.has_shared_store
    if needs_barrier and not visitor.has_barrier:
        return False, (
            "Kernel performs a reduction and writes to shared memory but "
            "contains no tl.barrier() call. Race conditions likely."
        )

    return True, (
        "Barrier check passed "
        f"(has_reduction={visitor.has_reduction}, "
        f"has_store={visitor.has_shared_store}, "
        f"has_barrier={visitor.has_barrier})."
    )


# 3. Timing-manipulation detector

class _TimingVisitor(ast.NodeVisitor):
    """
    Detects two timing-manipulation patterns:

    Pattern A  Missing torch.cuda.synchronize() around timing code.
        We look for time.time() / time.perf_counter() / torch.cuda.Event
        usage without a synchronize() call in the same scope.

    Pattern B  Separate CUDA streams for timing vs compute.
        If the kernel is launched on a non-default stream but timing
        uses the default stream, results are artificially fast.
    """

    def __init__(self):
        self.timing_calls = []       # lines where time is sampled
        self.sync_calls = []         # lines where synchronize() is called
        self.stream_launches = []    # lines where non-default stream used

    def _attr_chain(self, node) -> str:
        """Reconstruct dotted attribute chain as a string."""
        if isinstance(node, ast.Attribute):
            return self._attr_chain(node.value) + "." + node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ""

    def visit_Call(self, node: ast.Call):
        chain = self._attr_chain(node.func)

        # Timing calls
        if chain in ("time.time", "time.perf_counter", "time.monotonic",
                     "torch.cuda.Event"):
            self.timing_calls.append(node.lineno)

        # Synchronize calls
        if chain in ("torch.cuda.synchronize",) or \
                (isinstance(node.func, ast.Attribute) and
                 node.func.attr == "synchronize"):
            self.sync_calls.append(node.lineno)

        # Stream keyword in kernel launch  kernel[grid](..., stream=s)
        for kw in node.keywords:
            if kw.arg == "stream":
                self.stream_launches.append(node.lineno)

        self.generic_visit(node)


def check_timing_manipulation(kernel) -> tuple:
    """
    Return (True, detail) if no timing-manipulation patterns are detected,
    (False, detail) otherwise.
    """
    try:
        src = _get_source(kernel)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError) as e:
        return False, f"Could not parse source: {e}"

    visitor = _TimingVisitor()
    visitor.visit(tree)

    issues = []

    if visitor.timing_calls and not visitor.sync_calls:
        issues.append(
            f"Timing sampled at line(s) {visitor.timing_calls} but "
            "torch.cuda.synchronize() was never called. "
            "GPU work may not have completed when time is recorded."
        )

    if visitor.stream_launches:
        issues.append(
            f"Kernel launched with explicit stream at line(s) "
            f"{visitor.stream_launches}. Verify timing is on the same stream."
        )

    if issues:
        return False, " | ".join(issues)

    return True, "No timing-manipulation patterns detected."


# 4. Partial-computation detector

class _PartialComputationVisitor(ast.NodeVisitor):
    """
    Estimates what fraction of computation is done inside the custom kernel
    vs delegated back to PyTorch/reference calls.

    Heuristic counts:
      - PyTorch op calls (torch.*, F.*) as 'delegated FLOPs'
      - Triton kernel launches (subscript-call pattern) as 'custom FLOPs'

    A high ratio of torch calls to triton launches is suspicious.
    """

    # PyTorch functions that do significant compute
    _TORCH_COMPUTE_OPS = {
        "matmul", "mm", "bmm", "linear", "conv2d", "softmax",
        "layer_norm", "batch_norm", "relu", "gelu", "sigmoid",
        "sum", "mean", "max", "min", "norm", "dot",
    }

    def __init__(self):
        self.torch_op_calls = []
        self.triton_launches = []

    def _is_torch_compute(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr in self._TORCH_COMPUTE_OPS
        return False

    def visit_Call(self, node: ast.Call):
        if self._is_torch_compute(node):
            self.torch_op_calls.append(node.lineno)
        if isinstance(node.func, ast.Subscript):
            self.triton_launches.append(node.lineno)
        self.generic_visit(node)


def check_partial_computation(kernel, max_torch_ratio: float = 0.5) -> tuple:
    """
    Return (True, detail) if the custom kernel does most of its own work,
    (False, detail) if the ratio of delegated PyTorch calls to total
    compute calls exceeds `max_torch_ratio`.

    Args:
        kernel: source string or callable.
        max_torch_ratio: fraction threshold above which we flag delegation.
                         Default 0.5  more than half the ops delegated.
    """
    try:
        src = _get_source(kernel)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError) as e:
        return False, f"Could not parse source: {e}"

    visitor = _PartialComputationVisitor()
    visitor.visit(tree)

    n_torch = len(visitor.torch_op_calls)
    n_triton = len(visitor.triton_launches)
    total = n_torch + n_triton

    if total == 0:
        return True, "No compute ops detected; skipping partial-computation check."

    ratio = n_torch / total
    if ratio > max_torch_ratio:
        return False, (
            f"High PyTorch delegation ratio: {ratio:.0%} of compute calls "
            f"({n_torch}/{total}) are standard PyTorch ops at "
            f"lines {visitor.torch_op_calls}. "
            "Kernel may be delegating most work back to the reference."
        )

    return True, (
        f"Partial-computation check passed: "
        f"{n_triton} Triton launch(es), {n_torch} PyTorch op(s) "
        f"(delegation ratio {ratio:.0%})."
    )