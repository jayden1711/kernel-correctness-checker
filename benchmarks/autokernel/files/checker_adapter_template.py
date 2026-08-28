"""
FILL THIS IN with a real import of your checker, then wire the function
below into run_benchmark.py's SYSTEMS dict.

The harness only requires:
    my_checker_system(entry, is_mutant, rng) -> (passed: bool, dt: float, detail: str | None)

Below are two common shapes your real checker might already have, and how
to adapt each. Delete whichever doesn't apply.
"""
import time

# ---------------------------------------------------------------------------
# TODO: replace with your real import, e.g.:
#   from kernelchecker.run_checker import check_kernel
#   from kernelchecker.layers import structural_check, numeric_check, algebraic_check
# ---------------------------------------------------------------------------


# --- Shape A: you have one function that takes (op, ref_fn, cand_fn) and
#              returns True/False (or True/False + a reason) -----------------
def my_checker_system_shape_a(entry, is_mutant, rng):
    fn = entry["mutant_fn"] if is_mutant else entry["ref_fn"]
    t0 = time.perf_counter()

    # TODO: replace this line with your real call, e.g.:
    #   result = check_kernel(op=entry["op"], reference=entry["ref_fn"], candidate=fn)
    #   passed, detail = result.passed, result.failed_layer
    passed, detail = True, None  # <-- placeholder, delete once wired in

    dt = time.perf_counter() - t0
    return passed, dt, detail


# --- Shape B: your checker is three separate functions (one per layer),
#              like the demo_placeholder_checker.py stand-in --------------
def my_checker_system_shape_b(entry, is_mutant, rng):
    fn = entry["mutant_fn"] if is_mutant else entry["ref_fn"]
    t0 = time.perf_counter()

    # TODO: replace these three calls with your real layer functions.
    # Keep the short-circuit order (cheapest/fastest check first) --
    # structural is usually near-free, numeric is cheap, algebraic can be
    # the most expensive if it runs many trials.
    if not structural_check(entry["op"], fn):
        return False, time.perf_counter() - t0, "structural"
    if not numeric_check(entry["op"], entry["ref_fn"], fn, entry["input_fn"], rng):
        return False, time.perf_counter() - t0, "numeric"
    if not algebraic_check(entry["op"], fn, entry["input_fn"], rng):
        return False, time.perf_counter() - t0, "algebraic"

    dt = time.perf_counter() - t0
    return True, dt, None


# --- Layer ablation variants (for the "is the 3-layer design load-bearing"
#     table) -- copy this pattern once shape_b is wired in for real --------
def my_checker_numeric_only(entry, is_mutant, rng):
    fn = entry["mutant_fn"] if is_mutant else entry["ref_fn"]
    t0 = time.perf_counter()
    passed = numeric_check(entry["op"], entry["ref_fn"], fn, entry["input_fn"], rng)
    return passed, time.perf_counter() - t0, ("numeric" if not passed else None)


def my_checker_algebraic_only(entry, is_mutant, rng):
    fn = entry["mutant_fn"] if is_mutant else entry["ref_fn"]
    t0 = time.perf_counter()
    passed = algebraic_check(entry["op"], fn, entry["input_fn"], rng)
    return passed, time.perf_counter() - t0, ("algebraic" if not passed else None)


def my_checker_structural_only(entry, is_mutant, rng):
    fn = entry["mutant_fn"] if is_mutant else entry["ref_fn"]
    t0 = time.perf_counter()
    passed = structural_check(entry["op"], fn)
    return passed, time.perf_counter() - t0, ("structural" if not passed else None)
