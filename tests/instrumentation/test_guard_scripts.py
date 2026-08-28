"""
Pytest wrappers for the executor guard scripts -- SUBPROCESS ONLY.

`check_forkserver_executor.py` and `check_batch_executor.py` stub
`sys.modules["torch"]` process-wide and must never be collected in-process
(tests/instrumentation/README.md explains the leak). Until 2026-08-28 that
made them manual-only: `pytest.ini` collects `test_*.py`, so a reseed
regression -- the exact silent-and-severe failure the forkserver guard
exists for -- would have shipped with a green suite.

These wrappers close that hole without touching the isolation argument:
each guard runs in its own interpreter via subprocess, exactly as the
manual invocation does, and the wrapper asserts exit code 0. The stubs
live and die inside the child process; nothing leaks into the pytest
process. Wall cost is ~1-2 s per guard (they import no real torch).

The OTHER check_* scripts stay deliberately unwrapped: they guard
instrumentation whose regressions surface in their own reports, and one
(`check_kernel_executed_probe.py`) needs a real torch and is expected to
fail on the dev machine. The two wrapped here are the ones whose failure
mode is silent (reseed collapse; drain-loop/fallback semantics) and whose
default just flipped ON (use_forkserver, 2026-08-28).
"""
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent

GUARDS = [
    "check_forkserver_executor.py",
    "check_batch_executor.py",
]


@pytest.mark.parametrize("script", GUARDS)
def test_guard_script_passes(script):
    proc = subprocess.run(
        [sys.executable, str(_HERE / script)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"{script} failed (rc={proc.returncode}).\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2000:]}"
    )
