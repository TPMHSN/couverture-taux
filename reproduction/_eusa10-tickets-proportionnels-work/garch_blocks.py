"""Inventory-aware solution of the unchanged one-ticket Bellman problem."""

from atexit import register
from contextlib import suppress
from os import environ
from pathlib import Path
from struct import pack
import subprocess
from subprocess import PIPE, Popen

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

import numpy as np


HORIZON = 60
_PROCESS = None


def _close_garch_process():
    global _PROCESS
    if _PROCESS is not None:
        with suppress(BrokenPipeError):
            _PROCESS.stdin.close()
        _PROCESS.wait(timeout=10)
        _PROCESS.stdout.close()
        _PROCESS = None


register(_close_garch_process)


def garch_thresholds(rates, last_mean, indices, weights, state, tickets, base_size, first=1, last=HORIZON-1):
    """Solve the same recurrence on the original rate and variance grid."""
    global _PROCESS
    if _PROCESS is None:
        engine = Path(__file__).with_name("GarchBellman.dll")
        _PROCESS = Popen(
            ["dotnet", str(engine)], stdin=PIPE, stdout=PIPE, creationflags=CREATE_NO_WINDOW,
            env=dict(environ, DOTNET_TieredCompilation="0"),
        )

    states, branches = indices.shape
    _PROCESS.stdin.write(pack("<8i", states, branches, state, tickets, base_size, HORIZON, first, last))
    for array, dtype in ((rates, "<f8"), (last_mean, "<f8"), (indices, "<i4"), (weights, "<f8")):
        _PROCESS.stdin.write(np.ascontiguousarray(array, dtype=dtype).data)
    _PROCESS.stdin.flush()
    size = (HORIZON + 1) * (tickets + 1) * 8
    raw = _PROCESS.stdout.read(size)
    if len(raw) != size:
        raise RuntimeError(f"Expected {size} bytes from GARCH Bellman, received {len(raw)}.")
    return np.frombuffer(raw, dtype="<f8").reshape(HORIZON + 1, tickets + 1)



