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


def reachable_minimum(month, original_tickets):
    return max(0, original_tickets - (month - 1))


def constant_thresholds(
    rates, last_mean, expectation, tickets, sample, first=1, last=HORIZON - 1
):
    """Solve with the original constant-volatility numerical transitions."""
    output = np.full((HORIZON + 1, tickets + 1), np.nan)
    next_low = reachable_minimum(HORIZON - 1, tickets)
    next_high = 1
    zero = np.zeros_like(rates)
    savings = np.minimum(zero, rates - last_mean)
    next_values = np.stack([last_mean, zero, savings] if next_low == 0 else [last_mean, savings])
    if HORIZON - 1 <= last:
        output[HORIZON - 1, max(1, next_low):] = sample(HORIZON - 1, last_mean)

    for month in range(HORIZON - 2, first - 1, -1):
        low = reachable_minimum(month, tickets)
        high = min(tickets, HORIZON - month)
        reachable = np.arange(max(1, low), high + 1)
        future = expectation(next_values)
        mean = future[0]
        wait = future[1 + np.minimum(reachable, next_high) - next_low]
        trade = future[1 + np.minimum(reachable - 1, next_high) - next_low]
        savings = np.minimum(wait, rates - mean + trade)
        if month <= last:
            for k, marginal in zip(reachable, mean + (wait - trade), strict=True):
                output[month, k] = sample(month, marginal)
            if high < tickets:
                output[month, high+1:] = sample(month, mean)
        prefix = [mean, zero] if low == 0 else [mean]
        next_values = np.vstack([*prefix, savings])
        next_low, next_high = low, high
    return output


def _close_garch_process():
    global _PROCESS
    if _PROCESS is not None:
        with suppress(BrokenPipeError):
            _PROCESS.stdin.close()
        _PROCESS.wait(timeout=10)
        _PROCESS.stdout.close()
        _PROCESS = None


register(_close_garch_process)


def garch_thresholds(rates, last_mean, indices, weights, state, tickets, first=1, last=HORIZON-1):
    """Solve the same recurrence on the original rate and variance grid."""
    global _PROCESS
    if _PROCESS is None:
        engine = Path(__file__).with_name("GarchBellman.dll")
        _PROCESS = Popen(
            ["dotnet", str(engine)], stdin=PIPE, stdout=PIPE, creationflags=CREATE_NO_WINDOW,
            env=dict(environ, DOTNET_TieredCompilation="0"),
        )

    states, branches = indices.shape
    _PROCESS.stdin.write(pack("<7i", states, branches, state, tickets, HORIZON, first, last))
    for array, dtype in ((rates, "<f8"), (last_mean, "<f8"), (indices, "<i4"), (weights, "<f8")):
        _PROCESS.stdin.write(np.ascontiguousarray(array, dtype=dtype).data)
    _PROCESS.stdin.flush()
    size = (HORIZON + 1) * (tickets + 1) * 8
    raw = _PROCESS.stdout.read(size)
    if len(raw) != size:
        raise RuntimeError(f"Expected {size} bytes from GARCH Bellman, received {len(raw)}.")
    return np.frombuffer(raw, dtype="<f8").reshape(HORIZON + 1, tickets + 1)

