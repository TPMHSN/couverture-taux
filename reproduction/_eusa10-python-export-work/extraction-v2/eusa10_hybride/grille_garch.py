"""Finite-horizon EUSA10 hedge decisions with AR(1)–GARCH(1,1) state."""

import math

import numpy as np


HORIZON = 60


def _rate_offsets(steps, rate_step):
    """Keep a uniform core and spread the knots farther apart in the tails."""
    core = min(steps, math.ceil(500 / rate_step))
    regular = np.arange(1, core + 1)
    if steps == core:
        return regular
    tail_count = min(steps - core, max(48, math.ceil(220 / rate_step)))
    if tail_count == steps - core:
        tail = np.arange(core + 1, steps + 1)
    else:
        tail = np.unique(np.rint(np.geomspace(core + 1, steps, tail_count)))
    return np.concatenate((regular, tail))


def _state_grids(a, phi, omega, alpha, beta, h_next, spot, nodes, weights,
                 rate_step, vol_nodes):
    """Place the current rate and volatility at exact grid knots.

    Use finite-horizon moments to bound the rate and volatility search region.
    The volatility axis is logarithmically spaced in standard deviation, with
    interpolation linear in standard deviation between consecutive knots.
    """
    z_mean = float(weights @ nodes)
    z_variance = float(weights @ (nodes - z_mean) ** 2)
    multipliers = alpha * nodes**2 + beta
    c_mean = float(weights @ multipliers)
    c_second = float(weights @ multipliers**2)
    c_variance = float(weights @ (multipliers - c_mean) ** 2)
    c_min = float(multipliers.min())
    c_max = float(multipliers.max())
    mean_h, variance_h = h_next, 0.0
    low_h, high_h = h_next, max(h_next, omega + c_max * h_next)
    smallest_h = h_next
    mean_rate, variance_r = spot, 0.0
    low_rate, high_rate = spot, spot
    for _ in range(HORIZON - 1):
        mean_rate = a + phi * mean_rate + z_mean * math.sqrt(mean_h)
        variance_r = phi**2 * variance_r + z_variance * mean_h
        radius = 6 * math.sqrt(variance_r)
        low_rate = min(low_rate, mean_rate - radius)
        high_rate = max(high_rate, mean_rate + radius)
        variance_h = c_second * variance_h + c_variance * mean_h**2
        mean_h = omega + c_mean * mean_h
        spread_h = math.sqrt(variance_h)
        high_h = max(high_h, mean_h + 4 * spread_h, omega + c_max * mean_h)
        smallest_h = omega + c_min * smallest_h
        low_h = min(low_h, smallest_h)

    if not np.isfinite((low_rate, high_rate, low_h, high_h)).all():
        raise ValueError("The finite-horizon state grid is unbounded at this resolution.")
    # Add two points beyond each statistical boundary for rate extrapolation.
    left = math.ceil((spot - low_rate) / rate_step) + 2
    right = math.ceil((high_rate - spot) / rate_step) + 2
    current_vol = math.sqrt(h_next)
    if alpha == 0 and beta == 0:
        vols = np.unique((math.sqrt(omega), current_vol))
    elif math.isclose(low_h, high_h, rel_tol=1e-10, abs_tol=0.0):
        vols = np.array([current_vol])
    else:
        vols = np.geomspace(math.sqrt(low_h), math.sqrt(high_h), vol_nodes)
        # Replacing an interior knot preserves the bounds and avoids an
        # additional interpolation on the observed conditional volatility.
        if current_vol <= vols[0]:
            vols[0] = current_vol
        elif current_vol >= vols[-1]:
            vols[-1] = current_vol
        else:
            closest = 1 + int(np.argmin(abs(np.log(vols[1:-1] / current_vol))))
            vols[closest] = current_vol
    left_offsets = _rate_offsets(left, rate_step)
    right_offsets = _rate_offsets(right, rate_step)
    rates = spot + rate_step * np.concatenate((-left_offsets[::-1], [0], right_offsets))
    if len(rates) * len(vols) > 180_000:
        raise ValueError("The state grid is too large; increase the rate step.")
    state = int(np.argmin(abs(vols - current_vol))) * len(rates) + len(left_offsets)
    return rates, vols, state


def _kernel(a, phi, omega, alpha, beta, rates, vols, nodes, weights):
    """Precompute indices and weights for semi-Lagrangian Bellman transitions."""
    nrate, nvol, shocks = len(rates), len(vols), len(nodes)
    next_rate = a + phi * rates[None, :, None] + vols[:, None, None] * nodes
    r0 = np.clip(np.searchsorted(rates, next_rate, side="right") - 1, 0, nrate - 2)
    rfrac = (next_rate - rates[r0]) / (rates[r0 + 1] - rates[r0])
    # Linear rate extrapolation preserves the affine payoff at the boundary.
    wr0, wr1 = (1 - rfrac) * weights, rfrac * weights
    if nvol == 1:
        indices = np.stack((r0, r0 + 1), axis=-1)
        coefficients = np.stack((wr0, wr1), axis=-1)
        return indices.reshape(-1, shocks * 2), coefficients.reshape(-1, shocks * 2)

    next_vol = np.sqrt(omega + vols[:, None] ** 2 * (alpha * nodes**2 + beta))
    v0 = np.clip(np.searchsorted(vols, next_vol, side="right") - 1, 0, nvol - 2)
    # Beyond the volatility grid retain the closest boundary value.
    vfrac = np.clip((next_vol - vols[v0]) / (vols[v0 + 1] - vols[v0]), 0, 1)
    base = r0 + (v0 * nrate)[:, None, :]
    vfrac = vfrac[:, None, :]
    indices = np.stack((base, base + 1, base + nrate, base + nrate + 1), axis=-1)
    coefficients = np.stack((wr0 * (1 - vfrac), wr1 * (1 - vfrac),
                             wr0 * vfrac, wr1 * vfrac), axis=-1)
    return indices.reshape(-1, shocks * 4), coefficients.reshape(-1, shocks * 4)
