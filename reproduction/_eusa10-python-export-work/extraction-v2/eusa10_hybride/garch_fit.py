"""Gaussian GARCH(1,1) fit for already-calibrated, zero-mean residuals in bp."""

from math import exp, log, pi

import numpy as np


# Optimize stationary log-variance, persistence, and the ARCH share.
# This box is equivalent to omega > 0, alpha >= 0, beta >= 0,
# alpha + beta <= 1 - 1e-6, with broad bounds on normalized variance.
_LOWER = np.array([-16.0, 0.0, 0.0])
_UPPER = np.array([16.0, 1.0 - 1e-6, 1.0])
_IDENTITY = np.eye(3)
_STARTS = (
    (0.0, 0.15, 1.00),
    (0.0, 0.50, 0.60),
    (-0.4, 0.80, 0.24),
    (0.0, 0.94, 0.10),
    (0.4, 0.99, 0.035),
    (0.0, 0.998, 0.12),
    (0.0, 0.998, 0.00),
)


def _loss(point, squares):
    log_variance, persistence, arch_share = map(float, point)
    variance = exp(log_variance)
    omega = variance * (1.0 - persistence)
    alpha = persistence * arch_share
    beta = persistence * (1.0 - arch_share)

    h = variance
    dl, dp, dq = variance, 0.0, 0.0
    total = gl = gp = gq = 0.0
    for value in squares:
        # Differentiate log(h) + value / h before assimilating value.
        inverse = 1.0 / h
        ratio = value * inverse
        factor = (1.0 - ratio) * inverse
        total += log(h) + ratio
        gl += factor * dl
        gp += factor * dp
        gq += factor * dq

        dl = omega + beta * dl
        dp = -variance + arch_share * value + (1.0 - arch_share) * h + beta * dp
        dq = persistence * (value - h) + beta * dq
        h = omega + alpha * value + beta * h

    factor = 0.5 / len(squares)
    return total * factor, np.array([gl * factor, gp * factor, gq * factor])


def _line_search(point, loss, gradient, direction, squares):
    size = 1.0
    for _ in range(32):
        candidate = np.clip(point + size * direction, _LOWER, _UPPER)
        step = candidate - point
        descent = float(gradient @ step)
        if descent < 0.0:
            new_loss, new_gradient = _loss(candidate, squares)
            if new_loss <= loss + 1e-4 * descent:
                return candidate, new_loss, new_gradient, step
        size *= 0.5
    return None


def _local_fit(start, squares):
    point = np.array(start, dtype=float)
    loss, gradient = _loss(point, squares)
    inverse = _IDENTITY.copy()
    for _ in range(120):
        on_lower = point <= _LOWER + 1e-12
        on_upper = point >= _UPPER - 1e-12
        active = (on_lower & (gradient > 0.0)) | (on_upper & (gradient < 0.0))
        projected = np.where(active, 0.0, gradient)
        if np.max(np.abs(projected)) < 2e-7:
            break

        direction = -(inverse @ projected)
        blocked = active | (on_lower & (direction < 0.0)) | (on_upper & (direction > 0.0))
        direction[blocked] = 0.0
        accepted = _line_search(point, loss, gradient, direction, squares)
        if accepted is None:
            inverse = _IDENTITY.copy()
            accepted = _line_search(point, loss, gradient, -projected, squares)
            if accepted is None:
                break

        next_point, next_loss, next_gradient, step = accepted
        change = next_gradient - gradient
        curvature = float(step @ change)
        if curvature > 1e-10 * np.linalg.norm(step) * np.linalg.norm(change):
            # Inverse-BFGS update with the actual (projected) step.
            transformed = inverse @ change
            inverse += (
                (curvature + change @ transformed) / (curvature * curvature)
                * np.outer(step, step)
                - (np.outer(transformed, step) + np.outer(step, transformed)) / curvature
            )
            if not np.isfinite(inverse).all() or np.max(np.abs(inverse)) > 1e12:
                inverse = _IDENTITY.copy()
        point, loss, gradient = next_point, next_loss, next_gradient
        if np.max(np.abs(step)) < 1e-10:
            break
    return point, loss


def fit_garch(residuals, latest_residual=None):
    """Return (omega, alpha, beta, h_next, loglik) for past residuals in bp.

    The Gaussian likelihood starts at the fitted stationary variance. Residuals
    are scaled for optimization only. Omega and h_next are returned in bp²,
    and loglik includes the Gaussian constant in the original units.

    latest_residual, if supplied, updates only the variance filter after the
    fit. Pass None for the forecast made before observing that residual.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or len(residuals) < 3 or not np.isfinite(residuals).all():
        raise ValueError("At least three finite, one-dimensional residuals are required.")
    squared = residuals * residuals
    mean_squared = float(squared.mean())
    if not np.isfinite(mean_squared):
        raise ValueError("Residual variance overflows in bp squared.")
    if latest_residual is not None:
        latest_residual = float(latest_residual)
        if not np.isfinite(latest_residual):
            raise ValueError("The latest residual must be finite.")

    scale = max(mean_squared, 1e-12)
    squares = tuple((squared / scale).tolist())
    # The alpha=0 edge has constant variance regardless of beta under the
    # stationary initialization. Use its identifiable representation beta=0.
    best = np.array([log(max(mean_squared / scale, exp(_LOWER[0]))), 0.0, 0.0])
    best_loss, _ = _loss(best, squares)
    if mean_squared > 0.0:
        for start in _STARTS:
            candidate, loss = _local_fit(start, squares)
            if loss < best_loss - 1e-8:
                best, best_loss = candidate, loss

    log_variance, persistence, arch_share = map(float, best)
    h = exp(log_variance) * scale
    omega = h * (1.0 - persistence)
    alpha = persistence * arch_share
    beta = persistence * (1.0 - arch_share)
    for value in squared.tolist():
        h = omega + alpha * value + beta * h
    if latest_residual is not None:
        h = omega + alpha * latest_residual * latest_residual + beta * h

    loglik = -len(squares) * (best_loss + 0.5 * (log(2.0 * pi) + log(scale)))
    return omega, alpha, beta, h, loglik

