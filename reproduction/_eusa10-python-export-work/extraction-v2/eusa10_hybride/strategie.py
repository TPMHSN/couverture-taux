"""Modèles de taux, calcul à rebours et décision mensuelle de l’arbre hybride."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .garch_fit import fit_garch
from .grille_garch import _kernel, _state_grids

HORIZON = 60
CALIBRATION = 60
LOOKBACK = 6
TICKETS = 6
TICKET_EUR = 50_000_000
RATE_GRID = np.linspace(-500, 1300, 3601)
ORIGINAL_SHOCKS = np.random.default_rng(20261119).standard_normal(6)
ORIGINAL_WEIGHTS = np.full(6, 1 / 6)
hermite_points, hermite_weights = np.polynomial.hermite.hermgauss(10)
GARCH_SHOCKS = sqrt(2) * hermite_points
GARCH_WEIGHTS = hermite_weights / sqrt(np.pi)


@dataclass(frozen=True)
class Decision:
    mois: int
    taux_bp: float
    reference_6_mois_bp: float
    tickets_avant: int
    modele: str | None
    seuil_bp: float | None
    marge_bp: float | None
    continuation_attendre_pb_cumules: float | None
    continuation_apres_ticket_pb_cumules: float | None
    action: str
    montant_eur: int


def calibrate_ar1(history_bp: Sequence[float]) -> tuple[float, float, float]:
    """Ajuste r[t+1] = a + phi*r[t] + epsilon sur les 60 fixings disponibles."""
    history = np.asarray(history_bp, dtype=float)
    if history.shape != (CALIBRATION,) or not np.isfinite(history).all():
        raise ValueError("La calibration requiert 60 fixings précédents, exprimés en pb.")
    x, y = history[:-1], history[1:]
    dx = x - x.mean()
    variance = np.dot(dx, dx)
    if variance <= 0:
        raise ValueError("La calibration AR(1) requiert des taux historiques variables.")
    phi = np.dot(dx, y - y.mean()) / variance
    a = y.mean() - phi * x.mean()
    residuals = y - a - phi * x
    sigma = np.sqrt(np.dot(residuals, residuals) / (len(residuals) - 2))
    return float(a), float(phi), float(sigma)


def constant_expectation(a: float, phi: float, sigma: float, nodes, weights) -> Callable:
    step = RATE_GRID[1] - RATE_GRID[0]
    next_rate = a + phi * RATE_GRID[:, None] + sigma * nodes[None, :]
    low = np.clip(np.floor((next_rate - RATE_GRID[0]) / step).astype(np.intp), 0, len(RATE_GRID) - 2)
    fraction = (next_rate - RATE_GRID[low]) / step
    low_weight = (1 - fraction) * weights
    high_weight = fraction * weights

    def expectation(values):
        return (np.einsum("kgn,gn->kg", values[:, low], low_weight)
                + np.einsum("kgn,gn->kg", values[:, low + 1], high_weight))

    return expectation


def continuation_thresholds(
    state_rates, terminal_mean, expectation, read_at_state, first_month=1, last_month=HORIZON - 1
):
    """Calcule le seuil C(k) - C(k-1) pour chaque mois et solde positif.

    Les colonnes de la valeur de Bellman sont indexées par l'état de taux et,
    dans le modèle GARCH, de volatilité. La première dimension est le solde k.
    """
    quantities = np.arange(TICKETS + 1)[:, None]
    value = quantities * state_rates
    thresholds = np.full((HORIZON + 1, TICKETS + 1), np.nan)
    for month in range(HORIZON - 1, first_month - 1, -1):
        if month == HORIZON - 1:
            # Le coût terminal k*r est affine. Le backtest utilise ici a + phi*r.
            future = quantities * terminal_mean
        else:
            future = expectation(value)
        marginal = future[1:] - future[:-1]
        if month <= last_month:
            thresholds[month, 1:] = read_at_state(month, marginal)
        value[1:] = np.minimum(future[1:], state_rates + future[:-1])
    return thresholds


def original_thresholds(history_bp, observed_path_bp, first_month=1):
    a, phi, sigma = calibrate_ar1(history_bp)
    expectation = constant_expectation(a, phi, sigma, ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS)

    def read(month, marginal):
        spot = observed_path_bp[month - 1]
        return [np.interp(spot, RATE_GRID, row) for row in marginal]

    return continuation_thresholds(
        RATE_GRID, a + phi * RATE_GRID, expectation, read, first_month, len(observed_path_bp)
    )


def garch_thresholds(history_bp, spot_bp, first_month=1):
    history = np.asarray(history_bp, dtype=float)
    a, phi, _ = calibrate_ar1(history)
    residuals = history[1:] - a - phi * history[:-1]
    innovation = float(spot_bp) - a - phi * history[-1]
    omega, alpha, beta, variance, _ = fit_garch(residuals, innovation)
    sigma = sqrt(variance)
    if alpha == beta == 0:
        expectation = constant_expectation(a, phi, sigma, GARCH_SHOCKS, GARCH_WEIGHTS)

        def read(_month, marginal):
            return [np.interp(spot_bp, RATE_GRID, row) for row in marginal]

        return continuation_thresholds(RATE_GRID, a + phi * RATE_GRID, expectation, read, first_month)

    rates, vols, state = _state_grids(
        a, phi, omega, alpha, beta, sigma**2, float(spot_bp), GARCH_SHOCKS, GARCH_WEIGHTS, 2.0, 21
    )
    indices, coefficients = _kernel(a, phi, omega, alpha, beta, rates, vols, GARCH_SHOCKS, GARCH_WEIGHTS)
    states = np.tile(rates, len(vols))

    def expectation(values):
        return np.einsum("ksb,sb->ks", values[:, indices], coefficients)

    return continuation_thresholds(
        states, a + phi * states, expectation, lambda _month, marginal: marginal[:, state], first_month
    )


def model_at(spot_bp: float, reference_bp: float) -> str:
    return "GARCH" if spot_bp > reference_bp else "AR(1)"


def decision_from_thresholds(month, tickets, spot_bp, reference_bp, thresholds=None) -> Decision:
    """Transforme la ligne de seuils du mois en décision pour le solde réel."""
    spot = float(spot_bp)
    reference = float(reference_bp)
    if not 1 <= month <= HORIZON or not 0 <= tickets <= TICKETS:
        raise ValueError("Le mois doit être de 1 à 60 et le solde de 0 à 6 tickets.")
    if tickets == 0 or month == HORIZON:
        action = "solde_couvert" if tickets == 0 else "couvrir_solde"
        return Decision(month, spot, reference, tickets, None, None, None, None, None,
                        action, tickets * TICKET_EUR)

    limits = np.asarray(thresholds[1:tickets + 1], dtype=float)
    if limits.shape != (tickets,) or not np.isfinite(limits).all():
        raise ValueError("Seuils de continuation invalides pour ce fixing.")
    threshold = float(limits[-1])
    continuation = np.r_[0., limits.cumsum()]
    trade = spot < threshold
    return Decision(
        month, spot, reference, tickets, model_at(spot, reference), threshold, threshold - spot,
        float(continuation[-1]), float(continuation[-2]), "traiter" if trade else "attendre",
        TICKET_EUR if trade else 0,
    )


def decide(rates_bp: Sequence[float], start_index: int, fixing_index: int, tickets: int) -> Decision:
    """Évalue un fixing à partir de la série historique en pb et du solde avant décision."""
    rates = np.asarray(rates_bp, dtype=float)
    if rates.ndim != 1 or not CALIBRATION <= start_index <= fixing_index < len(rates):
        raise ValueError("Le départ requiert 60 fixings antérieurs et le fixing doit être connu.")
    month = fixing_index - start_index + 1
    if not 1 <= month <= HORIZON or not 0 <= tickets <= TICKETS:
        raise ValueError("Le fixing doit appartenir au programme de 60 mois et le solde vaut de 0 à 6 tickets.")
    if not np.isfinite(rates[start_index - CALIBRATION:fixing_index + 1]).all():
        raise ValueError("L'historique doit contenir uniquement des taux finis.")
    spot = rates[fixing_index]
    reference = rates[fixing_index - LOOKBACK]
    if tickets == 0 or month == HORIZON:
        return decision_from_thresholds(month, tickets, spot, reference)
    if model_at(spot, reference) == "GARCH":
        thresholds = garch_thresholds(rates[fixing_index - CALIBRATION:fixing_index], spot, month)
    else:
        thresholds = original_thresholds(
            rates[start_index - CALIBRATION:start_index], rates[start_index:fixing_index + 1], month
        )
    return decision_from_thresholds(month, tickets, spot, reference, thresholds[month])

