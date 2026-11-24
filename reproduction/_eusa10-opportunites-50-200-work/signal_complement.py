"""Décision unitaire de l’hybride 1 Md€ avec complément de 50 M€."""

from argparse import ArgumentParser
from datetime import date
import json
from math import sqrt
from pathlib import Path

import numpy as np

from replay_small import (
    CALIBRATION, CAP, GARCH_SHOCKS, GARCH_WEIGHTS, HORIZON, INVENTORY, LARGE_BALANCE,
    LOOKBACK, ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS, RATE_GRID, UNIT, calibrate_ar1,
    fit_garch, fixed_constant,
)
from run_proportional import garch
from eusa10_hybride.donnees import read_fixings


def ordre_tickets(mois, solde, taux, seuil, quartile, debut_complement=25):
    """Appliquer la règle au fixing, en nombres de tickets de 50 M€."""
    if solde == 0:
        return 0, "solde"
    if mois == HORIZON:
        return solde, "échéance"
    if taux < seuil:
        return min(CAP, solde), "arbre"
    if mois >= debut_complement and solde >= LARGE_BALANCE and taux < quartile:
        return 1, "complément"
    return 0, "attendre"


def decider(rates_bp, depart, fixing, solde_m):
    """Calculer un signal avec les seules cotations connues au fixing.

    rates_bp contient un fixing par mois, dans l'ordre, en points de base.
    depart et fixing sont des indices de la série. solde_m est en M€.
    """
    rates = np.asarray(rates_bp, dtype=float)
    mois = fixing - depart + 1
    if rates.ndim != 1 or not CALIBRATION <= depart <= fixing < len(rates) or not 1 <= mois <= HORIZON:
        raise ValueError("Il faut 60 fixings précédant le lancement et un mois de programme entre 1 et 60.")
    if not 0 <= solde_m <= UNIT * INVENTORY or solde_m % UNIT:
        raise ValueError("Le solde doit être un multiple de 50 M€ compris entre 0 et 1 000 M€.")
    if not np.isfinite(rates[depart - CALIBRATION:fixing + 1]).all():
        raise ValueError("Tous les taux utilisés doivent être finis.")

    solde = int(solde_m // UNIT)
    taux = float(rates[fixing])
    reference = float(rates[fixing - LOOKBACK])
    quartile = float(np.quantile(rates[fixing - 12:fixing], .25))
    modele = "garch" if taux > reference else "original"
    seuil = None

    def lire(_mois, courbes):
        return [np.interp(taux, RATE_GRID, courbe) for courbe in courbes]

    if solde and mois < HORIZON:
        if modele == "original":
            fit = calibrate_ar1(rates[depart - CALIBRATION:depart])
            seuil = fixed_constant(*fit, ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS, lire, last=mois)[mois, solde]
        else:
            history = rates[fixing - CALIBRATION:fixing]
            a, phi, _ = calibrate_ar1(history)
            residuals = history[1:] - a - phi * history[:-1]
            omega, alpha, beta, variance, _ = fit_garch(residuals, taux - a - phi * history[-1])
            if alpha == beta == 0:
                seuil = fixed_constant(a, phi, sqrt(variance), GARCH_SHOCKS, GARCH_WEIGHTS,
                                       lire, last=mois)[mois, solde]
            else:
                seuil = garch(a, phi, omega, alpha, beta, variance, taux, last=mois)[mois, solde]
        seuil = float(seuil)

    nombre, motif = ordre_tickets(mois, solde, taux, seuil, quartile)
    return dict(mois=mois, modele=modele, taux_bp=taux, reference_bp=reference,
                seuil_bp=seuil, marge_bp=None if seuil is None else seuil - taux,
                quartile_12_fixings_bp=quartile, solde_avant_m=solde_m,
                ordre_m=nombre * UNIT, motif=motif)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("depart", type=date.fromisoformat)
    parser.add_argument("fixing", type=date.fromisoformat)
    parser.add_argument("solde", type=int)
    args = parser.parse_args()
    workbook = Path(__file__).resolve().parents[2] / "donnees/originaux/EUSA10.xlsx"
    dates, rates = read_fixings(workbook)
    index = {day: i for i, day in enumerate(dates)}
    if args.depart not in index or args.fixing not in index:
        parser.error("Les dates doivent correspondre à des fixings mensuels du classeur.")
    try:
        result = decider(rates, index[args.depart], index[args.fixing], args.solde)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
