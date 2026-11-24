"""Rolling backtest of the hybrid against equal monthly execution."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .strategie import (
    CALIBRATION, HORIZON, LOOKBACK, TICKET_EUR, TICKETS,
    decision_from_thresholds, garch_thresholds, model_at, original_thresholds,
)


@dataclass(frozen=True)
class Result:
    start: date
    end: date
    linear_bp: float
    hybrid_bp: float
    ticket_months: tuple[int, ...]

    @property
    def gain_bp(self):
        return self.linear_bp - self.hybrid_bp

    @property
    def volume_m60_eur(self):
        return self.ticket_months.count(HORIZON) * TICKET_EUR


def _solve_for_date(task):
    index, history, spot, first_month = task
    return index, garch_thresholds(history, spot, first_month)


def _replay(dates, rates, start, modern):
    path = rates[start:start + HORIZON]
    legacy = original_thresholds(rates[start - CALIBRATION:start], path)
    remaining = TICKETS
    trades = []
    for month, spot in enumerate(path, 1):
        if not remaining:
            break
        index = start + month - 1
        reference = rates[index - LOOKBACK]
        if month == HORIZON:
            row = None
        else:
            row = modern[index][month] if model_at(spot, reference) == "GARCH" else legacy[month]
        decision = decision_from_thresholds(month, remaining, spot, reference, row)
        count = decision.montant_eur // TICKET_EUR
        remaining -= count
        trades.extend([month] * count)
    if remaining or len(trades) != TICKETS:
        raise RuntimeError("Le programme doit couvrir exactement 300 M€.")
    hybrid = float(path[np.asarray(trades) - 1].mean())
    return Result(dates[start], dates[start + HORIZON - 1], float(path.mean()), hybrid, tuple(trades))


def run(dates, rates, starts=None, workers=3):
    """Run whole 60-fixing programmes following at least 60 past fixings."""
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (len(dates),) or not np.isfinite(rates).all():
        raise ValueError("Les dates et cotations doivent former une série alignée de taux finis.")
    last_start = len(rates) - HORIZON
    selected = list(range(CALIBRATION, last_start + 1) if starts is None else starts)
    if not selected or any(i < CALIBRATION or i > last_start for i in selected):
        raise ValueError("Chaque départ requiert 60 fixings antérieurs et 60 fixings de programme.")
    if workers < 1:
        raise ValueError("Il faut au moins un processus de calcul.")

    # The fitted GARCH for an absolute fixing can serve several programme ages.
    ages = {}
    for start in selected:
        for month in range(1, HORIZON):
            index = start + month - 1
            if rates[index] > rates[index - LOOKBACK]:
                ages[index] = min(month, ages.get(index, month))
    tasks = [
        (i, rates[i - CALIBRATION:i], rates[i], first_month)
        for i, first_month in sorted(ages.items())
    ]
    if workers == 1 or len(tasks) <= 1:
        modern = dict(map(_solve_for_date, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            modern = dict(pool.map(_solve_for_date, tasks))
    return [_replay(dates, rates, start, modern) for start in selected]


def summary(results):
    gains = np.array([row.gain_bp for row in results])
    return {
        "programmes": len(results),
        "gain_moyen_pb": float(gains.mean()),
        "gain_median_pb": float(np.median(gains)),
        "victoires": int((gains > 1e-8).sum()),
        "defaites": int((gains < -1e-8).sum()),
        "egalites": int((np.abs(gains) <= 1e-8).sum()),
        "quantile_5_pct_pb": float(np.quantile(gains, 0.05)),
        "pire_gain_pb": float(gains.min()),
        "meilleur_gain_pb": float(gains.max()),
    }


def write_csv(output: Path, results):
    output.parent.mkdir(parents=True, exist_ok=True)

    def french(value):
        return f"{value:.12f}".replace(".", ",")

    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=output.parent, encoding="utf-8-sig", newline="", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, delimiter=";")
            writer.writerow([
                "Départ", "Fin", "Taux linéaire (%)", "Taux hybride (%)", "Gain (pb)",
                "Mois des tickets de 50 M€", "Solde couvert au mois 60 (EUR)",
            ])
            for row in results:
                writer.writerow([
                    row.start.isoformat(), row.end.isoformat(), french(row.linear_bp / 100),
                    french(row.hybrid_bp / 100), french(row.gain_bp),
                    " | ".join(map(str, row.ticket_months)), row.volume_m60_eur,
                ])


        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
