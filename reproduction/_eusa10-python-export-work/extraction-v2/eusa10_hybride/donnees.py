"""Read the monthly EUSA10 fixings used by the historical replay."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter


def first_target_day(month: pd.Period) -> date:
    day = month.start_time.date()
    year = day.year
    sunday = easter(year)
    closed = {
        date(year, 1, 1),
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
        sunday - timedelta(days=2),
        sunday + timedelta(days=1),
    }
    while day.weekday() >= 5 or day in closed:
        day += timedelta(days=1)
    return day


def read_fixings(workbook: Path) -> tuple[tuple[date, ...], np.ndarray]:
    """Read Date and Taux in B:C of Feuil1 and return fixings in basis points."""
    quotes = pd.read_excel(workbook, sheet_name="Feuil1", header=1, usecols="B:C")
    if list(quotes.columns) != ["Date", "Taux"]:
        raise ValueError("La deuxième ligne de Feuil1 doit contenir Date et Taux en B:C.")
    quotes = quotes.dropna(subset=["Taux"]).copy()
    if quotes.empty or quotes["Date"].isna().any():
        raise ValueError("Le classeur doit fournir une date pour chaque cotation.")
    quotes["Date"] = pd.to_datetime(quotes["Date"], errors="raise").dt.normalize()
    quotes["Taux"] = pd.to_numeric(quotes["Taux"], errors="raise")
    if quotes["Date"].duplicated().any() or not np.isfinite(quotes["Taux"]).all():
        raise ValueError("Il faut une seule cotation finie par date.")

    months = pd.period_range(
        quotes["Date"].min().to_period("M"), quotes["Date"].max().to_period("M"), freq="M"
    )
    date_index = pd.DatetimeIndex([first_target_day(month) for month in months])
    monthly = quotes.set_index("Date")["Taux"].reindex(date_index)
    if monthly.isna().any():
        missing = ", ".join(monthly.index[monthly.isna()].strftime("%Y-%m-%d"))
        raise ValueError(f"Fixings de début de mois absents du classeur : {missing}.")
    rates = monthly.to_numpy(dtype=float) * 10_000
    if not np.isfinite(rates).all():
        raise ValueError("La conversion des taux en points de base dépasse la plage numérique.")
    return tuple(value.date() for value in date_index), rates

