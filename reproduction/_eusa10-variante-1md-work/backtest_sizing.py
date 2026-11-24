"""Paired 1bn EUSA10 hybrid sizing replay on the frozen daily-launch sample."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
sys.path.insert(0, str(ROOT / "_eusa10-python-export-work" / "extraction-v2"))
from eusa10_hybride.donnees import read_fixings
from eusa10_hybride.garch_fit import fit_garch
from eusa10_hybride.grille_garch import _kernel, _state_grids
from eusa10_hybride.strategie import (
    CALIBRATION, GARCH_SHOCKS, GARCH_WEIGHTS, HORIZON, LOOKBACK, ORIGINAL_SHOCKS,
    ORIGINAL_WEIGHTS, RATE_GRID, calibrate_ar1, constant_expectation,
)
from bellman_sizing import constant_thresholds, garch_thresholds, reachable_minimum


BOOK = ROOT.parent / "donnees/originaux/EUSA10.xlsx"
REFERENCE = ROOT / "reference-hybride-300m.csv"
REF_HASH = "2f1cef61ba93891f2908205473bfb1000f19d77b94611c6c7c5111a76293190e"
BOOK_HASH = "c4113193efb62f3042960abe9922d471509351b8f1330251690f6568df98d7e0"
OUT = ROOT / "backtest-eusa10-hybride-variante-1md-20m.csv"
TICKETS = 50
TICKET_MILLIONS = 20
NOV_DATE = date(2026, 11, 2)


def setup():
    assert sha256(REFERENCE.read_bytes()).hexdigest() == REF_HASH
    assert sha256(BOOK.read_bytes()).hexdigest() == BOOK_HASH
    original_dates, original_rates = read_fixings(BOOK)
    with np.load(ROOT / "_eusa10-hybrid-test-work" / "thresholds.npz") as cache:
        assert np.array_equal(original_rates, cache["source"])
    dates = original_dates + (NOV_DATE,)
    rates = np.r_[original_rates, 340.1]
    assert original_dates[-1] == date(2026, 10, 1)
    month_index = {d.isoformat()[:7]: i for i, d in enumerate(dates)}
    quotes = pd.read_excel(BOOK, sheet_name="Feuil1", header=1, usecols="B:C").dropna(subset=["Taux"])
    daily_bp = dict(zip(quotes["Date"].dt.strftime("%Y-%m-%d"), quotes["Taux"].to_numpy(dtype=float)*10000))
    assert len(daily_bp) == len(quotes)
    group = {}
    with REFERENCE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            j = month_index[row["Mois de lancement"]]
            spot = daily_bp[row["Date de lancement"]]
            assert abs(spot - 100 * decimal(row["EUSA10 au premier fixing (%)"])) < 1e-9
            assert (spot > rates[j-LOOKBACK]) == (row["Modèle au premier fixing"] == "GARCH")
            row["spot_bp"] = spot
            group.setdefault(j, []).append(row)
    expected = list(range(CALIBRATION, len(dates) - HORIZON + 1))
    assert list(group) == expected and sum(map(len, group.values())) == 4607
    for j, rows in group.items():
        assert rows[0]["Date de lancement"] == dates[j].isoformat()
        assert abs(100 * decimal(rows[0]["EUSA10 au premier fixing (%)"]) - rates[j]) < 1e-9
        assert all(r["Dernier fixing"] == dates[j+HORIZON-1].isoformat() for r in rows)
    fits = json.loads((ROOT / "_eusa10-garch-full-work/full.json").read_text("utf-8"))["fits"]
    assert [d.isoformat() for d in original_dates[CALIBRATION:]] == [f["date"] for f in fits]
    return dates, rates, group, fits


def decimal(s):
    return float(s.replace(",", "."))


def constant_model(a, phi, sigma, nodes, weights, sample, tickets, last=HORIZON-1):
    expectation = constant_expectation(a, phi, sigma, nodes, weights)
    return constant_thresholds(RATE_GRID, a + phi * RATE_GRID, expectation, tickets, sample, last=last)


def garch_grid(a, phi, omega, alpha, beta, variance, spot):
    nodes, weights = GARCH_SHOCKS, GARCH_WEIGHTS
    rates, vols, state = _state_grids(
        a, phi, omega, alpha, beta, sqrt(variance)**2, spot, nodes, weights, 2.0, 21,
    )
    indices, coefficients = _kernel(a, phi, omega, alpha, beta, rates, vols, nodes, weights)
    states = np.tile(rates, len(vols))
    return states, a + phi * states, indices, coefficients, state


def canonical(job):
    j, spot, params, tickets = job
    a, phi = params["a_bp"], params["phi"]
    omega, alpha, beta, sigma = (params[k] for k in ("omega_bp2", "alpha", "beta", "sigma_bp"))
    if alpha == beta == 0:
        sample = lambda _month, marginal: np.interp(spot, RATE_GRID, marginal)
        result = constant_model(a, phi, sigma, GARCH_SHOCKS, GARCH_WEIGHTS, sample, tickets)
    else:
        result = garch_thresholds(*garch_grid(a, phi, omega, alpha, beta, sigma**2, spot), tickets)
    return j, result


def trade_path(path, lag, initial_trade, garch_limits, old_limits, tickets):
    trades = [1] if initial_trade else []
    remaining = tickets - int(initial_trade)
    for month in range(2, HORIZON):
        if remaining == 0:
            break
        assert remaining >= reachable_minimum(month, tickets)
        spot = path[month-1]
        limits = garch_limits[month-1] if spot > lag[month-1] else old_limits
        threshold = limits[month, remaining]
        assert np.isfinite(threshold)
        if spot < threshold:
            trades.append(month)
            remaining -= 1
    trades.extend([HORIZON] * remaining)
    assert len(trades) == tickets and len(set(m for m in trades if m < HORIZON)) == tickets - remaining
    return np.asarray(trades)


def cohort(job):
    j, dates, prices, rates, garch_limits, tickets = job
    past = rates[j-CALIBRATION:j]
    path = rates[j:j+HORIZON]
    lag = rates[j-LOOKBACK:j+HORIZON-LOOKBACK]
    current_up = prices > lag[0]
    a, phi, sigma = calibrate_ar1(past)
    initial = np.full(len(prices), np.nan)
    original = None
    if (~current_up).any() or (path[1:-1] <= lag[1:-1]).any():
        def sample(month, values):
            if month == 1:
                initial[~current_up] = np.interp(prices[~current_up], RATE_GRID, values)
            return np.interp(path[month-1], RATE_GRID, values)
        original = constant_model(a, phi, sigma, ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS, sample, tickets)

    if current_up[0]:
        initial[0] = garch_limits[0, 1, tickets]
    up_indices = np.flatnonzero(current_up[1:]) + 1
    if len(up_indices):
        residuals = past[1:] - a - phi * past[:-1]
        omega, alpha, beta, past_variance, _ = fit_garch(residuals)
        if alpha == beta == 0:
            def sample_initial(_month, values):
                initial[up_indices] = np.interp(prices[up_indices], RATE_GRID, values)
                return float(values[0])
            constant_model(
                a, phi, sqrt(past_variance), GARCH_SHOCKS, GARCH_WEIGHTS, sample_initial, tickets, last=1,
            )
        else:
            for i in up_indices:
                spot = float(prices[i])
                innovation = spot - a - phi * past[-1]
                variance = omega + alpha * innovation * innovation + beta * past_variance
                grid = garch_grid(a, phi, omega, alpha, beta, variance, spot)
                initial[i] = garch_thresholds(*grid, tickets, last=1)[1, tickets]
    assert np.isfinite(initial).all()

    replay = []
    plans = {}
    for d, spot, threshold, upward in zip(dates, prices, initial, current_up, strict=True):
        trade = bool(spot < threshold)
        if trade not in plans:
            plans[trade] = trade_path(path, lag, trade, garch_limits, original, tickets)
        trades = plans[trade]
        actual = np.r_[spot, path[1:]]
        linear = actual.mean()
        hybrid = actual[trades-1].mean()
        replay.append((d, linear, hybrid, float(threshold), upward, trades))
    return j, replay


def combine(result, group, monthly_rates):
    rows = []
    for j in sorted(result):
        for new, old in zip(result[j], group[j], strict=True):
            d, linear, hybrid, limit, up, trades = new
            assert d == old["Date de lancement"]
            assert abs(linear - 100 * decimal(old["Taux linéaire (%)"])) <= 1e-8
            old_hybrid = 100 * decimal(old["Taux hybride (%)"])
            old_gain = decimal(old["Gain (pb)"])
            assert abs(linear - old_hybrid - old_gain) < 1e-8
            old_trades = np.asarray([int(m) for m in old["Mois des 6 tickets"].split(" | ")])
            spot = old["spot_bp"]
            actual = np.r_[spot, monthly_rates[j+1:j+HORIZON]]
            assert abs(old_hybrid - actual[old_trades-1].mean()) < 1e-8
            old_forced_pct = float((old_trades == HORIZON).mean()) * 100
            forced_m = int((trades == HORIZON).sum()) * TICKET_MILLIONS
            gain = linear - hybrid
            assert abs(gain - old_gain - (old_hybrid - hybrid)) < 1e-8
            rows.append(dict(
                date=d, first_day=old["Départ au 1er jour TARGET"] == "oui", end=old["Dernier fixing"],
                spot=spot, model="GARCH" if up else "AR(1)", limit=limit, linear=linear, hybrid=hybrid,
                gain=gain, old_hybrid=old_hybrid, old_gain=old_gain, delta=gain-old_gain,
                forced_m=forced_m, old_forced_pct=old_forced_pct, trades=trades.tolist(),
                first_trade=int(trades[0]), coverage_m36=int((trades<=36).sum())*TICKET_MILLIONS,
                coverage_m48=int((trades<=48).sum())*TICKET_MILLIONS,
            ))
    return rows


def performance(rows, metric):
    values = np.asarray([r[metric] for r in rows])
    ties = np.abs(values) <= 1e-8
    return dict(
        count=len(rows), mean=float(values.mean()), median=float(np.median(values)),
        wins=int((values > 1e-8).sum()), losses=int((values < -1e-8).sum()), ties=int(ties.sum()),
        p05=float(np.quantile(values, 0.05)), p95=float(np.quantile(values, 0.95)),
        worst=float(values.min()), worst_date=rows[int(values.argmin())]["date"],
        best=float(values.max()), best_date=rows[int(values.argmax())]["date"],
        winrate=float((values > 1e-8).mean()),
    )


def describe(rows):
    forced = np.asarray([r["forced_m"] for r in rows])
    return dict(
        gain=performance(rows, "gain"), old_gain=performance(rows, "old_gain"),
        delta=performance(rows, "delta"),
        rates_pct=dict(
            linear=float(np.mean([r["linear"] for r in rows])) / 100,
            hybrid=float(np.mean([r["hybrid"] for r in rows])) / 100,
            reference=float(np.mean([r["old_hybrid"] for r in rows])) / 100,
        ),
        forced=dict(
            mean_m=float(forced.mean()), median_m=float(np.median(forced)),
            p05_m=float(np.quantile(forced, .05)), p95_m=float(np.quantile(forced, .95)),
            max_m=int(forced.max()), count=int((forced>0).sum()), at_least_500m=int((forced>=500).sum()),
            mean_old_pct=float(np.mean([r["old_forced_pct"] for r in rows])),
            old_pct_at_least_50=sum(r["old_forced_pct"]>=50 for r in rows),
        ),
        median_first_trade=float(np.median([r["first_trade"] for r in rows])),
        mean_coverage_36=float(np.mean([r["coverage_m36"] for r in rows])),
        mean_coverage_48=float(np.mean([r["coverage_m48"] for r in rows])),
    )


def report(rows):
    assert len(rows) == len({r["date"] for r in rows}) == 4607
    assert sum(r["first_day"] for r in rows) == 216
    assert all(len(r["trades"]) * TICKET_MILLIONS == 1000 and max(r["trades"]) >= 50 for r in rows)
    assert abs(np.mean([r["old_gain"] for r in rows]) - 45.915878084075) < 1e-9
    report = dict(
        all=describe(rows), monthly=describe([r for r in rows if r["first_day"]]),
        historical_only=describe([r for r in rows if r["end"] != NOV_DATE.isoformat()]),
        periods={label:describe([r for r in rows if first <= r["date"] < after])
                 for label, first, after in [
                     ("2004-2014", "2004", "2015"), ("2015-2016", "2015", "2017"),
                     ("2017-2019", "2017", "2020"), ("2020-2021", "2020", "2022"),
                 ]},
    )
    (WORK / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = [
        "Date de lancement", "Dernier fixing", "Départ au 1er jour TARGET",
        "EUSA10 au lancement (%)", "Modèle au lancement", "Seuil de déclenchement au lancement (%)",
        "Taux linéaire (%)", "Taux hybride 1 Md€ (%)", "Gain 1 Md€ vs linéaire (pb)",
        "Taux hybride de référence (%)", "Gain de référence vs linéaire (pb)", "Surplus de gain vs référence (pb)",
        "Solde couvert au mois 60 (M€)", "Part du solde au mois 60 (%)", "Part du solde de référence au mois 60 (%)",
        "Montant couvert à fin M36 (M€)", "Montant couvert à fin M48 (M€)", "Mois des 50 tickets",
    ]
    def fr(value):
        return f"{value:.10f}".replace(".", ",")
    temporary = OUT.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        for r in rows:
            writer.writerow([
                r["date"], r["end"], "oui" if r["first_day"] else "non", fr(r["spot"]/100),
                r["model"], fr(r["limit"]/100), fr(r["linear"]/100), fr(r["hybrid"]/100),
                fr(r["gain"]), fr(r["old_hybrid"]/100), fr(r["old_gain"]), fr(r["delta"]), r["forced_m"],
                fr(r["forced_m"]/10), fr(r["old_forced_pct"]), r["coverage_m36"], r["coverage_m48"],
                " | ".join(map(str, r["trades"])),
            ])
    temporary.replace(OUT)
    print(json.dumps(report["all"], ensure_ascii=True), flush=True)


def run(workers):
    begin = perf_counter()
    dates, rates, group, fits = setup()
    canon_path = WORK / "canonical-50.npz"
    expected_shape = (len(dates), HORIZON+1, TICKETS+1)
    canonical_jobs = [
        (j, rates[j], fits[j-CALIBRATION], TICKETS) for j in range(CALIBRATION, len(dates)-1)
        if rates[j] > rates[j-LOOKBACK]
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if canon_path.exists():
            with np.load(canon_path) as prev:
                assert np.array_equal(rates, prev["rates"])
                modern = prev["thresholds"]
            assert modern.shape == expected_shape
            print("Seuils GARCH des débuts de mois chargés.", flush=True)
        else:
            modern = np.full(expected_shape, np.nan)
            futures = [pool.submit(canonical, job) for job in canonical_jobs]
            for n, future in enumerate(as_completed(futures), 1):
                j, thresholds = future.result()
                modern[j] = thresholds
                if n % 40 == 0 or n == len(futures):
                    print(f"Seuils GARCH début de mois {n}/{len(futures)} ({perf_counter()-begin:.1f} s).", flush=True)
            np.savez_compressed(canon_path, thresholds=modern, rates=rates)
        for j, *_ in canonical_jobs:
            for month in range(1, HORIZON):
                assert np.isfinite(modern[j, month, max(1, reachable_minimum(month, TICKETS)):]).all()
        jobs = []
        for j, rows in group.items():
            launch_dates = [r["Date de lancement"] for r in rows]
            prices = np.asarray([r["spot_bp"] for r in rows])
            prices[0] = rates[j]
            jobs.append((j, launch_dates, prices, rates, modern[j:j+HORIZON], TICKETS))
        results = {}
        futures = [pool.submit(cohort, job) for job in jobs]
        total = 0
        for n, future in enumerate(as_completed(futures), 1):
            j, replays = future.result()
            results[j] = replays
            total += len(replays)
            if n % 24 == 0 or n == len(futures):
                print(f"{total}/4607 départs rejoués ({n}/216 mois, {perf_counter()-begin:.1f} s).", flush=True)
    rows = combine(results, group, rates)
    report(rows)
    print(f"CSV vérifié. Durée {perf_counter()-begin:.1f} s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args().workers)


