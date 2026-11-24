"""Paired EUSA10 replay for several lot sizes, with the frozen hybrid as control."""

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt
from pathlib import Path
from time import perf_counter

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
sys.path.insert(0, str(ROOT / "_eusa10-variante-1md-work"))
from backtest_sizing import (
    CALIBRATION, GARCH_SHOCKS, GARCH_WEIGHTS, HORIZON, LOOKBACK, ORIGINAL_SHOCKS,
    ORIGINAL_WEIGHTS, RATE_GRID, calibrate_ar1, constant_expectation, decimal, fit_garch,
    garch_grid, setup, trade_path,
)
from bellman_compare import constant_thresholds, garch_thresholds


SIZES = (50, 100, 200)
ALL_SIZES = (20, *SIZES)
CONTROL = 6
COUNTS = tuple(sorted({1000 // size for size in SIZES} | {CONTROL}))
MIN_TICKETS, MAX_TICKETS = min(COUNTS), max(COUNTS)
PREVIOUS = ROOT / "backtest-eusa10-hybride-variante-1md-20m.csv"
OUTPUT = ROOT / "backtest-eusa10-hybride-1md-comparaison-tickets.csv"


def constant(a, phi, sigma, nodes, weights, sample, last=HORIZON-1):
    expectation = constant_expectation(a, phi, sigma, nodes, weights)
    return constant_thresholds(
        RATE_GRID, a + phi * RATE_GRID, expectation, MAX_TICKETS, MIN_TICKETS, sample, last=last,
    )


def garch(a, phi, omega, alpha, beta, variance, spot, last=HORIZON-1):
    grid = garch_grid(a, phi, omega, alpha, beta, variance, spot)
    return garch_thresholds(*grid, MAX_TICKETS, MIN_TICKETS, last=last)


def canonical(job):
    j, spot, fit = job
    a, phi, omega, alpha, beta, sigma = (
        fit[key] for key in ("a_bp", "phi", "omega_bp2", "alpha", "beta", "sigma_bp")
    )
    if alpha == beta == 0:
        def sample(_month, _k, values):
            return np.interp(spot, RATE_GRID, values)
        limits = constant(a, phi, sigma, GARCH_SHOCKS, GARCH_WEIGHTS, sample)
    else:
        limits = garch(a, phi, omega, alpha, beta, sigma**2, spot)
    return j, limits


def cohort(job):
    j, prices, rates, modern = job
    history = rates[j-CALIBRATION:j]
    path = rates[j:j+HORIZON]
    lag = rates[j-LOOKBACK:j+HORIZON-LOOKBACK]
    current_up = prices > lag[0]
    initial = np.full((MAX_TICKETS+1, len(prices)), np.nan)
    a, phi, sigma = calibrate_ar1(history)

    legacy = None
    if (~current_up).any() or (path[1:-1] <= lag[1:-1]).any():
        def sample_legacy(month, k, values):
            if month == 1 and k in COUNTS:
                initial[k, ~current_up] = np.interp(prices[~current_up], RATE_GRID, values)
            return np.interp(path[month-1], RATE_GRID, values)
        legacy = constant(a, phi, sigma, ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS, sample_legacy)

    if current_up[0]:
        initial[:, 0] = modern[0, 1, :]
    up = np.flatnonzero(current_up[1:]) + 1
    if len(up):
        residuals = history[1:] - a - phi * history[:-1]
        omega, alpha, beta, previous_variance, _ = fit_garch(residuals)
        if alpha == beta == 0:
            def sample_initial(_month, k, values):
                if k in COUNTS:
                    initial[k, up] = np.interp(prices[up], RATE_GRID, values)
                return float(values[0])
            constant(a, phi, sqrt(previous_variance), GARCH_SHOCKS, GARCH_WEIGHTS, sample_initial, last=1)
        else:
            for i in up:
                spot = float(prices[i])
                innovation = spot - a - phi * history[-1]
                variance = omega + alpha * innovation * innovation + beta * previous_variance
                initial[:, i] = garch(a, phi, omega, alpha, beta, variance, spot, last=1)[1]

    assert np.isfinite(initial[list(COUNTS), :]).all()
    plans = {}
    result = []
    for i, spot in enumerate(prices):
        actual = np.r_[spot, path[1:]]
        linear = float(actual.mean())
        simulations = {}
        for count in COUNTS:
            limit = initial[count, i]
            enter = bool(spot < limit)
            key = count, enter
            if key not in plans:
                plans[key] = trade_path(path, lag, enter, modern, legacy, count)
            trades = plans[key]
            fixed_avg = float(actual[trades-1].mean())
            simulations[count] = fixed_avg, trades.tolist(), float(limit)
        result.append((linear, simulations))
    return j, result


def program(fixed_avg, trades, tickets, size, linear, first_limit=None):
    assert len(trades) == tickets and tickets * size == 1000
    assert sorted(trades) == trades and 1 <= trades[0] <= trades[-1] <= HORIZON
    early = [month for month in trades if month < HORIZON]
    assert len(set(early)) == len(early)
    forced = (tickets-len(early))*size
    return dict(
        rate=fixed_avg, gain=linear-fixed_avg, forced=forced, trade_months=trades,
        covered_12=sum(month <= 12 for month in trades)*size,
        covered_24=sum(month <= 24 for month in trades)*size,
        covered_36=sum(month <= 36 for month in trades)*size,
        largest_fixing=max(size if early else 0, forced),
        finish=trades[-1], first_limit=first_limit,
    )


def combine(results, group, dates, rates):
    with PREVIOUS.open(encoding="utf-8-sig", newline="") as stream:
        previous = {row["Date de lancement"]: row for row in csv.DictReader(stream, delimiter=";")}
    launches = []
    for j, runs in sorted(results.items()):
        path = rates[j:j+HORIZON]
        for (linear, simulated), initial in zip(runs, group[j], strict=True):
            day = initial["Date de lancement"]
            end = initial["Dernier fixing"]
            prior = previous.pop(day)
            assert end == prior["Dernier fixing"] == dates[j+HORIZON-1].isoformat()
            assert abs(linear - 100 * decimal(initial["Taux linéaire (%)"])) < 1e-8
            assert abs(linear - 100 * decimal(prior["Taux linéaire (%)"])) < 1e-8
            control_rate, control_months, _ = simulated[CONTROL]
            assert " | ".join(map(str, control_months)) == initial["Mois des 6 tickets"], (day, control_months)
            assert abs(control_rate - 100 * decimal(initial["Taux hybride (%)"])) < 1e-8
            initial_spot = initial["spot_bp"]
            actual = np.r_[initial_spot, path[1:]]
            variants = {}
            for size in SIZES:
                tickets = 1000 // size
                rate, trades, limit = simulated[tickets]
                assert abs(rate - np.dot(np.bincount(trades, minlength=HORIZON+1)[1:], actual)/tickets) < 1e-10
                variants[size] = program(rate, trades, tickets, size, linear, limit)
            trades_20 = [int(m) for m in prior["Mois des 50 tickets"].split(" | ")]
            rate_20 = float(actual[np.asarray(trades_20)-1].mean())
            assert abs(rate_20 - 100*decimal(prior["Taux hybride 1 Md€ (%)"])) < 1e-8
            variants[20] = program(rate_20, trades_20, 50, 20, linear)
            launches.append(dict(
                day=day, end=end, monthly=day == dates[j].isoformat(), spot=initial_spot, linear=linear,
                terminal_rate=float(path[-1]), control_gain=linear-control_rate, variants=variants,
            ))
    assert not previous and len(launches) == 4607 and sum(r["monthly"] for r in launches) == 216
    return launches


def stats(launches, size):
    variants = [r["variants"][size] for r in launches]
    gains = np.asarray([p["gain"] for p in variants])
    remaining = np.asarray([p["forced"] for p in variants])
    worst = int(np.argmin(gains))
    return dict(
        n=len(launches), mean=float(np.mean(gains)), median=float(np.median(gains)),
        wins=int((gains>1e-8).sum()), ties=int((abs(gains)<=1e-8).sum()),
        worst=float(gains[worst]), worst_start=launches[worst]["day"],
        p05=float(np.quantile(gains, .05)), p95=float(np.quantile(gains, .95)),
        forced_mean=float(remaining.mean()), forced_median=float(np.median(remaining)),
        forced_p95=float(np.quantile(remaining, .95)), forced_max=int(remaining.max()),
        forced_count=int((remaining>0).sum()), forced_at_least_500=int((remaining>=500).sum()),
        median_completion=float(np.median([p["finish"] for p in variants])),
        mean_12=float(np.mean([p["covered_12"] for p in variants])),
        mean_24=float(np.mean([p["covered_24"] for p in variants])),
        mean_36=float(np.mean([p["covered_36"] for p in variants])),
        largest_fixing_mean=float(np.mean([p["largest_fixing"] for p in variants])),
    )


def paired(launches, bigger, smaller):
    differences = np.asarray([r["variants"][bigger]["gain"] - r["variants"][smaller]["gain"] for r in launches])
    return dict(
        mean=float(differences.mean()), median=float(np.median(differences)),
        wins=int((differences > 1e-8).sum()), losses=int((differences < -1e-8).sum()),
        p05=float(np.quantile(differences, .05)),
    )


def write_outputs(launches):
    bins = {
        "all": launches,
        "monthly": [r for r in launches if r["monthly"]],
        "2004-2014": [r for r in launches if "2004" <= r["day"] < "2015"],
        "2015-2016": [r for r in launches if "2015" <= r["day"] < "2017"],
        "2017-2019": [r for r in launches if "2017" <= r["day"] < "2020"],
        "2020-2021": [r for r in launches if "2020" <= r["day"] < "2022"],
    }
    result = {key:{size:stats(rows, size) for size in ALL_SIZES} for key, rows in bins.items()}
    result["paired"] = {f"{a}-{b}":paired(launches, a, b) for a,b in [(50,20),(100,50),(200,100),(200,50)]}
    (WORK/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = [
        "Date de lancement", "Dernier fixing", "Départ au 1er jour TARGET", "EUSA10 au lancement (%)",
        "EUSA10 au dernier fixing (%)", "Taux linéaire (%)", "Gain référence 300 M€ (pb)",
    ]
    for size in ALL_SIZES:
        fields += [
            f"Taux hybride {size} M€ (%)", f"Gain {size} M€ (pb)", f"Solde M60 {size} M€ (M€)",
            f"Couvert à fin M12 {size} M€ (M€)", f"Couvert à fin M24 {size} M€ (M€)",
            f"Mois de couverture intégrale {size} M€", f"Mois des tickets {size} M€",
        ]
    temp = OUTPUT.with_suffix(".pending")
    def fr(value):
        return f"{value:.10f}".replace(".", ",")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        for r in launches:
            line = [
                r["day"], r["end"], "oui" if r["monthly"] else "non",
                fr(r["spot"]/100), fr(r["terminal_rate"]/100), fr(r["linear"]/100), fr(r["control_gain"]),
            ]
            for size in ALL_SIZES:
                variant = r["variants"][size]
                line.extend([
                    fr(variant["rate"]/100), fr(variant["gain"]), variant["forced"], variant["covered_12"],
                    variant["covered_24"], variant["finish"], " | ".join(map(str, variant["trade_months"])),
                ])
            writer.writerow(line)
    temp.replace(OUTPUT)
    print("Reproduction des 4 607 séquences de référence et vérification du taux du linéaire sur chaque départ.", flush=True)
    print(OUTPUT.name, flush=True)


def main():
    begin = perf_counter()
    dates, rates, group, fits = setup()
    with ProcessPoolExecutor(max_workers=6) as workers:
        cache = WORK/"canonical.npz"
        if cache.exists():
            with np.load(cache) as saved:
                assert np.array_equal(saved["rates"], rates)
                modern = saved["thresholds"]
        else:
            jobs = [(j, rates[j], fits[j-CALIBRATION]) for j in range(CALIBRATION,len(rates)-1)
                    if rates[j]>rates[j-LOOKBACK]]
            modern = np.full((len(rates), HORIZON+1, MAX_TICKETS+1), np.nan)
            futures = [workers.submit(canonical, job) for job in jobs]
            for i, completed in enumerate(as_completed(futures),1):
                j, limits = completed.result()
                modern[j] = limits
                if i % 48 == 0 or i == len(jobs):
                    print(f"Seuils GARCH mensuels {i}/{len(jobs)} en {perf_counter()-begin:.1f} s.", flush=True)
            np.savez_compressed(cache, rates=rates, thresholds=modern)

        jobs = [(j, np.asarray([r["spot_bp"] for r in rows]), rates, modern[j:j+HORIZON])
                for j, rows in group.items()]
        futures = [workers.submit(cohort, job) for job in jobs]
        results = {}
        complete = 0
        for n, completed in enumerate(as_completed(futures), 1):
            j, values = completed.result()
            results[j] = values
            complete += len(values)
            if n % 36 == 0 or n == len(jobs):
                print(f"{complete}/4607 départs rejoués en {perf_counter()-begin:.1f} s.", flush=True)
    launches = combine(results, group, dates, rates)
    write_outputs(launches)


if __name__ == "__main__":
    main()

