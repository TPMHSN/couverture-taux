"""Paired replay of proportional hedge amounts using the 200 M€ decision signal."""

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
sys.path.insert(0, str(ROOT/"_eusa10-variante-1md-work"))
from backtest_sizing import (
    CALIBRATION, GARCH_SHOCKS, GARCH_WEIGHTS, HORIZON, LOOKBACK, ORIGINAL_SHOCKS,
    ORIGINAL_WEIGHTS, RATE_GRID, calibrate_ar1, constant_expectation, decimal, fit_garch,
    garch_grid, setup,
)
from garch_blocks import garch_thresholds

TOTAL = 1000
UNIT = 50
INVENTORY = TOTAL // UNIT
REFERENCE_SIZE = 200 // UNIT
MAX_SIZE = 300 // UNIT
SCALES = (10, 25, 50)
POLICIES = (0, *SCALES)
BASELINE = ROOT / "backtest-eusa10-hybride-1md-comparaison-tickets.csv"
OUTPUT = ROOT / "backtest-eusa10-1md-tickets-proportionnels.csv"


def const_thresholds(a, phi, sigma, nodes, weights, read, last=HORIZON-1):
    """Quote the break-even rate of a 200m block or the smaller remaining balance."""
    expectation = constant_expectation(a, phi, sigma, nodes, weights)
    last_mean = a + phi * RATE_GRID
    out = np.full((HORIZON+1, INVENTORY+1), np.nan)
    next_high = REFERENCE_SIZE
    correction = np.arange(next_high+1)[:,None]*np.minimum(0, RATE_GRID-last_mean)
    next_values = np.vstack((last_mean, correction))
    if HORIZON-1 <= last:
        out[HORIZON-1,1:] = read(HORIZON-1, np.broadcast_to(last_mean, (INVENTORY,len(last_mean))))

    for month in range(HORIZON-2, 0, -1):
        high = min(INVENTORY, REFERENCE_SIZE*(HORIZON-month))
        k = np.arange(1, high+1)
        qty = np.minimum(k, REFERENCE_SIZE)
        future = expectation(next_values)
        mean = future[0]
        wait = future[1+np.minimum(k,next_high)]
        trade = future[1+np.minimum(k-qty,next_high)]
        if month <= last:
            margins = np.vstack((mean+(wait-trade)/qty[:,None],
                                 np.broadcast_to(mean,(INVENTORY-high,len(mean)))))
            out[month,1:] = read(month, margins)
        savings = np.minimum(wait, qty[:,None]*(RATE_GRID-mean)+trade)
        next_values = np.vstack((mean, np.zeros_like(mean), savings))
        next_high = high
    return out


def garch(a, phi, omega, alpha, beta, variance, spot, last=HORIZON-1):
    grid = garch_grid(a, phi, omega, alpha, beta, variance, spot)
    return garch_thresholds(*grid, INVENTORY, REFERENCE_SIZE, last=last)


def canonical(job):
    j, spot, pars = job
    a, phi, omega, alpha, beta, sigma = (
        pars[name] for name in ("a_bp","phi","omega_bp2","alpha","beta","sigma_bp")
    )
    if alpha == beta == 0:
        def read(_month, margins):
            return [np.interp(spot, RATE_GRID, row) for row in margins]
        limits = const_thresholds(a,phi,sigma,GARCH_SHOCKS,GARCH_WEIGHTS,read)
    else:
        limits = garch(a,phi,omega,alpha,beta,sigma**2,spot)
    return j, limits


def quote_quantity(balance, margin, scale):
    if margin <= 0:
        return 0
    if scale == 0:
        return min(balance, REFERENCE_SIZE)
    lots = min(MAX_SIZE, 1+int((MAX_SIZE-1)*margin/scale))
    return min(balance, lots)


def allocations(path, lag, modern, legacy, initial, scale):
    amounts = np.zeros(HORIZON, dtype=np.int64)
    amounts[0] = initial
    balance = INVENTORY - initial
    for i in range(1,HORIZON-1):
        if balance == 0:
            break
        limit = modern[i,i+1,balance] if path[i]>lag[i] else legacy[i+1,balance]
        assert np.isfinite(limit)
        quantity = quote_quantity(balance, limit-path[i], scale)
        amounts[i] = quantity
        balance -= quantity
    amounts[-1] = balance
    assert amounts.sum() == INVENTORY and 0 <= amounts[:-1].min() <= amounts[:-1].max() <= MAX_SIZE
    return amounts


def cohort(job):
    j, prices, rates, modern = job
    history = rates[j-CALIBRATION:j]
    path = rates[j:j+HORIZON]
    lag = rates[j-LOOKBACK:j+HORIZON-LOOKBACK]
    up = prices > lag[0]
    a, phi, sigma = calibrate_ar1(history)
    initial = np.full(len(prices), np.nan)
    legacy = None
    if (~up).any() or (path[1:-1] <= lag[1:-1]).any():
        def read_legacy(month, margins):
            if month == 1:
                initial[~up] = np.interp(prices[~up], RATE_GRID, margins[-1])
            return [np.interp(path[month-1], RATE_GRID, row) for row in margins]
        legacy = const_thresholds(a,phi,sigma,ORIGINAL_SHOCKS,ORIGINAL_WEIGHTS,read_legacy)
    if up[0]:
        initial[0] = modern[0,1,INVENTORY]

    up_indices = np.flatnonzero(up[1:]) + 1
    if len(up_indices):
        residuals = history[1:] - a - phi*history[:-1]
        omega, alpha, beta, prev, _ = fit_garch(residuals)
        if alpha == beta == 0:
            def read_const(_m, margins):
                initial[up_indices] = np.interp(prices[up_indices], RATE_GRID, margins[-1])
                return margins[:,0]
            const_thresholds(a,phi,sqrt(prev),GARCH_SHOCKS,GARCH_WEIGHTS,read_const,last=1)
        else:
            for idx in up_indices:
                spot = float(prices[idx])
                innovation = spot-a-phi*history[-1]
                variance = omega+alpha*innovation*innovation+beta*prev
                # The launch balance is a multiple of 200m for all the policies.
                # Its continuation only needs the five original block counts.
                grid = garch_grid(a, phi, omega, alpha, beta, variance, spot)
                blocks = TOTAL // 200
                initial[idx] = garch_thresholds(*grid, blocks, 1, last=1)[1, blocks]
    assert np.isfinite(initial).all()

    order_plans = {}
    rows = []
    for spot,limit in zip(prices,initial,strict=True):
        actual = np.r_[spot,path[1:]]
        variants = {}
        for scale in POLICIES:
            quantity = quote_quantity(INVENTORY,limit-spot,scale)
            key = scale, quantity
            if key not in order_plans:
                order_plans[key] = allocations(path,lag,modern,legacy,quantity,scale)
            orders = order_plans[key]
            average = float(orders @ actual / INVENTORY)
            variants[scale] = average, orders
        rows.append((float(actual.mean()), float(limit-spot), variants))
    return j, rows


def collect(computed, group):
    with BASELINE.open(encoding="utf-8-sig",newline="") as stream:
        frozen = {r["Date de lancement"]:r for r in csv.DictReader(stream,delimiter=";")}
    all_rows = []
    for j, rows in sorted(computed.items()):
        for initial, result in zip(group[j], rows, strict=True):
            day = initial["Date de lancement"]
            old = frozen.pop(day)
            linear, initial_margin, variants = result
            assert abs(linear-100*decimal(old["Taux linéaire (%)"])) < 1e-8
            baseline_avg, baseline_orders = variants[0]
            assert abs(baseline_avg-100*decimal(old["Taux hybride 200 M€ (%)"])) < 1e-8
            baseline_months = np.repeat(np.arange(1,HORIZON+1),baseline_orders//REFERENCE_SIZE)
            assert " | ".join(map(str,baseline_months)) == old["Mois des tickets 200 M€"], day
            assert baseline_orders[-1]*UNIT == int(old["Solde M60 200 M€ (M€)"])
            all_rows.append(dict(
                date=day,end=old["Dernier fixing"],monthly=old["Départ au 1er jour TARGET"]=="oui",
                linear=linear,margin_first=initial_margin,variants=variants,
            ))
    assert not frozen and len(all_rows)==4607 and sum(r["monthly"] for r in all_rows)==216
    return all_rows


def describe(rows, scale):
    linear = np.asarray([r["linear"] for r in rows])
    rates = np.asarray([r["variants"][scale][0] for r in rows])
    base = np.asarray([r["variants"][0][0] for r in rows])
    gains = linear-rates
    difference = base-rates
    amounts = np.asarray([r["variants"][scale][1] for r in rows],dtype=int)*UNIT
    terminal = amounts[:,-1]
    idx_worst, idx_best = int(np.argmin(gains)), int(np.argmax(gains))
    counts = np.bincount(amounts[:,:-1].ravel()//UNIT,minlength=MAX_SIZE+1)
    return dict(
        n=len(rows),mean=float(gains.mean()),median=float(np.median(gains)),sd=float(gains.std(ddof=1)),
        wins=int((gains>1e-8).sum()),losses=int((gains< -1e-8).sum()),ties=int((abs(gains)<=1e-8).sum()),
        p01=float(np.quantile(gains,.01)),p05=float(np.quantile(gains,.05)),p25=float(np.quantile(gains,.25)),
        p75=float(np.quantile(gains,.75)),p95=float(np.quantile(gains,.95)),p99=float(np.quantile(gains,.99)),
        worst=float(gains[idx_worst]),worst_date=rows[idx_worst]["date"],
        best=float(gains[idx_best]),best_date=rows[idx_best]["date"],
        delta_mean=float(difference.mean()),delta_median=float(np.median(difference)),
        better=int((difference>1e-8).sum()),worse=int((difference< -1e-8).sum()),
        terminal_mean=float(terminal.mean()),terminal_median=float(np.median(terminal)),
        terminal_p95=float(np.quantile(terminal,.95)),terminal_max=int(terminal.max()),
        terminal_800=int((terminal>=800).sum()),terminal_full=int((terminal==1000).sum()),
        mean_covered_m12=float(amounts[:,:12].sum(axis=1).mean()),mean_covered_m24=float(amounts[:,:24].sum(axis=1).mean()),
        voluntary_order_counts={UNIT*i:int(counts[i]) for i in range(1,MAX_SIZE+1)},
    )


def report(rows):
    cohorts = {
        "all": rows,
        "monthly": [r for r in rows if r["monthly"]],
        "2004-2014": [r for r in rows if r["date"]<"2015"],
        "2015-2016": [r for r in rows if "2015"<=r["date"]<"2017"],
        "2017-2019": [r for r in rows if "2017"<=r["date"]<"2020"],
        "2020-2021": [r for r in rows if "2020"<=r["date"]<"2022"],
    }
    stats = {subset:{h:describe(items,h) for h in POLICIES} for subset,items in cohorts.items()}
    (WORK/"summary.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    header = ["Date de lancement","Dernier fixing","Départ au 1er jour TARGET",
              "Taux linéaire (%)","Marge au premier fixing (pb)"]
    for h in POLICIES:
        label = "Fixe 200 M€" if h==0 else f"300 M€ à +{h} pb"
        header.extend(f"{label} · {field}" for field in (
            "Taux (%)","Gain (pb)","Écart de gain au fixe 200 M€ (pb)","Solde M60 (M€)",
            "Couvert à M24 (M€)","Exécutions mois:montant en M€",
        ))
    def fr(v):
        return f"{v:.10f}".replace(".",",")
    temp = OUTPUT.with_suffix(".pending")
    with temp.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.writer(stream,delimiter=";")
        writer.writerow(header)
        for r in rows:
            line = [r["date"],r["end"],"oui" if r["monthly"] else "non",fr(r["linear"]/100),fr(r["margin_first"])]
            baseline = r["variants"][0][0]
            for h in POLICIES:
                rate, orders = r["variants"][h]
                line.extend([
                    fr(rate/100),fr(r["linear"]-rate),fr(baseline-rate),int(orders[-1])*UNIT,
                    int(orders[:24].sum())*UNIT," | ".join(f"{m}:{int(q)*UNIT}" for m,q in enumerate(orders,1) if q),
                ])
            writer.writerow(line)
    temp.replace(OUTPUT)
    print("Les 4 607 séquences 200 M€ correspondent à la référence.",flush=True)
    print(OUTPUT.name,flush=True)


def run():
    start = perf_counter()
    dates, rates, group, fits = setup()
    with np.load(ROOT/"_eusa10-comparaison-tickets-work/canonical.npz") as prior:
        assert np.array_equal(rates,prior["rates"])
        previous = prior["thresholds"][:,:,1:6]
    cache = WORK/"canonical-blocks.npz"
    with ProcessPoolExecutor(max_workers=6) as pool:
        if cache.exists():
            with np.load(cache) as saved:
                assert np.array_equal(rates,saved["rates"])
                modern = saved["thresholds"]
        else:
            modern = np.full((len(dates),HORIZON+1,INVENTORY+1),np.nan)
            jobs = [(j,rates[j],fits[j-CALIBRATION]) for j in range(CALIBRATION,len(dates)-1)
                    if rates[j]>rates[j-LOOKBACK]]
            futures = [pool.submit(canonical,job) for job in jobs]
            for n, future in enumerate(as_completed(futures),1):
                j, limits = future.result()
                modern[j] = limits
                if n%48==0 or n==len(futures):
                    print(f"Seuils de début de mois {n}/{len(futures)} ({perf_counter()-start:.1f} s).",flush=True)
            np.savez_compressed(cache,thresholds=modern,rates=rates)
        supported = np.isfinite(previous)
        gap = np.max(abs(modern[:,:,REFERENCE_SIZE::REFERENCE_SIZE][supported]-previous[supported]))
        assert gap < 1e-7, gap
        print(f"Écart maximal sur les seuils GARCH des multiples de 200 M€ : {gap:.3g} pb.",flush=True)
        jobs = [(j,np.asarray([r["spot_bp"] for r in rows]),rates,modern[j:j+HORIZON]) for j,rows in group.items()]
        futures = [pool.submit(cohort,job) for job in jobs]
        outputs = {}
        for n, future in enumerate(as_completed(futures),1):
            j, replay = future.result()
            outputs[j] = replay
            if n%36==0 or n==len(futures):
                print(f"{sum(map(len,outputs.values()))}/4607 départs rejoués ({perf_counter()-start:.1f} s).",flush=True)
    report(collect(outputs,group))


if __name__=="__main__":
    run()


