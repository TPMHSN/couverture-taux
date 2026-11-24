"""Test optional smaller EUSA10 orders with the same pricing models and paired starts."""

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
sys.path.insert(0, str(ROOT/"_eusa10-tickets-proportionnels-work"))
from run_proportional import (
    BASELINE, CALIBRATION, GARCH_SHOCKS, GARCH_WEIGHTS, HORIZON, INVENTORY, LOOKBACK,
    ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS, OUTPUT as FIRST_MARGIN, RATE_GRID, REFERENCE_SIZE,
    TOTAL, UNIT, calibrate_ar1, constant_expectation, const_thresholds as fixed_constant,
    decimal, fit_garch, garch_grid, setup,
)
from garch_flex import garch_thresholds

CAP = REFERENCE_SIZE
LARGE_BALANCE = 800 // UNIT
SHORT_HORIZON = 12
SHORT_AGE = HORIZON-SHORT_HORIZON
POLICIES = ("fixed","flex","small","year")
LABELS = {
    "fixed":"Fixe 200 M€", "flex":"50 à 200 M€",
    "small":"200 M€ + petits ordres sur horizon restant", "year":"200 M€ + 50 M€ sur 12 mois",
}
OUTPUT = ROOT/"backtest-eusa10-1md-petits-tickets.csv"


def constant_flex(a, phi, sigma, nodes, weights, sample, last=HORIZON-1):
    expectation = constant_expectation(a,phi,sigma,nodes,weights)
    mean = a+phi*RATE_GRID
    next_high = CAP
    final = np.zeros_like(RATE_GRID)
    values = np.vstack((mean, np.arange(CAP+1)[:,None]*np.minimum(0,RATE_GRID-mean)))
    output = np.full((HORIZON+1,INVENTORY+1),np.nan)
    if HORIZON-1<=last:
        output[HORIZON-1,1:] = sample(HORIZON-1,np.broadcast_to(mean,(INVENTORY,len(mean))))

    for month in range(HORIZON-2,0,-1):
        high = min(INVENTORY,CAP*(HORIZON-month))
        balances = np.arange(1,high+1)
        future = expectation(values)
        mean = future[0]
        delta = RATE_GRID-mean
        wait = future[1+np.minimum(balances,next_high)]
        if month<=last:
            one_less = future[1+np.minimum(balances-1,next_high)]
            marginal = np.vstack((mean+wait-one_less,np.broadcast_to(mean,(INVENTORY-high,len(mean)))))
            output[month,1:] = sample(month,marginal)
        optimal = wait.copy()
        for quantity in range(1,CAP+1):
            left_after = balances[quantity-1:]-quantity
            future_after = future[1+np.minimum(left_after,next_high)]
            target = optimal[quantity-1:]
            np.minimum(target,quantity*delta+future_after,out=target)
        values = np.vstack((mean,final,optimal))
        next_high = high
    return output


def solve_garch(a,phi,omega,alpha,beta,variance,spot,last=HORIZON-1):
    grid = garch_grid(a,phi,omega,alpha,beta,variance,spot)
    return garch_thresholds(*grid,INVENTORY,CAP,last=last)


def canonical(job):
    j, spot, pars = job
    a,phi,omega,alpha,beta,sigma = (
        pars[key] for key in ("a_bp","phi","omega_bp2","alpha","beta","sigma_bp")
    )
    if alpha==beta==0:
        def at_spot(_month, values):
            return [np.interp(spot,RATE_GRID,row) for row in values]
        values = constant_flex(a,phi,sigma,GARCH_SHOCKS,GARCH_WEIGHTS,at_spot)
    else:
        values = solve_garch(a,phi,omega,alpha,beta,sigma**2,spot)
    return j, values


def flexible_quantity(balance, spot, curve, max_order):
    n = min(balance,max_order)
    benefit = np.cumsum(curve[balance:balance-n:-1]-spot)
    best = int(np.argmax(benefit))
    if benefit[best] <= 1e-8:
        return 0, 0.
    return best+1, float(benefit[best]/(best+1))


def choose(rule, balance, spot, fixed_margin, flex_curve, short_margin):
    if rule=="flex":
        amount, _ = flexible_quantity(balance,spot,flex_curve,CAP)
        return amount, False
    if fixed_margin>0:
        return min(balance,CAP), False
    if balance<LARGE_BALANCE or rule=="fixed":
        return 0, False
    if rule=="small":
        amount, _ = flexible_quantity(balance,spot,flex_curve,CAP-1)
        return amount, amount>0
    if rule=="year" and short_margin>1e-8:
        return 1, True
    return 0, False


def portfolio(rule, path, lag, garch, old, fixed_garch, fixed_legacy, short_legacy, first, first_extra):
    orders = np.zeros(HORIZON, dtype=np.int64)
    extra = np.zeros(HORIZON,dtype=bool)
    orders[0], extra[0] = first, first_extra
    balance = INVENTORY-first
    for i in range(1,HORIZON-1):
        if balance==0:
            break
        m, spot = i+1, path[i]
        if spot>lag[i]:
            fixed_margin = fixed_garch[i,m,balance]-spot
            curve = garch[i,m,:]
            short_margin = garch[i,max(SHORT_AGE,m),1]-spot
        else:
            fixed_margin = fixed_legacy[m,balance]-spot
            curve = old[m,:]
            short_margin = (short_legacy[i] if m<SHORT_AGE else old[m,1])-spot
        assert np.isfinite(fixed_margin) and np.isfinite(short_margin) and np.isfinite(curve[1:]).all()
        amount, flag = choose(rule,balance,spot,fixed_margin,curve,short_margin)
        orders[i],extra[i] = amount,flag
        balance -= amount
    orders[-1] = balance
    assert orders.sum()==INVENTORY and np.all((orders[:-1]>=0)&(orders[:-1]<=CAP))
    return orders,extra


def cohort(job):
    j, prices, initial_fixed, rates, garch, core = job
    path = rates[j:j+HORIZON]
    lag = rates[j-LOOKBACK:j+HORIZON-LOOKBACK]
    past = rates[j-CALIBRATION:j]
    up = prices>lag[0]
    a,phi,sigma = calibrate_ar1(past)
    initial = np.full((len(prices),INVENTORY+1),np.nan)
    initial_short = np.full(len(prices),np.nan)
    short_legacy = np.full(SHORT_AGE,np.nan)
    old = fixed = None

    if (~up).any() or (path[1:-1]<=lag[1:-1]).any():
        def sample(month, values):
            if month==1:
                for k,row in enumerate(values,1):
                    initial[~up,k] = np.interp(prices[~up],RATE_GRID,row)
            if month==SHORT_AGE:
                short_legacy[:] = np.interp(path[:SHORT_AGE],RATE_GRID,values[0])
                initial_short[~up] = np.interp(prices[~up],RATE_GRID,values[0])
            return [np.interp(path[month-1],RATE_GRID,row) for row in values]
        old = constant_flex(a,phi,sigma,ORIGINAL_SHOCKS,ORIGINAL_WEIGHTS,sample)
        def sample_fixed(month,values):
            return [np.interp(path[month-1],RATE_GRID,row) for row in values]
        fixed = fixed_constant(a,phi,sigma,ORIGINAL_SHOCKS,ORIGINAL_WEIGHTS,sample_fixed)
    if up[0]:
        initial[0,:] = garch[0,1,:]
        initial_short[0] = garch[0,SHORT_AGE,1]

    off_grid = np.flatnonzero(up[1:])+1
    if len(off_grid):
        residuals = past[1:]-a-phi*past[:-1]
        omega,alpha,beta,past_variance,_ = fit_garch(residuals)
        if alpha==beta==0:
            def sample_const(month, values):
                if month==1:
                    for k,row in enumerate(values,1):
                        initial[off_grid,k] = np.interp(prices[off_grid],RATE_GRID,row)
                if month==SHORT_AGE:
                    initial_short[off_grid] = np.interp(prices[off_grid],RATE_GRID,values[0])
                return values[:,0]
            constant_flex(a,phi,sqrt(past_variance),GARCH_SHOCKS,GARCH_WEIGHTS,sample_const,last=SHORT_AGE)
        else:
            for idx in off_grid:
                spot = float(prices[idx])
                innovation = spot-a-phi*past[-1]
                variance = omega+alpha*innovation*innovation+beta*past_variance
                value = solve_garch(a,phi,omega,alpha,beta,variance,spot,last=SHORT_AGE)
                initial[idx,:] = value[1,:]
                initial_short[idx] = value[SHORT_AGE,1]
    assert np.isfinite(initial[:,1:]).all() and np.isfinite(initial_short).all()

    memo = {}
    results = []
    for i,spot in enumerate(prices):
        actual = np.r_[spot,path[1:]]
        linear = float(actual.mean())
        variants = {}
        for rule in POLICIES:
            order, flag = choose(rule,INVENTORY,spot,initial_fixed[i],initial[i,:],initial_short[i]-spot)
            key = rule,order,flag
            if key not in memo:
                memo[key] = portfolio(rule,path,lag,garch,old,core,fixed,short_legacy,order,flag)
            quantities,extra = memo[key]
            average = float(quantities @ actual/INVENTORY)
            variants[rule] = average, quantities, extra
        results.append((linear,variants))
    return j,results


def load_file(file):
    with file.open(encoding="utf-8-sig",newline="") as f:
        return {r["Date de lancement"]:r for r in csv.DictReader(f,delimiter=";")}


def collect(result, group, baseline):
    rows = []
    for j,batch in sorted(result.items()):
        for r,(linear,variants) in zip(group[j],batch,strict=True):
            date = r["Date de lancement"]
            fixed = baseline[date]
            rate,orders,_ = variants["fixed"]
            months = np.repeat(np.arange(1,HORIZON+1),orders//CAP)
            assert " | ".join(map(str,months)) == fixed["Mois des tickets 200 M€"], date
            assert abs(rate-100*decimal(fixed["Taux hybride 200 M€ (%)"]))<1e-8
            assert abs(linear-100*decimal(fixed["Taux linéaire (%)"]))<1e-8
            rows.append(dict(date=date,end=fixed["Dernier fixing"],linear=linear,variants=variants,
                             monthly=fixed["Départ au 1er jour TARGET"]=="oui"))
    return rows


def summarize(rows, rule):
    base = np.array([r["variants"]["fixed"][1][-1] for r in rows])*UNIT
    quantities = np.array([r["variants"][rule][1] for r in rows])*UNIT
    extras = np.array([r["variants"][rule][2] for r in rows])
    gain = np.asarray([r["linear"]-r["variants"][rule][0] for r in rows])
    delta = np.asarray([r["variants"]["fixed"][0]-r["variants"][rule][0] for r in rows])
    m60 = quantities[:,-1]
    is_large = m60>=800
    shock = gain+100*(1/60-m60/TOTAL)
    return dict(
        n=len(rows),mean=float(gain.mean()),median=float(np.median(gain)),
        p05=float(np.quantile(gain,.05)),worst=float(gain.min()),
        wins=int((gain>1e-8).sum()),losses=int((gain< -1e-8).sum()),
        delta=float(delta.mean()),beats_reference=int((delta>1e-8).sum()),
        m60_count_800=int(is_large.sum()),m60_full=int((m60==1000).sum()),m60_p95=float(np.quantile(m60,.95)),
        m60_mean=float(m60.mean()),m60_median=float(np.median(m60)),m60_some=int((m60>0).sum()),
        fixings_more_800=int(((base<800)&is_large).sum()),fixings_removed_800=int(((base>=800)&(~is_large)).sum()),
        on_base_large_delta=float(delta[base>=800].mean()) if (base>=800).any() else None,
        smaller_orders=int(((quantities[:,:-1]>0)&(quantities[:,:-1]<200)).sum()),
        first_programmes_extra=int(extras.any(axis=1).sum()),extra_orders=int(extras.sum()),
        mean_extra_notional=float(np.where(extras,quantities,0).sum(axis=1).mean()),
        shock100_mean=float(shock.mean()),shock100_p05=float(np.quantile(shock,.05)),
        shock100_worst=float(shock.min()),shock100_losses=int((shock< -1e-8).sum()),
    )


def report(rows):
    assert len(rows)==4607 and sum(r["monthly"] for r in rows)==216
    groups={"all":rows,"monthly":[r for r in rows if r["monthly"]],
            "2004-2014":[r for r in rows if r["date"]<"2015"],
            "2017-2019":[r for r in rows if "2017"<=r["date"]<"2020"],
            "2020-2021":[r for r in rows if "2020"<=r["date"]<"2022"]}
    stats={group:{rule:summarize(seq,rule) for rule in POLICIES} for group,seq in groups.items()}
    (WORK/"summary.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    columns = ["Date de lancement","Dernier fixing","Taux linéaire (%)"]
    for rule in POLICIES:
        columns.extend(LABELS[rule]+" · "+field for field in [
            "Gain (pb)","Taux (%)","Solde M60 (M€)","Exécutions mois:montant en M€","Mois des compléments",
        ])
    def fr(v):
        return f"{v:.10f}".replace(".",",")
    pending = OUTPUT.with_suffix(".pending")
    with pending.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f,delimiter=";")
        writer.writerow(columns)
        for r in rows:
            line=[r["date"],r["end"],fr(r["linear"]/100)]
            for rule in POLICIES:
                rate,q,extra = r["variants"][rule]
                line.extend([fr(r["linear"]-rate),fr(rate/100),int(q[-1])*UNIT,
                             " | ".join(f"{k}:{int(n)*UNIT}" for k,n in enumerate(q,1) if n),
                             " | ".join(str(i+1) for i in np.flatnonzero(extra))])
            writer.writerow(line)
    pending.replace(OUTPUT)
    print("Les 4 607 séquences du fixe ont été reproduites. "+OUTPUT.name,flush=True)


def main():
    start=perf_counter()
    dates,rates,group,fits=setup()
    baseline=load_file(BASELINE)
    base_margins = load_file(FIRST_MARGIN)
    with np.load(ROOT/"_eusa10-tickets-proportionnels-work/canonical-blocks.npz") as cached:
        assert np.array_equal(cached["rates"],rates)
        block=cached["thresholds"]
    path=WORK/"flexible.npz"
    with ProcessPoolExecutor(max_workers=6) as executor:
        if path.exists():
            with np.load(path) as f:
                assert np.array_equal(f["rates"],rates)
                current=f["thresholds"]
        else:
            current=np.full((len(rates),HORIZON+1,INVENTORY+1),np.nan)
            task=[(j,rates[j],fits[j-CALIBRATION]) for j in range(CALIBRATION,len(rates)-1)
                  if rates[j]>rates[j-LOOKBACK]]
            futures=[executor.submit(canonical,job) for job in task]
            for n,future in enumerate(as_completed(futures),1):
                j,curves=future.result()
                current[j]=curves
                if n%48==0 or n==len(futures):
                    print(f"Seuils flexibles mensuels {n}/{len(futures)} en {perf_counter()-start:.1f} s.",flush=True)
            np.savez_compressed(path,rates=rates,thresholds=current)

        samples=[(j,np.asarray([r["spot_bp"] for r in seq]),
                  np.asarray([decimal(base_margins[r["Date de lancement"]]["Marge au premier fixing (pb)"]) for r in seq]),
                  rates,current[j:j+HORIZON],block[j:j+HORIZON]) for j,seq in group.items()]
        futures=[executor.submit(cohort,job) for job in samples]
        results={}
        for n,future in enumerate(as_completed(futures),1):
            j,replay=future.result()
            results[j]=replay
            if n%36==0 or n==len(futures):
                print(f"{sum(map(len,results.values()))}/4607 départs rejoués en {perf_counter()-start:.1f} s.",flush=True)
    report(collect(results,group,baseline))


if __name__=="__main__":
    main()

