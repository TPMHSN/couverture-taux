"""Test optional 50m complements when the quote is low versus the prior year."""

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import numpy as np

from replay_small import (
    BASELINE, CALIBRATION, CAP, FIRST_MARGIN, HORIZON, INVENTORY, LOOKBACK,
    ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS, OUTPUT, RATE_GRID, ROOT, TOTAL, UNIT, WORK, 
    calibrate_ar1, decimal, fixed_constant, load_file, setup, summarize,
)
from signal_complement import ordre_tickets


GATES = (25,37)
LABELS = {m:f"50 M€ en bas de fourchette à partir de M{m}" for m in GATES}
FINAL = ROOT.parent / "sorties/eusa10/backtest-eusa10-complement-50m.csv"


def allocate(gate, path, lag, empirical, block, old, first):
    q = np.zeros(HORIZON,dtype=np.int64)
    extra = np.zeros(HORIZON,dtype=bool)
    balance = INVENTORY-first
    q[0] = first
    for i in range(1,HORIZON-1):
        if balance == 0:
            break
        month,spot = i+1,path[i]
        limit = block[i,month,balance] if spot>lag[i] else old[month,balance]
        assert np.isfinite(limit)
        q[i], motif = ordre_tickets(month, balance, spot, limit, empirical[i], gate)
        extra[i] = motif == "complément"
        balance-=q[i]
    q[-1]=balance
    assert q.sum()==INVENTORY
    return q,extra


def replay(job):
    j, prices, initial_margins, rates, block = job
    path=rates[j:j+HORIZON]
    lag=rates[j-LOOKBACK:j+HORIZON-LOOKBACK]
    empirical = np.array([np.quantile(rates[j+i-12:j+i],.25) for i in range(HORIZON)])
    old=None
    if (path[1:-1]<=lag[1:-1]).any():
        a,phi,sigma=calibrate_ar1(rates[j-CALIBRATION:j])
        def read(month,values):
            return [np.interp(path[month-1],RATE_GRID,row) for row in values]
        old=fixed_constant(a,phi,sigma,ORIGINAL_SHOCKS,ORIGINAL_WEIGHTS,read)
    planned={}
    batch=[]
    for spot,init in zip(prices,initial_margins,strict=True):
        actual=np.r_[spot,path[1:]]
        first=CAP if init>0 else 0
        if first not in planned:
            planned[first]={gate:allocate(gate,path,lag,empirical,block,old,first) for gate in GATES}
        cases={}
        for gate,(q,extra) in planned[first].items():
            cases[gate]=float(q@actual/INVENTORY),q,extra
        batch.append((float(actual.mean()),cases))
    return j,batch


def report(outputs,group,base,fix):
    rows=[]
    extensions=["Gain (pb)","Taux (%)","Solde M60 (M€)","Exécutions mois:montant en M€","Mois des compléments"]
    additions=[f"{LABELS[gate]} · {c}" for gate in GATES for c in extensions]
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    temp=FINAL.with_suffix(".pending")
    def fr(v):
        return f"{v:.10f}".replace(".",",")
    with temp.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f,delimiter=";")
        keys=list(next(iter(base.values())).keys())
        writer.writerow([*keys,*additions])
        for j,batch in sorted(outputs.items()):
            for original,(linear,cases) in zip(group[j],batch,strict=True):
                day=original["Date de lancement"]
                r=base[day]
                frozen=fix[day]
                assert abs(linear-100*decimal(r["Taux linéaire (%)"]))<1e-8
                start_months=[int(m) for m in frozen["Mois des tickets 200 M€"].split(" | ")]
                seq=CAP*np.bincount(start_months,minlength=HORIZON+1)[1:]
                cases["fixed"]=100*decimal(frozen["Taux hybride 200 M€ (%)"]),seq,np.zeros(HORIZON,dtype=bool)
                info=[r[k] for k in keys]
                for gate in GATES:
                    rate,q,add=cases[gate]
                    info.extend([fr(linear-rate),fr(rate/100),int(q[-1])*UNIT,
                                 " | ".join(f"{i+1}:{int(n)*UNIT}" for i,n in enumerate(q) if n),
                                 " | ".join(str(i+1) for i in np.flatnonzero(add))])
                writer.writerow(info)
                rows.append(dict(date=day,linear=linear,monthly=frozen["Départ au 1er jour TARGET"]=="oui",variants=cases))
    assert len(rows)==4607 and sum(r["monthly"] for r in rows)==216
    subsets={"all":rows,"monthly":[r for r in rows if r["monthly"]],
             "2004-2014":[r for r in rows if r["date"]<"2015"],
             "2017-2019":[r for r in rows if "2017"<=r["date"]<"2020"],
             "2020-2021":[r for r in rows if "2020"<=r["date"]<"2022"]}
    result={name:{m:summarize(seq,m) for m in GATES} for name,seq in subsets.items()}
    (FINAL.parent/"statistiques-complement-50m.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    temp.replace(FINAL)
    print(FINAL.name,flush=True)


def main():
    start=perf_counter()
    _,rates,group,_=setup()
    base = load_file(OUTPUT)
    fix = load_file(BASELINE)
    initial = load_file(FIRST_MARGIN)
    with np.load(ROOT/"_eusa10-tickets-proportionnels-work/canonical-blocks.npz") as cached:
        assert np.array_equal(rates,cached["rates"])
        block=cached["thresholds"]
    jobs = [(j,np.asarray([r["spot_bp"] for r in seq]),
             [decimal(initial[r["Date de lancement"]]["Marge au premier fixing (pb)"]) for r in seq],
             rates,block[j:j+HORIZON]) for j,seq in group.items()]
    outputs={}
    with ProcessPoolExecutor(max_workers=6) as workers:
        tasks=[workers.submit(replay,job) for job in jobs]
        for n,future in enumerate(as_completed(tasks),1):
            j,batch=future.result()
            outputs[j]=batch
            if n%72==0 or n==len(tasks):
                print(f"Filtre des taux observés · {n}/{len(tasks)} cohortes en {perf_counter()-start:.1f} s.",flush=True)
    report(outputs,group,base,fix)


if __name__=="__main__":
    main()

