"""Transfer the existing hybrid tree to USD high/low midpoints on a short history."""
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
REPRO = WORK
sys.path.insert(0, str(REPRO/"_eusa10-opportunites-50-200-work"))
from signal_complement import ordre_tickets
from run_proportional import (
    CALIBRATION as ORIGINAL_WINDOW, GARCH_SHOCKS, GARCH_WEIGHTS, HORIZON, INVENTORY,
    LOOKBACK, ORIGINAL_SHOCKS, ORIGINAL_WEIGHTS, RATE_GRID, TOTAL, UNIT, calibrate_ar1,
    const_thresholds, fit_garch, garch,
)

WINDOW = 36
OUTPUT = ROOT/"sorties/usdsb3l10y"
MIDS_CSV = OUTPUT/"milieux.csv"
RESULT_CSV = OUTPUT/"departs.csv"


def decimal_fr(value):
    return Decimal(value.strip().replace(" ", "").replace(",", "."))


def french(value):
    return str(value).replace(".", ",")


def read_quotes(source):
    data = []
    with source.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        reader.fieldnames = [s.strip() for s in reader.fieldnames]
        for row in reader:
            day=datetime.strptime(row["Date"],"%d/%m/%Y").date()
            high,low=decimal_fr(row["Plus Haut"]),decimal_fr(row["Plus Bas"])
            if not (high.is_finite() and low.is_finite() and high>=low):
                raise ValueError(f"Fourchette invalide le {day}.")
            mid=(high+low)/2
            data.append((day,high,low,mid))
    data.sort(key=lambda entry: entry[0])
    assert len({d for d,_,_,_ in data})==len(data)
    return data


def export_mids(data):
    with MIDS_CSV.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f,delimiter=";")
        writer.writerow(["Date","Plus haut (%)","Plus bas (%)","Milieu haut-bas (%)","En semaine"])
        for day,high,low,mid in data:
            writer.writerow([day.isoformat(),french(high),french(low),french(mid),
                             "oui" if day.weekday()<5 else "non"])


def grouped_quotes(data):
    # Midpoints are exported for all dates. Weekdays provide the simulation calendar.
    month_groups=defaultdict(list)
    for day,_,_,mid in data:
        if day.weekday()<5:
            month_groups[day.strftime("%Y-%m")].append((day,float(mid*100)))
    months=list(month_groups)
    period_numbers=[12*int(key[:4])+int(key[-2:]) for key in months]
    assert np.all(np.diff(period_numbers)==1)
    # The export starts during July. The first monthly calibration point is August.
    return [month_groups[month] for month in months[1:]]


def fit_ar1(history):
    """Use the production OLS calculation for the requested history length."""
    if len(history)==ORIGINAL_WINDOW:
        return calibrate_ar1(history)
    x,y=history[:-1],history[1:]
    dx=x-x.mean()
    dispersion=dx@dx
    if dispersion<=0 or not np.isfinite(history).all():
        raise ValueError("Taux insuffisamment variables pour estimer l'AR(1).")
    phi=(dx@(y-y.mean()))/dispersion
    a=y.mean()-phi*x.mean()
    residual=y-a-phi*x
    sigma=np.sqrt((residual@residual)/(len(residual)-2))
    return float(a),float(phi),float(sigma)


def modern_solver(job):
    month_index,spot,age,parameters=job
    a,phi,omega,alpha,beta,variance=parameters
    if alpha==beta==0:
        def read_limit(_age,margins):
            return [np.interp(spot,RATE_GRID,row) for row in margins]
        table=const_thresholds(a,phi,sqrt(variance),GARCH_SHOCKS,GARCH_WEIGHTS,read_limit,last=age)
    else:
        table=garch(a,phi,omega,alpha,beta,variance,spot,last=age)
    assert np.isfinite(table[age,1:]).all()
    return (month_index,spot),table


def garch_parameters(history):
    a,phi,_=fit_ar1(history)
    residuals=history[1:]-a-phi*history[:-1]
    omega,alpha,beta,variance,_=fit_garch(residuals)
    return a,phi,omega,alpha,beta,variance


def modern_tables(starts,groups,rates):
    required={}
    for start in starts:
        for _,spot in groups[start]:
            if spot>rates[start-LOOKBACK]:
                required.setdefault((start,spot),1)
        for fixing_index in range(start+1,start+HORIZON-1):
            spot=rates[fixing_index]
            if spot>rates[fixing_index-LOOKBACK]:
                key=(fixing_index,float(spot))
                required[key]=max(required.get(key,0),fixing_index-start+1)
    fits={j:garch_parameters(rates[j-WINDOW:j]) for j in {x[0] for x in required}}
    jobs=[]
    for (j,spot),age in required.items():
        a,phi,omega,alpha,beta,prev=fits[j]
        innovation=spot-a-phi*rates[j-1]
        variance=omega+alpha*innovation*innovation+beta*prev
        jobs.append((j,spot,age,(a,phi,omega,alpha,beta,variance)))
    print(f"Seuils GARCH requis: {len(jobs)}",flush=True)
    resolved={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(modern_solver,job) for job in jobs]
        for n,future in enumerate(as_completed(futures),1):
            key,solved=future.result()
            resolved[key]=solved
            if n%25==0 or n==len(jobs):
                print(f"Seuils GARCH calculés: {n}/{len(jobs)}",flush=True)
    return resolved


def replay_cohort(start,group,rates,modern):
    anchors=np.array(rates[start:start+HORIZON])
    prior=np.array(rates[start-LOOKBACK:start+HORIZON-LOOKBACK])
    assert len(anchors)==HORIZON
    quartiles=[float(np.quantile(rates[j-12:j],.25)) for j in range(start,start+HORIZON)]
    spots=np.asarray([s for _,s in group[start]])
    fit=fit_ar1(rates[start-WINDOW:start])
    initial_legacy=np.empty(len(spots))

    def read_legacy(m,margins):
        if m==1:
            initial_legacy[:]=np.interp(spots,RATE_GRID,margins[INVENTORY-1])
        return [np.interp(anchors[m-1],RATE_GRID,row) for row in margins]

    legacy=const_thresholds(*fit,ORIGINAL_SHOCKS,ORIGINAL_WEIGHTS,read_legacy)
    assert np.isfinite(initial_legacy).all()
    terminal=group[start+HORIZON-1][0][0]
    rows=[]
    for i,(launch,spot) in enumerate(group[start]):
        path=np.r_[spot,anchors[1:]]
        hedged=np.zeros(HORIZON,dtype=int)
        balance=INVENTORY
        passages=[]
        for j,r in enumerate(path):
            month,j_abs=j+1,start+j
            threshold=None
            if balance and month<HORIZON:
                if r>prior[j]:
                    threshold=modern[(j_abs,float(r))][month,balance]
                elif month==1:
                    threshold=initial_legacy[i]
                else:
                    threshold=legacy[month,balance]
                assert np.isfinite(threshold)
            q,reason=ordre_tickets(month,balance,r,threshold,quartiles[j])
            hedged[j]=q
            balance-=q
            if q:
                passages.append((month,q*UNIT,reason))
        assert balance==0 and hedged.sum()==INVENTORY
        assert max(hedged[:-1])*UNIT<=200
        weighted=float(hedged@path/INVENTORY)
        linear=float(path.mean())
        finish=int(np.flatnonzero(hedged.cumsum()==INVENTORY)[0]+1)
        rows.append(dict(depart=launch.isoformat(),dernier_fixing=terminal.isoformat(),
                         cohorte=launch.strftime("%Y-%m"),taux_ultime_pct=weighted/100,
                         taux_lineaire_pct=linear/100,gain_pb=linear-weighted,
                         solde_m60_musd=int(hedged[-1])*UNIT,mois_soldage=finish,
                         nombre_complements=sum(reason=="complément" for _,_,reason in passages),
                         ordres=" | ".join(f"{m}:{v}" for m,v,_ in passages),
                         premier_fixing_du_mois=(i==0)))
    return rows


def summarize(rows,data,groups,strict):
    gain=np.array([r["gain_pb"] for r in rows])
    terminal=np.array([r["solde_m60_musd"] for r in rows])
    first=[r for r in rows if r["premier_fixing_du_mois"]]
    number_near_zero=int((abs(gain)<=1e-9).sum())
    min_i,max_i=int(np.argmin(gain)),int(np.argmax(gain))
    cohort_count=len(first)
    return dict(cotations=len(data), week_end=sum(day.weekday()>=5 for day,_,_,_ in data),
                debut_data=data[0][0].isoformat(),fin_data=data[-1][0].isoformat(),
                midpoint_last_pct=float(data[-1][3]),month_count=len(groups),
                original_window=ORIGINAL_WINDOW,complete_original=strict,calibration=WINDOW,
                horizon=HORIZON,n=len(rows),cohort_count=cohort_count,overlap_month_min=HORIZON-cohort_count+1,
                first_launch=rows[0]["depart"],last_launch=rows[-1]["depart"],
                mean_gain=float(gain.mean()),median_gain=float(np.median(gain)),
                p05=float(np.quantile(gain,.05)),sample_sd=float(gain.std(ddof=1)),
                earliest_terminal=rows[0]["dernier_fixing"],last_terminal=rows[-1]["dernier_fixing"],
                worst=float(gain[min_i]),worst_start=rows[min_i]["depart"],best=float(gain[max_i]),
                best_start=rows[max_i]["depart"],wins=int((gain>1e-9).sum()),
                losses=int((gain<-1e-9).sum()),near_zero=number_near_zero,
                mean_terminal=float(terminal.mean()),max_terminal=int(terminal.max()),
                p90_terminal=float(np.quantile(terminal,.9)),mean_completion=float(np.mean([r["mois_soldage"] for r in rows])),
                n_complements=int(sum(r["nombre_complements"]>0 for r in rows)),
                first_of_month=first)


def export_results(rows):
    labels=("Départ","Dernier fixing","Mois du départ","Taux Ultime (%)","Taux linéaire (%)",
            "Gain vs linéaire (pb)","Solde au mois 60 (M$)","Mois soldé",
            "Nombre de compléments","Ordres mois:montant (M$)","Premier fixing du mois")
    with RESULT_CSV.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f,delimiter=";")
        writer.writerow(labels)
        for row in rows:
            writer.writerow((row["depart"],row["dernier_fixing"],row["cohorte"],
                             french(round(row["taux_ultime_pct"],9)),french(round(row["taux_lineaire_pct"],9)),
                             french(round(row["gain_pb"],7)),row["solde_m60_musd"],row["mois_soldage"],
                             row["nombre_complements"],row["ordres"],"oui" if row["premier_fixing_du_mois"] else "non"))


def main(source):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data=read_quotes(source)
    export_mids(data)
    groups=grouped_quotes(data)
    anchors=np.asarray([q[0][1] for q in groups])
    strict=sum(len(groups[i]) for i in range(ORIGINAL_WINDOW,len(groups)-HORIZON+1))
    starts=range(WINDOW,len(groups)-HORIZON+1)
    expected=sum(len(groups[i]) for i in starts)
    print(f"Historique: {len(groups)} fixings mensuels exploitables, {strict} programmes avec calibration à 60.",flush=True)
    print(f"Rejeu à {WINDOW} fixings: {expected} départs quotidiens en {len(starts)} mois.",flush=True)
    assert strict==0 and expected>0
    modern=modern_tables(starts,groups,anchors)
    rows=[r for i in starts for r in replay_cohort(i,groups,anchors,modern)]
    assert len(rows)==len({r["depart"] for r in rows})==expected
    report=summarize(rows,data,groups,strict)
    export_results(rows)
    (OUTPUT/"resultats.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUTPUT/"statistiques.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]))
