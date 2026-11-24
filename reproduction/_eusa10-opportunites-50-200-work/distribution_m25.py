"""Compare gain distributions on the identical set of 4,607 hedge programmes."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
SOURCE = ROOT/"backtest-eusa10-1md-opportunites-petits-tickets.csv"
OUTPUT = ROOT/"distribution-eusa10-complement-mois25.png"
BASE = "Fixe 200 M€ · "
VARIANT = "50 M€ en bas de fourchette à partir de M25 · "
BG, INK, MUTED, GRID = "#f7f9fc", "#142639", "#4e6479", "#dbe3e9"
TEAL, LOSS, BASE_COLOR = "#086a80", "#af5431", "#506079"


def fr(number,digits=2,signed=False):
    return format(number,f"{'+' if signed else ''},.{digits}f").replace(",","\u202f").replace(".",",").replace("-","−")


def take():
    with SOURCE.open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream,delimiter=";"))
    series = lambda key: np.asarray([float(r[key].replace(",",".")) for r in rows])
    active = series(VARIANT+"Gain (pb)")
    fixed = series(BASE+"Gain (pb)")
    assert len(rows) == 4607 and len({r["Date de lancement"] for r in rows}) == len(rows)
    assert np.isfinite(active).all() and np.isfinite(fixed).all()
    winners = int((active>0).sum())
    losses = int((active<0).sum())
    high_before = int(sum(int(r[BASE+"Solde M60 (M€)"])>=800 for r in rows))
    high_after = int(sum(int(r[VARIANT+"Solde M60 (M€)"])>=800 for r in rows))
    assert (winners,losses,high_before,high_after) == (4516,91,528,0)
    return active,fixed,winners,losses,high_after


def tidy(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y",color=GRID,linewidth=.9)
    for side,s in ax.spines.items():
        s.set_visible(side=="bottom")
        s.set_color("#bccbd6")
    ax.tick_params(axis="both",length=0,pad=8,labelsize=11,colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_ylim(bottom=0)


def main():
    g,b,wins,losers,high_after=take()
    N=len(g)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.labelsize":12})
    fig=plt.figure(figsize=(14,7.55),facecolor=BG)
    fig.text(.064,.924,"Gains avec le complément de 50 M€",fontsize=28,fontweight="bold",color=INK)
    fig.text(.066,.872,"Filtre à partir du mois 25  ·  4 607 programmes de 1 Md€ sur cinq ans  ·  Avant coûts",
             fontsize=12.9,color=MUTED)
    kpis=[
        (.065,"GAIN MOYEN",fr(g.mean(),signed=True)+" pb",TEAL),
        (.280,"GAIN MÉDIAN",fr(float(np.median(g)),signed=True)+" pb",TEAL),
        (.497,"DÉPARTS GAGNANTS",fr(100*wins/N,1)+" %",TEAL),
        (.724,"SOLDE ≥ 800 M€ AU MOIS 60",str(high_after)+" départ",TEAL),
    ]
    for x,label,value,color in kpis:
        fig.text(x,.799,label,fontsize=10.5,fontweight="bold",color=MUTED)
        fig.text(x,.748,value,fontsize=23,fontweight="bold",color=color)
    fig.add_artist(Line2D([.065,.95],[.715,.715],transform=fig.transFigure,color=GRID,linewidth=.95))

    ax=fig.add_axes((.088,.19,.559,.422),facecolor=BG)
    ax.set_title("Tous les gains",loc="left",fontsize=15,fontweight="bold",pad=25,color=INK)
    common=np.arange(-20,151,10)
    counts,edges=np.histogram(g,common)
    base_counts,_=np.histogram(b,common)
    assert counts.sum()==base_counts.sum()==N and 0 in edges
    bars=ax.bar(edges[:-1],counts*100/N,width=np.diff(edges),align="edge",
                color=np.where(edges[:-1]<0,LOSS,TEAL),edgecolor=BG,linewidth=1,alpha=.89)
    for bar,edge in zip(bars,edges[:-1],strict=True):
        if edge<0:
            bar.set_hatch("///")
    ax.stairs(base_counts*100/N,edges,baseline=None,color=BASE_COLOR,linestyle=(0,(3,2)),linewidth=1.6,zorder=4)
    ax.axvline(0,color=INK,linewidth=1.35)
    ax.set_xlim(edges[0],edges[-1])
    ax.set_xticks(np.arange(-20,141,20))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:fr(x,0)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:fr(y,0)+" %"))
    ax.set_xlabel("Gain face au linéaire (pb)",labelpad=11)
    ax.set_ylabel("Part des 4 607 départs",labelpad=10)
    tidy(ax)
    proxies=[
        Patch(facecolor=TEAL,alpha=.89,label="Complément à M25"),
        Line2D([0,1],[0,0],color=BASE_COLOR,linestyle=(0,(3,2)),linewidth=1.6,label="Arbre à 200 M€"),
    ]
    ax.legend(handles=proxies,loc="lower right",bbox_to_anchor=(1,1.03),ncol=2,
              frameon=False,fontsize=9.8,labelcolor=INK,handlelength=1.6,columnspacing=1.15,borderpad=.05)

    z=fig.add_axes((.727,.19,.223,.422),facecolor=BG)
    z.set_title("Zoom sur les pertes",loc="left",fontsize=15,fontweight="bold",pad=25,color=INK)
    lost=g[g<0]
    bins=np.arange(-15,.01,2.5)
    lc,le=np.histogram(lost,bins)
    assert lc.sum()==losers
    z.bar(le[:-1],lc,width=np.diff(le),align="edge",color=LOSS,edgecolor=BG,linewidth=1,alpha=.91)
    z.set_xlim(bins[0],0)
    z.set_ylim(0,1.28*lc.max())
    z.set_xticks(np.arange(-15,1,5))
    z.xaxis.set_major_formatter(FuncFormatter(lambda x,_:fr(x,0)))
    z.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    z.set_xlabel("Gain négatif (pb)",labelpad=11)
    z.set_ylabel("Nombre de départs",labelpad=9)
    tidy(z)
    z.text(.98,.91,fr(losers,0)+" départs · "+fr(losers/N*100,1)+" % du total",
           color=MUTED,fontsize=10.2,ha="right",va="top",transform=z.transAxes)
    fig.text(.065,.073,"Un gain positif signifie que la variante a fixé un taux moyen inférieur à celui du linéaire.",
             fontsize=10.5,color=INK)
    fig.text(.065,.044,"EUSA10.xlsx et fixing du 2 novembre 2026 fourni  ·  Fenêtres historiques chevauchantes",
             fontsize=9.6,color=MUTED)
    fig.savefig(OUTPUT,dpi=170,facecolor=BG)
    plt.close(fig)
    print(OUTPUT, OUTPUT.stat().st_size)


if __name__=="__main__":
    main()

