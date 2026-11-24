"""Command-line entry points for historical replay and inspection of a fixing."""

from argparse import ArgumentParser
from dataclasses import asdict
from datetime import date
from pathlib import Path
import json
import sys

from .backtest import run, summary, write_csv
from .donnees import read_fixings
from .strategie import CALIBRATION, HORIZON, decide


def lookup(dates, token):
    month_only = len(token) == 7
    key = date.fromisoformat(token + "-01" if month_only else token)
    for i, fixing in enumerate(dates):
        if (month_only and fixing.replace(day=1) == key) or fixing == key:
            return i
    raise ValueError(f"Aucun fixing correspondant à {token}.")


def build_parser():
    parser = ArgumentParser(description="Arbre hybride EUSA10 avec tickets de 50 M€ et horizon de 60 mois.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    backtest = subcommands.add_parser("backtest", help="Rejouer un ou plusieurs programmes complets.")
    backtest.add_argument("--xlsx", type=Path, required=True, help="Classeur historique EUSA10.")
    backtest.add_argument("--depart", help="Départ à rejouer, au format AAAA-MM ou AAAA-MM-JJ.")
    backtest.add_argument("--output", type=Path, default=Path("backtest_eusa10_hybride.csv"),
                          help="Chemin du CSV produit.")
    backtest.add_argument("--jobs", type=int, default=3, help="Processus de calcul du GARCH.")

    signal = subcommands.add_parser("signal", help="Évaluer une décision à un fixing et un solde donnés.")
    signal.add_argument("--xlsx", type=Path, required=True)
    signal.add_argument("--depart", required=True, help="Début du programme, au format AAAA-MM ou AAAA-MM-JJ.")
    signal.add_argument("--date", required=True, help="Date du fixing à évaluer, au format AAAA-MM ou AAAA-MM-JJ.")
    signal.add_argument("--tickets-restants", type=int, required=True, choices=range(7))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        dates, rates = read_fixings(args.xlsx)
        if args.command == "signal":
            start = lookup(dates, args.depart)
            index = lookup(dates, args.date)
            payload = asdict(decide(rates, start, index, args.tickets_restants))
            payload = {
                "date_de_depart": dates[start].isoformat(),
                "date_de_fixing": dates[index].isoformat(),
                **payload,
            }
        else:
            starts = [lookup(dates, args.depart)] if args.depart else None
            if args.output.resolve() == args.xlsx.resolve():
                raise ValueError("Le chemin de sortie doit être distinct du classeur source.")
            count = len(starts) if starts is not None else max(0, len(dates) - CALIBRATION - HORIZON + 1)
            print(f"Rejeu de {count} programme(s) sur {len(dates)} fixings mensuels.", file=sys.stderr)
            results = run(dates, rates, starts=starts, workers=args.jobs)
            write_csv(args.output, results)
            payload = {**summary(results), "csv": str(args.output)}
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Erreur — {error}\n")


if __name__ == "__main__":
    main()

