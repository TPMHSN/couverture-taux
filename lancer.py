"""Relancer les backtests de couverture de taux."""
from argparse import ArgumentParser
from pathlib import Path
from subprocess import run
import sys

ROOT = Path(__file__).resolve().parent
REPRO = ROOT / "reproduction"

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("serie", choices=("eusa10", "usd"))
    args = parser.parse_args()
    if args.serie == "eusa10":
        command = [REPRO / "_eusa10-opportunites-50-200-work/replay_observed.py"]
    else:
        command = [REPRO / "replay_usd.py", ROOT / "donnees/originaux/USDSB3L10Y.csv"]
    run([sys.executable, *map(str, command)], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
