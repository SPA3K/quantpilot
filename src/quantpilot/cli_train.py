"""CLI entry point for training sector models."""

import argparse
import json
import logging
import sys

from quantpilot.config import ensure_dirs, SECTORS
from quantpilot.tools.training import train_sector_model


def main():
    parser = argparse.ArgumentParser(description="QuantPilot Model Trainer")
    parser.add_argument("--sector", "-s", help="Sector ID or name (e.g. cpo, CPO光模块)")
    parser.add_argument("--all", action="store_true", help="Train all pre-defined sectors")
    parser.add_argument("--start-date", default="20240101", help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYYMMDD)")
    parser.add_argument("--splits", type=int, default=5, help="CV folds")
    parser.add_argument("--list", action="store_true", help="List available sectors")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_dirs()

    if args.list:
        print("Available sectors:")
        for sid, cfg in SECTORS.items():
            print(f"  {sid:12s} — {cfg['name']}")
        return

    if args.all:
        sectors = list(SECTORS.keys())
    elif args.sector:
        sectors = [args.sector]
    else:
        parser.print_help()
        return

    for sector in sectors:
        print(f"\n{'='*50}")
        print(f"Training: {sector}")
        print(f"{'='*50}")
        result = train_sector_model(
            sector=sector,
            start_date=args.start_date,
            end_date=args.end_date,
            n_splits=args.splits,
        )
        data = json.loads(result)
        if data.get("success"):
            m = data["metrics"]
            print(f"✅ {sector}: IC={m['ic_mean']:.4f} IR={m['ir']:.2f}")
            print(f"   Top factors: {', '.join(f['name'] for f in data.get('top_factors', [])[:5])}")
        else:
            print(f"❌ {sector}: {data.get('error')}")


if __name__ == "__main__":
    main()
