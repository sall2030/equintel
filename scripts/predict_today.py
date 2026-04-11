#!/usr/bin/env python3
"""
Fetch today's racecards from The Racing API, predict win/place probabilities,
and identify value bets by comparing model fair odds to bookmaker odds.

Usage:
    python scripts/predict_today.py
    python scripts/predict_today.py --date 2026-04-12
    python scripts/predict_today.py --course Cheltenham
    python scripts/predict_today.py --min-edge 10 --max-odds 20

Requirements:
    - Trained models in models/ (run train_model.py first)
    - .env file with RACING_API_USERNAME and RACING_API_PASSWORD
    - Historical DB in data/raceform.db (for form feature computation)
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.api_client import RacingAPIClient
from src.data_loader import load_raw
from src.predict import predict_race
from src.train import load_models
from src.value import find_value_bets, format_predictions_table, format_value_table


def main():
    parser = argparse.ArgumentParser(description="Predict today's horse races")
    parser.add_argument("--date", default=None, help="Race date YYYY-MM-DD (default: today)")
    parser.add_argument("--course", default=None, help="Filter by course name")
    parser.add_argument("--db", default="data/raceform.db", help="Path to historical DB")
    parser.add_argument("--models", default="models", help="Models directory")
    parser.add_argument("--min-edge", type=float, default=5.0, help="Minimum edge %% for value bets")
    parser.add_argument("--max-odds", type=float, default=50.0, help="Max bookie odds to consider")
    parser.add_argument("--min-odds", type=float, default=2.0, help="Min bookie odds to consider")
    parser.add_argument("--show-all", action="store_true", help="Show all predictions (not just value bets)")
    args = parser.parse_args()

    race_date = date.fromisoformat(args.date) if args.date else date.today()

    # Load models
    print(f"Loading models from {args.models}/...")
    try:
        models = load_models(args.models)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load historical data for form features
    print(f"Loading historical data from {args.db}...")
    if not os.path.exists(args.db):
        print(f"ERROR: Historical database not found at {args.db}")
        print("Download from: https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025")
        sys.exit(1)

    historical_df = load_raw(args.db)
    print(f"Loaded {len(historical_df):,} historical rows")

    # Fetch racecards
    print(f"\nFetching racecards for {race_date.strftime('%A %d %B %Y')}...")
    client = RacingAPIClient()
    races = client.get_racecards(race_date)

    if not races:
        print("No races found for this date.")
        sys.exit(0)

    # Filter by course
    if args.course:
        races = [r for r in races if args.course.lower() in r["meta"]["course"].lower()]
        if not races:
            print(f"No races found at {args.course}")
            sys.exit(0)

    print(f"Found {len(races)} race(s)\n")

    all_value_bets = []

    for race in races:
        meta = race["meta"]
        runners = race["runners"]

        if len(runners) < 2:
            continue

        course = meta.get("course", "Unknown")
        off_time = meta.get("off_time", "")
        race_name = meta.get("race_name", "")
        dist = meta.get("dist", "")
        going = meta.get("going", "")

        print("=" * 70)
        print(f"  {course}  {off_time}  |  {race_name}")
        print(f"  {dist}  |  Going: {going}  |  {len(runners)} runners")
        print("=" * 70)

        try:
            preds = predict_race(runners, meta, historical_df, models)
        except Exception as e:
            print(f"  ERROR predicting race: {e}")
            continue

        if args.show_all:
            print(format_predictions_table(preds))
            print()

        value_bets = find_value_bets(
            preds,
            min_value_pct=args.min_edge,
            min_bookie_odds=args.min_odds,
            max_bookie_odds=args.max_odds,
        )

        if not value_bets.empty:
            print("  VALUE BETS:")
            print(format_value_table(value_bets))
            value_bets["course"] = course
            value_bets["off_time"] = off_time
            value_bets["race_name"] = race_name
            all_value_bets.append(value_bets)
        else:
            print("  No value bets found.")
        print()

    # Summary
    if all_value_bets:
        combined = pd.concat(all_value_bets, ignore_index=True)
        combined = combined.sort_values("edge_pct", ascending=False)
        print("\n" + "=" * 70)
        print("TOP VALUE BETS OF THE DAY")
        print("=" * 70)
        top = combined[["horse", "course", "off_time", "bet_type",
                         "bookie_odds", "model_prob", "edge_pct", "kelly_stake_pct"]].head(10)
        print(top.to_string(index=False))
        print()
        print("NOTE: Kelly stake % is half-Kelly (conservative). Never bet more than you")
        print("can afford to lose. This tool is for research and educational purposes.")
    else:
        print("No value bets found across all races today.")


if __name__ == "__main__":
    main()
