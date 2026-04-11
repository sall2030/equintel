#!/usr/bin/env python3
"""
Build the feature dataset from the raw SQLite database.

Usage:
    python scripts/build_features.py --db data/raceform.db --out data/features.parquet

Downloads the SQLite DB from:
    https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw
from src.features import build_features


def main():
    parser = argparse.ArgumentParser(description="Build feature dataset from SQLite DB")
    parser.add_argument("--db", default="data/raceform.db", help="Path to raceform.db")
    parser.add_argument("--out", default="data/features.parquet", help="Output parquet path")
    parser.add_argument("--limit", type=int, default=None, help="Row limit for dev/testing")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: Database not found at {args.db}")
        print()
        print("Please download the Kaggle dataset:")
        print("  https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025")
        print()
        print("Then place the extracted raceform.db in the data/ directory.")
        sys.exit(1)

    print(f"Loading raw data from {args.db}...")
    df = load_raw(args.db, limit=args.limit)
    print(f"Loaded {len(df):,} rows")

    print("\nBuilding features (this takes a while on the full dataset)...")
    features = build_features(df, show_progress=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    features.to_parquet(args.out, index=False)
    print(f"\nFeatures saved to {args.out}")
    print(f"Shape: {features.shape}")
    print(f"\nTarget distribution:")
    print(f"  Win rate:   {features['won'].mean():.3f}")
    print(f"  Place rate: {features['placed'].mean():.3f}")


if __name__ == "__main__":
    main()
