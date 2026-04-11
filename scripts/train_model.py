#!/usr/bin/env python3
"""
Train win and place prediction models from the feature dataset.

Usage:
    python scripts/train_model.py --features data/features.parquet --models models/

Run build_features.py first.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Train win/place XGBoost models")
    parser.add_argument("--features", default="data/features.parquet", help="Feature parquet path")
    parser.add_argument("--models", default="models", help="Directory to save models")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"ERROR: Feature file not found at {args.features}")
        print("Run scripts/build_features.py first.")
        sys.exit(1)

    print(f"Loading features from {args.features}...")
    features = pd.read_parquet(args.features)
    print(f"Loaded {len(features):,} rows")

    results = train(features, models_dir=args.models, verbose=True)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    for name, r in results.items():
        print(f"\n{name.upper()} model:")
        print(f"  AUC:   {r['auc']:.4f}")
        print(f"  Brier: {r['brier']:.4f}")
    print(f"\nModels saved to {args.models}/")


if __name__ == "__main__":
    main()
