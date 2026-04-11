"""
Value bet identification.

Compares the model's fair odds to the bookmaker's offered odds.
A "value bet" is one where the bookmaker's odds exceed the model's fair odds,
meaning the bookmaker is offering a better price than the horse's true probability
warrants.

Value = bookie_decimal_odds * model_probability - 1
  > 0 means positive expected value (value bet)
  < 0 means negative expected value (skip)

Also computes Kelly Criterion stake sizes.
"""

import numpy as np
import pandas as pd


def find_value_bets(
    predictions: pd.DataFrame,
    min_value_pct: float = 5.0,
    min_bookie_odds: float = 2.0,
    max_bookie_odds: float = 100.0,
) -> pd.DataFrame:
    """
    Identify value bets from predictions.

    Args:
        predictions: Output of predict_race(), must have columns:
            horse, bookie_odds, win_prob, place_prob, fair_win_odds, fair_place_odds
        min_value_pct: Minimum edge % to flag as a value bet (default 5%)
        min_bookie_odds: Ignore very short-priced horses (default 2.0 = Evens)
        max_bookie_odds: Ignore very long shots (default 100.0)

    Returns:
        DataFrame of value bets sorted by edge descending.
    """
    df = predictions.copy()
    df = df[df["bookie_odds"].notna() & (df["bookie_odds"] >= min_bookie_odds) & (df["bookie_odds"] <= max_bookie_odds)]

    value_rows = []

    for _, row in df.iterrows():
        bookie_odds = float(row["bookie_odds"])

        for bet_type in ("win", "place"):
            model_prob = float(row[f"{bet_type}_prob"])
            fair_odds = float(row[f"fair_{bet_type}_odds"])

            if model_prob <= 0:
                continue

            # Expected value per unit stake: E[V] = p * odds - 1
            ev = model_prob * bookie_odds - 1.0
            value_pct = ev * 100.0

            if value_pct < min_value_pct:
                continue

            # Kelly fraction: f = (p * b - q) / b  where b = bookie_odds - 1, q = 1-p
            b = bookie_odds - 1.0
            q = 1.0 - model_prob
            kelly = (model_prob * b - q) / b
            half_kelly = kelly / 2.0  # More conservative

            value_rows.append({
                "horse": row["horse"],
                "bet_type": bet_type,
                "bookie_odds": bookie_odds,
                "fair_odds": fair_odds,
                "model_prob": round(model_prob * 100, 1),
                "implied_bookie_prob": round(100.0 / bookie_odds, 1),
                "edge_pct": round(value_pct, 1),
                "kelly_stake_pct": round(half_kelly * 100, 1),
            })

    if not value_rows:
        return pd.DataFrame(columns=["horse", "bet_type", "bookie_odds", "fair_odds",
                                     "model_prob", "implied_bookie_prob", "edge_pct", "kelly_stake_pct"])

    result = pd.DataFrame(value_rows).sort_values("edge_pct", ascending=False)
    return result.reset_index(drop=True)


def format_predictions_table(predictions: pd.DataFrame) -> str:
    """Return a human-readable predictions table."""
    cols = ["horse", "jockey", "win_prob", "place_prob", "fair_win_odds", "fair_place_odds", "bookie_odds"]
    cols = [c for c in cols if c in predictions.columns]
    df = predictions[cols].copy()

    df["win_prob"] = df["win_prob"].apply(lambda x: f"{x*100:.1f}%")
    df["place_prob"] = df["place_prob"].apply(lambda x: f"{x*100:.1f}%")
    df["fair_win_odds"] = df["fair_win_odds"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    df["fair_place_odds"] = df["fair_place_odds"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    df["bookie_odds"] = df["bookie_odds"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    return df.to_string(index=True)


def format_value_table(value_bets: pd.DataFrame) -> str:
    """Return a human-readable value bets table."""
    if value_bets.empty:
        return "No value bets found."
    return value_bets.to_string(index=False)
