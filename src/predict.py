"""
Generate win/place probability predictions for a racecard.

Given a racecard (list of runners with their attributes and the historical DB),
this module:
  1. Computes features for each runner
  2. Runs the win and place models
  3. Normalises win probabilities per race (they must sum to 1)
  4. Converts probabilities to implied fair odds
"""

import numpy as np
import pandas as pd

from .features import MODEL_FEATURES, FormLookup
from .data_loader import parse_sp, parse_distance_furlongs, parse_going


def predict_race(
    runners: list[dict],
    race_meta: dict,
    historical_df: pd.DataFrame,
    models: dict,
) -> pd.DataFrame:
    """
    Predict win/place probabilities for a single race.

    Args:
        runners: List of runner dicts from the racecard. Expected keys:
            horse, jockey, trainer, age, weight_lbs, draw, official_rating,
            rpr, ts, sp_decimal (current market price, if available)
        race_meta: Dict with race-level info:
            date, course, dist, going, race_class, ran, prize
        historical_df: Full historical DataFrame (from data_loader.load_raw)
        models: Dict with 'win' and 'place' sub-dicts from train.load_models()

    Returns:
        DataFrame with one row per runner, sorted by predicted win probability.
    """
    race_date = pd.Timestamp(race_meta["date"])
    dist_f = parse_distance_furlongs(race_meta.get("dist"))
    going_num = parse_going(race_meta.get("going"))
    course = race_meta.get("course", "")
    ran = len(runners)

    lookup = FormLookup(historical_df)

    feature_rows = []
    for runner in runners:
        sp = runner.get("sp_decimal") or parse_sp(runner.get("sp", ""))
        log_sp = np.log(sp) if sp and sp > 0 else np.nan
        implied_prob = 1.0 / sp if sp and sp > 0 else np.nan

        h = lookup.horse_features(
            horse=runner["horse"],
            date=race_date,
            dist_f=dist_f,
            going_num=going_num,
            course=course,
        )
        j = lookup.jockey_features(runner.get("jockey", ""), race_date)
        t = lookup.trainer_features(runner.get("trainer", ""), race_date)

        class_val = float(race_meta.get("race_class", np.nan)) if race_meta.get("race_class") else np.nan
        class_change = np.nan
        if pd.notna(class_val) and pd.notna(h.get("class_last_run")):
            class_change = class_val - h["class_last_run"]

        feat = {
            "horse": runner["horse"],
            "jockey": runner.get("jockey", ""),
            "trainer": runner.get("trainer", ""),
            "bookie_odds": runner.get("bookie_odds", np.nan),

            # Race context
            "dist_f": dist_f,
            "going_num": going_num,
            "field_size": float(ran),
            "race_class": class_val,
            "prize": float(race_meta.get("prize", np.nan)) if race_meta.get("prize") else np.nan,

            # Runner attributes
            "age": float(runner.get("age", np.nan)),
            "weight_lbs": float(runner.get("weight_lbs", np.nan)),
            "draw": float(runner.get("draw", np.nan)),
            "draw_pct": float(runner.get("draw", np.nan)) / ran if runner.get("draw") else np.nan,
            "official_rating": float(runner.get("official_rating", np.nan)),
            "rpr": float(runner.get("rpr", np.nan)),
            "ts": float(runner.get("ts", np.nan)),

            # Market
            "sp_decimal": sp,
            "log_sp": log_sp,
            "implied_prob": implied_prob,

            "class_change": class_change,
        }
        feat.update(h)
        feat.update(j)
        feat.update(t)
        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)

    # Build model input matrix
    cols = [c for c in MODEL_FEATURES if c in feat_df.columns]
    X = feat_df[cols].values.astype(float)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_medians[j]

    results = feat_df[["horse", "jockey", "trainer", "bookie_odds"]].copy()

    for target in ("win", "place"):
        m = models[target]
        X_scaled = m["scaler"].transform(X)
        raw_probs = m["model"].predict_proba(X_scaled)[:, 1]
        results[f"{target}_prob_raw"] = raw_probs

    # Normalise win probabilities to sum to 1 across the field
    win_sum = results["win_prob_raw"].sum()
    results["win_prob"] = results["win_prob_raw"] / (win_sum if win_sum > 0 else 1.0)

    # Place probability: normalise similarly
    place_sum = results["place_prob_raw"].sum()
    results["place_prob"] = results["place_prob_raw"] / (place_sum if place_sum > 0 else 1.0)

    # Fair odds = 1 / probability (no overround)
    results["fair_win_odds"] = 1.0 / results["win_prob"].clip(lower=0.001)
    results["fair_place_odds"] = 1.0 / results["place_prob"].clip(lower=0.001)

    results = results.drop(columns=["win_prob_raw", "place_prob_raw"])
    results = results.sort_values("win_prob", ascending=False).reset_index(drop=True)
    results.index += 1  # 1-based rank

    return results
