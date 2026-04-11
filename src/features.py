"""
Feature engineering for horse racing predictions.

All form/stats features are computed with a strict date cutoff so there is
no look-ahead leakage: when building features for a race on date D, only
races completed strictly before D are used.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a, b, default=np.nan):
    try:
        return a / b if b else default
    except Exception:
        return default


def _win_rate(sub: pd.DataFrame, col_filter=None) -> float:
    """Win rate within a filtered slice of historical rows."""
    if col_filter is not None:
        sub = sub[col_filter]
    if len(sub) == 0:
        return np.nan
    return sub["won"].mean()


def _place_rate(sub: pd.DataFrame) -> float:
    if len(sub) == 0:
        return np.nan
    return sub["placed"].mean()


def _avg_sp(sub: pd.DataFrame, n=5) -> float:
    vals = sub["sp_decimal"].dropna().tail(n)
    return vals.mean() if len(vals) else np.nan


# ---------------------------------------------------------------------------
# Rolling form lookup (per-entity historical stats)
# ---------------------------------------------------------------------------

class FormLookup:
    """
    Precomputes lookback statistics efficiently by sorting and grouping.

    Usage:
        lookup = FormLookup(df)
        features = lookup.compute(df)
    """

    def __init__(self, df: pd.DataFrame):
        # Work only with finished runners
        self.df = df[df["finished"]].copy()
        self.df = self.df.sort_values("date")

        # Pre-index by entity for faster lookups
        self._horse_groups = self.df.groupby("horse")
        self._jockey_groups = self.df.groupby("jockey")
        self._trainer_groups = self.df.groupby("trainer")
        self._course_groups = self.df.groupby("course")

    def _prior(self, group_dict, key, date) -> pd.DataFrame:
        """Return all rows for key strictly before date."""
        if key not in group_dict.groups:
            return pd.DataFrame()
        idx = group_dict.groups[key]
        sub = self.df.loc[idx]
        return sub[sub["date"] < date]

    def _prior_jockey(self, jockey, date, days=90) -> pd.DataFrame:
        cutoff = date - pd.Timedelta(days=days)
        sub = self._prior(self._jockey_groups, jockey, date)
        return sub[sub["date"] >= cutoff]

    def _prior_trainer(self, trainer, date, days=90) -> pd.DataFrame:
        cutoff = date - pd.Timedelta(days=days)
        sub = self._prior(self._trainer_groups, trainer, date)
        return sub[sub["date"] >= cutoff]

    def horse_features(self, horse, date, dist_f, going_num, course):
        h = self._prior(self._horse_groups, horse, date)
        if len(h) == 0:
            return {
                "horse_runs": 0,
                "horse_win_rate": np.nan,
                "horse_place_rate": np.nan,
                "horse_win_rate_dist": np.nan,
                "horse_win_rate_going": np.nan,
                "horse_win_rate_course": np.nan,
                "days_since_last_run": np.nan,
                "last_pos_1": np.nan,
                "last_pos_2": np.nan,
                "last_pos_3": np.nan,
                "avg_last3_pos": np.nan,
                "last_sp_decimal": np.nan,
                "avg_sp_last3": np.nan,
                "class_last_run": np.nan,
            }

        h = h.sort_values("date")
        runs = len(h)

        # Days since last run
        days_since = (date - h["date"].iloc[-1]).days

        # Last 3 positions (normalised by field size, lower=better)
        last3 = h.tail(3)[["pos_clean", "ran"]].copy()
        last3["pos_norm"] = last3["pos_clean"] / last3["ran"].clip(lower=1)
        pos_vals = last3["pos_norm"].tolist()
        # Pad to length 3
        while len(pos_vals) < 3:
            pos_vals.insert(0, np.nan)

        avg_last3 = np.nanmean(pos_vals)

        # Distance suitability (within 1 furlong)
        dist_mask = h["dist_f"].notna() & (np.abs(h["dist_f"] - dist_f) <= 1) if dist_f else pd.Series(False, index=h.index)

        # Going suitability
        going_mask = h["going_num"].notna() & (h["going_num"] == going_num) if going_num else pd.Series(False, index=h.index)

        # Course suitability
        course_mask = h["course"] == course

        return {
            "horse_runs": runs,
            "horse_win_rate": _win_rate(h),
            "horse_place_rate": _place_rate(h),
            "horse_win_rate_dist": _win_rate(h[dist_mask]) if dist_mask.any() else np.nan,
            "horse_win_rate_going": _win_rate(h[going_mask]) if going_mask.any() else np.nan,
            "horse_win_rate_course": _win_rate(h[course_mask]) if course_mask.any() else np.nan,
            "days_since_last_run": float(days_since),
            "last_pos_1": pos_vals[-1],
            "last_pos_2": pos_vals[-2],
            "last_pos_3": pos_vals[-3],
            "avg_last3_pos": avg_last3,
            "last_sp_decimal": h["sp_decimal"].iloc[-1] if "sp_decimal" in h.columns else np.nan,
            "avg_sp_last3": _avg_sp(h, n=3),
            "class_last_run": float(h["class"].iloc[-1]) if "class" in h.columns and pd.notna(h["class"].iloc[-1]) else np.nan,
        }

    def jockey_features(self, jockey, date):
        j = self._prior_jockey(jockey, date, days=90)
        return {
            "jockey_win_rate_90d": _win_rate(j),
            "jockey_place_rate_90d": _place_rate(j),
            "jockey_runs_90d": len(j),
        }

    def trainer_features(self, trainer, date):
        t = self._prior_trainer(trainer, date, days=90)
        return {
            "trainer_win_rate_90d": _win_rate(t),
            "trainer_place_rate_90d": _place_rate(t),
            "trainer_runs_90d": len(t),
        }


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """
    Build the full feature matrix from the raw loaded DataFrame.

    Returns a new DataFrame with one row per (horse, race), containing
    model-ready features and the target columns (won, placed).

    Only includes rows where the horse finished the race (non-NaN pos_clean).
    """
    # Drop non-finishers for training (can't learn from them meaningfully)
    df = df[df["finished"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Building features for {len(df):,} finished runner rows...")

    lookup = FormLookup(df)

    rows = []
    iterator = tqdm(df.itertuples(), total=len(df), disable=not show_progress)

    for row in iterator:
        h_feats = lookup.horse_features(
            horse=row.horse,
            date=row.date,
            dist_f=row.dist_f if hasattr(row, "dist_f") else np.nan,
            going_num=row.going_num if hasattr(row, "going_num") else np.nan,
            course=row.course,
        )
        j_feats = lookup.jockey_features(row.jockey, row.date)
        t_feats = lookup.trainer_features(row.trainer, row.date)

        # Class as numeric (1=highest, 7=lowest in UK flat)
        class_val = pd.to_numeric(str(row.class_).replace("Class ", "").strip(), errors="coerce") if hasattr(row, "class_") else np.nan

        # Class change from last run
        class_change = np.nan
        if pd.notna(class_val) and pd.notna(h_feats["class_last_run"]):
            class_change = float(class_val) - h_feats["class_last_run"]

        feat = {
            # Identifiers (not model features)
            "date": row.date,
            "race_id": row.race_id,
            "horse": row.horse,
            "jockey": row.jockey,
            "trainer": row.trainer,
            "course": row.course,

            # Race-level features
            "dist_f": row.dist_f if hasattr(row, "dist_f") else np.nan,
            "going_num": row.going_num if hasattr(row, "going_num") else np.nan,
            "field_size": float(row.ran) if pd.notna(row.ran) else np.nan,
            "race_class": float(class_val) if pd.notna(class_val) else np.nan,
            "prize": float(str(row.prize).replace(",", "").replace("£", "").replace("€", "")) if hasattr(row, "prize") and pd.notna(row.prize) else np.nan,

            # Runner-level features
            "age": float(row.age) if pd.notna(row.age) else np.nan,
            "weight_lbs": row.weight_lbs if hasattr(row, "weight_lbs") else np.nan,
            "draw": float(row.draw) if pd.notna(row.draw) else np.nan,
            "draw_pct": _safe_div(float(row.draw), float(row.ran)) if pd.notna(row.draw) and pd.notna(row.ran) else np.nan,
            "official_rating": float(row.official_rating) if hasattr(row, "official_rating") and pd.notna(row.official_rating) else np.nan,
            "rpr": float(row.rpr) if hasattr(row, "rpr") and pd.notna(row.rpr) else np.nan,
            "ts": float(row.ts) if hasattr(row, "ts") and pd.notna(row.ts) else np.nan,

            # Market feature (most predictive single signal)
            "sp_decimal": row.sp_decimal if hasattr(row, "sp_decimal") else np.nan,
            "log_sp": np.log(row.sp_decimal) if hasattr(row, "sp_decimal") and pd.notna(row.sp_decimal) and row.sp_decimal > 0 else np.nan,
            "implied_prob": row.implied_prob if hasattr(row, "implied_prob") else np.nan,

            "class_change": class_change,
        }
        feat.update(h_feats)
        feat.update(j_feats)
        feat.update(t_feats)

        # Targets
        feat["won"] = float(row.won)
        feat["placed"] = float(row.placed)

        rows.append(feat)

    features = pd.DataFrame(rows)
    print(f"Feature matrix shape: {features.shape}")
    return features


# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

# Columns used as model inputs (exclude identifiers and targets)
MODEL_FEATURES = [
    # Race context
    "dist_f",
    "going_num",
    "field_size",
    "race_class",

    # Runner attributes
    "age",
    "weight_lbs",
    "draw",
    "draw_pct",
    "official_rating",
    "rpr",
    "ts",

    # Market
    "log_sp",
    "implied_prob",

    # Horse form
    "horse_runs",
    "horse_win_rate",
    "horse_place_rate",
    "horse_win_rate_dist",
    "horse_win_rate_going",
    "horse_win_rate_course",
    "days_since_last_run",
    "last_pos_1",
    "last_pos_2",
    "last_pos_3",
    "avg_last3_pos",
    "last_sp_decimal",
    "avg_sp_last3",
    "class_change",

    # Jockey
    "jockey_win_rate_90d",
    "jockey_place_rate_90d",
    "jockey_runs_90d",

    # Trainer
    "trainer_win_rate_90d",
    "trainer_place_rate_90d",
    "trainer_runs_90d",
]
