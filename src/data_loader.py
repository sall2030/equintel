"""
Load and clean raw data from the Kaggle SQLite database.

Dataset: https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025
Schema (table: data):
  date, course, race_id, off, race_name, type, class, pattern,
  rating_band, age_band, sex_rest, dist, going, ran,
  num, pos, draw, ovr_btn, btn, horse, age, sex, wgt, hg,
  time, sp, jockey, trainer, sire, dam, damsire, owner,
  prize, or, rpr, ts, comment
"""

import re
import sqlite3
import pandas as pd
import numpy as np


GOING_MAP = {
    "firm": 1,
    "good to firm": 2,
    "good": 3,
    "good to soft": 4,
    "soft": 5,
    "heavy": 6,
    # Jumps-specific variants
    "hard": 1,
    "standard": 3,
    "standard to slow": 4,
    "slow": 5,
    "yielding": 4,
    "yielding to soft": 5,
}


def parse_sp(sp_str) -> float:
    """Convert fractional UK SP to decimal odds. Returns NaN on failure."""
    if pd.isna(sp_str):
        return np.nan
    s = str(sp_str).strip().lower()
    if s in ("evs", "evens", "1/1"):
        return 2.0
    # Already a float/int
    try:
        return float(s) + 1.0  # treat as numerator/1 if just a number? No - try fraction first
    except ValueError:
        pass
    m = re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den == 0:
            return np.nan
        return num / den + 1.0
    return np.nan


def parse_distance_furlongs(dist_str) -> float:
    """Convert distance string like '1m2f', '5f', '2m4f110y' to furlongs."""
    if pd.isna(dist_str):
        return np.nan
    s = str(dist_str).strip().lower().replace(" ", "")
    furlongs = 0.0
    # miles
    m = re.search(r"(\d+)m", s)
    if m:
        furlongs += int(m.group(1)) * 8
    # furlongs
    f = re.search(r"(\d+)f", s)
    if f:
        furlongs += int(f.group(1))
    # yards
    y = re.search(r"(\d+)y", s)
    if y:
        furlongs += int(y.group(1)) / 220.0
    if furlongs == 0:
        return np.nan
    return furlongs


def parse_going(going_str) -> int:
    """Map going description to ordinal 1 (fast) -> 6 (heavy)."""
    if pd.isna(going_str):
        return np.nan
    s = str(going_str).strip().lower()
    # Try exact match first
    if s in GOING_MAP:
        return GOING_MAP[s]
    # Partial match
    for key, val in sorted(GOING_MAP.items(), key=lambda x: -len(x[0])):
        if key in s:
            return val
    return np.nan


def parse_position(pos_str) -> float:
    """Parse finishing position; returns NaN for non-finishers (F, PU, UR, etc.)."""
    if pd.isna(pos_str):
        return np.nan
    s = str(pos_str).strip()
    try:
        return float(s)
    except ValueError:
        return np.nan  # Fell, Pulled Up, Unseated Rider, etc.


def parse_weight_lbs(wgt_str) -> float:
    """Parse weight in st-lb format (e.g. '9-2') to total pounds."""
    if pd.isna(wgt_str):
        return np.nan
    s = str(wgt_str).strip()
    m = re.match(r"^(\d+)-(\d+)$", s)
    if m:
        return int(m.group(1)) * 14 + int(m.group(2))
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_raw(db_path: str, limit: int = None) -> pd.DataFrame:
    """
    Load the raw SQLite database into a DataFrame with parsed columns.

    Args:
        db_path: Path to raceform.db
        limit: Optional row limit for development/testing

    Returns:
        DataFrame with cleaned columns
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df):,} rows, {df.columns.tolist()}")

    # Normalise column names (SQLite may use 'or' which is a Python keyword)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "or" in df.columns:
        df.rename(columns={"or": "official_rating"}, inplace=True)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Parse numeric fields
    df["sp_decimal"] = df["sp"].apply(parse_sp)
    df["implied_prob"] = 1.0 / df["sp_decimal"]
    df["dist_f"] = df["dist"].apply(parse_distance_furlongs)
    df["going_num"] = df["going"].apply(parse_going)
    df["pos_clean"] = df["pos"].apply(parse_position)
    df["weight_lbs"] = df["wgt"].apply(parse_weight_lbs)

    # Numeric coercions
    for col in ["age", "draw", "ran", "official_rating", "rpr", "ts", "ovr_btn", "btn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Finished the race (non-NaN position)
    df["finished"] = df["pos_clean"].notna()

    # Win / place targets
    df["won"] = (df["pos_clean"] == 1).astype(float)
    df["placed"] = df.apply(
        lambda r: float(
            pd.notna(r["pos_clean"])
            and r["pos_clean"] <= _place_cutoff(r.get("ran", np.nan))
        ),
        axis=1,
    )

    return df


def _place_cutoff(field_size) -> int:
    """Standard each-way place terms based on field size."""
    if pd.isna(field_size):
        return 3
    n = int(field_size)
    if n <= 4:
        return 1   # win only
    if n <= 7:
        return 2
    if n <= 15:
        return 3
    return 4   # 16+ runners: 4 places (in some markets)
