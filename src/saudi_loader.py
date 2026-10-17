"""
Saudi racing data loader for Equintel.

Loads structured results (CSV/JSON) from JCSA sources or manual exports
and maps them into the DataFrame format expected by features.py.

Optionally enriches with Saudi-specific features via saudi_features.py.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from .saudi_features import add_saudi_features


def load_saudi_csv(path: str | Path, add_features: bool = True) -> pd.DataFrame:
    """Load a normalized Saudi results CSV.

    Expected columns (minimum):
        date, race_id, course, horse, jockey, trainer,
        dist_f, going, ran, pos, sp_decimal, age, weight_lbs, draw
    Optional: surface, local_bred, temperature, official_rating, prize
    """
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "horse", "jockey", "trainer", "course"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic cleaning aligned with UK loader expectations
    df["finished"] = df.get("pos", pd.Series()).notna()
    df["won"] = (df.get("pos") == 1).astype(float)
    df["placed"] = (df.get("pos") <= 3).astype(float)

    if add_features:
        df = add_saudi_features(df)

    return df


def load_saudi_json(path: str | Path, add_features: bool = True) -> pd.DataFrame:
    """Load from JSON list of race/runner objects."""
    df = pd.read_json(path)
    df = load_saudi_csv_from_df(df)
    if add_features:
        df = add_saudi_features(df)
    return df


def load_saudi_csv_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an already-loaded DataFrame."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def example_schema() -> dict:
    """Return the recommended column schema for Saudi exports."""
    return {
        "date": "YYYY-MM-DD",
        "race_id": "unique race identifier",
        "course": "King Abdulaziz / King Khalid",
        "horse": "horse name",
        "jockey": "jockey name",
        "trainer": "trainer name",
        "dist_f": "distance in furlongs",
        "going": "track condition",
        "surface": "Dirt / Turf",
        "ran": "field size",
        "pos": "finishing position",
        "sp_decimal": "starting price decimal",
        "age": "age",
        "weight_lbs": "weight",
        "draw": "barrier",
        "local_bred": "boolean",
        "temperature": "optional Celsius",
    }
