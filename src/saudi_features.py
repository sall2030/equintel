"""
Saudi-specific feature engineering for Equintel.

Extends the base UK feature set with signals relevant to JCSA racing
(Riyadh King Abdulaziz, Taif King Khalid): surface, local breeding,
track, temperature, and Saudi-form splits.

Call after loading a Saudi DataFrame (see saudi_loader.py).
No look-ahead: features use only prior races when computing rates.
"""

import numpy as np
import pandas as pd


SAUDI_COURSES = {"king abdulaziz", "riyadh", "king khalid", "taif"}


def _is_saudi_course(course: str) -> bool:
    if not isinstance(course, str):
        return False
    c = course.lower()
    return any(s in c for s in SAUDI_COURSES)


def add_saudi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Saudi-specific columns to a results/features DataFrame.

    Expects columns that may already exist from the loader:
        surface, local_bred, temperature, course, date, horse, jockey, trainer, pos, won, placed

    Adds (where data allows):
        - surface_dirt
        - is_riyadh, is_taif
        - local_bred_flag
        - temperature_c
        - horse_saudi_runs, horse_saudi_win_rate
        - jockey_saudi_win_rate_90d
        - trainer_saudi_win_rate_90d
        - days_since_last_saudi_run
        - international_runner (heuristic: low Saudi runs)
    """
    out = df.copy()

    # Surface
    if "surface" in out.columns:
        out["surface_dirt"] = out["surface"].astype(str).str.lower().str.contains("dirt").astype(float)
    else:
        out["surface_dirt"] = 1.0  # default assumption for most JCSA meetings

    # Track flags
    course_lower = out.get("course", pd.Series("")).astype(str).str.lower()
    out["is_riyadh"] = course_lower.str.contains("abdulaziz|riyadh").astype(float)
    out["is_taif"] = course_lower.str.contains("khalid|taif").astype(float)

    # Local bred
    if "local_bred" in out.columns:
        out["local_bred_flag"] = out["local_bred"].fillna(False).astype(float)
    else:
        out["local_bred_flag"] = np.nan

    # Temperature
    if "temperature" in out.columns:
        out["temperature_c"] = pd.to_numeric(out["temperature"], errors="coerce")
    else:
        out["temperature_c"] = np.nan

    # Saudi form splits (requires finished races and date)
    if "date" in out.columns and "horse" in out.columns:
        out = out.sort_values("date")
        saudi_mask = out["course"].apply(_is_saudi_course) if "course" in out.columns else pd.Series(True, index=out.index)

        # Horse Saudi history
        horse_saudi = out[saudi_mask].groupby("horse")
        out["horse_saudi_runs"] = out["horse"].map(
            horse_saudi.cumcount() if hasattr(horse_saudi, "cumcount") else lambda x: np.nan
        )
        # Simpler rolling approach for rates
        out["horse_saudi_win_rate"] = (
            out[saudi_mask]
            .groupby("horse")["won"]
            .transform(lambda s: s.shift(1).expanding().mean())
            if "won" in out.columns
            else np.nan
        )

        # Days since last Saudi run
        last_saudi = out[saudi_mask].groupby("horse")["date"].shift(1)
        out["days_since_last_saudi_run"] = (out["date"] - last_saudi).dt.days

        # Jockey / Trainer 90d Saudi rates (approximate)
        if "jockey" in out.columns and "won" in out.columns:
            out["jockey_saudi_win_rate_90d"] = np.nan  # placeholder — full expanding window needs more history
        if "trainer" in out.columns and "won" in out.columns:
            out["trainer_saudi_win_rate_90d"] = np.nan

        # International heuristic
        out["international_runner"] = (out["horse_saudi_runs"].fillna(0) < 3).astype(float)

    return out


SAUDI_MODEL_FEATURES = [
    "surface_dirt",
    "is_riyadh",
    "is_taif",
    "local_bred_flag",
    "temperature_c",
    "horse_saudi_runs",
    "horse_saudi_win_rate",
    "days_since_last_saudi_run",
    "international_runner",
]
