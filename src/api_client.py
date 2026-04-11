"""
The Racing API client for fetching live racecards and odds.

API docs: https://api.theracingapi.com/documentation
Free tier: basic racecards for UK/Ireland.

Set credentials in .env:
    RACING_API_USERNAME=...
    RACING_API_PASSWORD=...
"""

import os
from datetime import date
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.theracingapi.com/v1"


class RacingAPIClient:
    def __init__(self, username: str = None, password: str = None):
        self.username = username or os.getenv("RACING_API_USERNAME")
        self.password = password or os.getenv("RACING_API_PASSWORD")
        if not self.username or not self.password:
            raise ValueError(
                "Set RACING_API_USERNAME and RACING_API_PASSWORD in your .env file. "
                "Sign up at https://www.theracingapi.com"
            )
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{BASE_URL}{endpoint}"
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_racecards(self, race_date: Optional[date] = None) -> list[dict]:
        """
        Fetch racecards for a given date (defaults to today).

        Returns a list of races, each containing a list of runners.
        """
        params = {}
        if race_date:
            params["date"] = race_date.strftime("%Y-%m-%d")

        data = self._get("/racecards/standard", params=params)
        races = data.get("racecards", [])
        return [self._parse_race(r) for r in races]

    def get_results(self, race_date: Optional[date] = None) -> list[dict]:
        """Fetch results for a given date."""
        params = {}
        if race_date:
            params["date"] = race_date.strftime("%Y-%m-%d")
        data = self._get("/results", params=params)
        return data.get("results", [])

    def _parse_race(self, raw: dict) -> dict:
        """Normalise a raw racecard into the format expected by predict_race()."""
        runners = []
        for r in raw.get("runners", []):
            # Decimal SP from market odds (try to get best available)
            sp = self._extract_sp(r)
            runners.append({
                "horse": r.get("horse", ""),
                "jockey": r.get("jockey", ""),
                "trainer": r.get("trainer", ""),
                "age": self._safe_int(r.get("age")),
                "weight_lbs": self._parse_weight(r.get("weight")),
                "draw": self._safe_int(r.get("draw")),
                "official_rating": self._safe_float(r.get("ofr") or r.get("official_rating")),
                "rpr": self._safe_float(r.get("rpr")),
                "ts": self._safe_float(r.get("ts")),
                "sp_decimal": sp,
                "bookie_odds": sp,  # For value comparison
            })

        # Parse distance and class
        dist_raw = raw.get("distance_f") or raw.get("dist") or ""
        race_class = raw.get("race_class") or raw.get("class") or ""
        try:
            race_class_num = float(str(race_class).replace("Class ", "").strip())
        except (ValueError, TypeError):
            race_class_num = None

        race_meta = {
            "date": raw.get("date", ""),
            "course": raw.get("course", ""),
            "race_id": raw.get("race_id", ""),
            "race_name": raw.get("race_name", ""),
            "dist": str(dist_raw),
            "going": raw.get("going", ""),
            "race_class": race_class_num,
            "ran": len(runners),
            "prize": raw.get("prize", ""),
            "off_time": raw.get("off_dt") or raw.get("off", ""),
        }

        return {"meta": race_meta, "runners": runners}

    def _extract_sp(self, runner: dict) -> Optional[float]:
        """Try to extract the current decimal odds for a runner."""
        # The Racing API sometimes provides 'sp' or 'odds' fields
        for key in ("sp", "odds", "win_sp", "decimal_sp"):
            val = runner.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def _safe_int(val) -> Optional[int]:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_weight(wgt_str) -> Optional[float]:
        """Parse weight string like '9-2' to pounds."""
        if not wgt_str:
            return None
        import re
        m = re.match(r"(\d+)-(\d+)", str(wgt_str))
        if m:
            return int(m.group(1)) * 14 + int(m.group(2))
        try:
            return float(wgt_str)
        except (ValueError, TypeError):
            return None
