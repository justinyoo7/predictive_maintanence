"""Mock Oracle client reading CSV tables with in-memory cache."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TABLES = [
    "dealer_cases",
    "assets",
    "parts",
    "tasks",
    "task_part_strain",
    "events",
    "failures",
    "parts_used",
    "trip_outcomes",
    "inventory_snapshot",
]


class OracleMockClient:
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or (
            Path(__file__).resolve().parents[1] / "data" / "oracle_mock"
        )
        self._cache: Dict[str, pd.DataFrame] = {}
        self.reload()

    def reload(self) -> None:
        self._cache = {}
        for table in TABLES:
            path = self.data_dir / f"{table}.csv"
            if not path.exists():
                self._cache[table] = pd.DataFrame()
                continue
            self._cache[table] = pd.read_csv(path)

    def table_names(self) -> List[str]:
        return TABLES.copy()

    def get_table(self, table_name: str) -> pd.DataFrame:
        if table_name not in TABLES:
            raise ValueError(f"Unknown table '{table_name}'")
        return self._cache.get(table_name, pd.DataFrame()).copy()

    def preview_table(self, name: str, limit: int = 50) -> List[dict]:
        df = self.get_table(name)
        if df.empty:
            return []
        return df.head(max(1, limit)).to_dict(orient="records")

    def fetch_parts(self) -> List[dict]:
        return self.get_table("parts").to_dict(orient="records")

    def fetch_inventory(self, region: Optional[str] = None) -> List[dict]:
        df = self.get_table("inventory_snapshot")
        if region:
            df = df[df["region"] == region]
        return df.to_dict(orient="records")

    def fetch_event(self, event_id: str) -> Optional[dict]:
        events = self.get_table("events")
        if events.empty:
            return None
        event_rows = events[events["event_id"] == event_id]
        if event_rows.empty:
            return None
        event = event_rows.iloc[0].to_dict()

        case_row = self.get_table("dealer_cases")
        case = case_row[case_row["case_id"] == event["case_id"]]
        if not case.empty:
            event["dealer_case"] = case.iloc[0].to_dict()

        used = self.get_table("parts_used")
        event["parts_used"] = used[used["event_id"] == event_id].to_dict(orient="records")
        return event

    def fetch_events(
        self,
        limit: Optional[int] = None,
        region: Optional[str] = None,
        only_demo: bool = False,
    ) -> List[dict]:
        events = self.get_table("events")
        if events.empty:
            return []
        cases = self.get_table("dealer_cases")[["case_id", "region", "symptom_text"]]
        joined = events.merge(cases, on="case_id", how="left")
        if region:
            joined = joined[joined["region"] == region]
        if only_demo and "is_demo" in joined.columns:
            joined = joined[joined["is_demo"] == True]  # noqa: E712
        if limit is not None:
            joined = joined.head(max(1, limit))
        return joined.to_dict(orient="records")

    def fetch_dealer_cases(
        self,
        limit: Optional[int] = None,
        region: Optional[str] = None,
        severity_min: Optional[int] = None,
    ) -> List[dict]:
        df = self.get_table("dealer_cases")
        if df.empty:
            return []
        if region:
            df = df[df["region"] == region]
        if severity_min is not None:
            df = df[df["severity"] >= severity_min]
        if limit is not None:
            df = df.head(max(1, limit))
        return df.to_dict(orient="records")


oracle_client = OracleMockClient()
