"""Mock Oracle client reading CSV tables with in-memory cache."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TABLES = [
    "dealer_cases",
    "parts",
    "parts_ordered",
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

    def fetch_dealer_cases(
        self,
        limit: Optional[int] = None,
        severity_min: Optional[int] = None,
        tractor_model: Optional[str] = None,
    ) -> List[dict]:
        df = self.get_table("dealer_cases")
        if df.empty:
            return []
        if severity_min is not None:
            df = df[df["severity"] >= severity_min]
        if tractor_model:
            df = df[df["tractor_model"] == tractor_model]
        if limit is not None:
            df = df.head(max(1, limit))
        return df.to_dict(orient="records")


oracle_client = OracleMockClient()
