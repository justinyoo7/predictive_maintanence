#!/usr/bin/env python3
"""Generate reproducible dealer-diagnosis CSV tables for demo/training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PARTS = [
    {
        "part_id": "wheel",
        "part_name": "Wheel Assembly",
        "unit_cost": 320.0,
        "category": "running_gear",
        "base_failure_signal": 0.95,
    },
    {
        "part_id": "body",
        "part_name": "Body Frame",
        "unit_cost": 1800.0,
        "category": "structure",
        "base_failure_signal": 0.45,
    },
    {
        "part_id": "windshield",
        "part_name": "Windshield Panel",
        "unit_cost": 240.0,
        "category": "cabin",
        "base_failure_signal": 0.65,
    },
    {
        "part_id": "engine",
        "part_name": "Engine Core",
        "unit_cost": 5200.0,
        "category": "powertrain",
        "base_failure_signal": 0.75,
    },
]

TRACTOR_MODELS = [
    {"tractor_model": "L2501", "class": "compact", "power_band": "25-35hp"},
    {"tractor_model": "L3902", "class": "compact", "power_band": "35-45hp"},
    {"tractor_model": "MX5400", "class": "utility", "power_band": "50-60hp"},
    {"tractor_model": "M5-111", "class": "utility", "power_band": "100-115hp"},
    {"tractor_model": "M6-141", "class": "ag", "power_band": "130-145hp"},
]

ISSUE_LIBRARY = {
    "engine": [
        "engine knocking and power loss under heavy load",
        "intermittent stalling with low torque output",
        "smoke from engine bay and reduced acceleration",
        "hard starting with rough idle and vibration",
        "check-engine light and repeated misfire under load",
    ],
    "wheel": [
        "tire puncture and low pressure warning",
        "uneven wheel wear with repeated air loss",
        "wheel wobble and traction reduction on turns",
        "rim impact damage after rough terrain operation",
        "abnormal tire vibration at transport speed",
    ],
    "windshield": [
        "cracked glass spreading across windshield",
        "front glass chips with limited driver visibility",
        "windshield vibration noise and seal gap",
        "spider crack pattern after debris strike",
        "wiper sweep scratches and line-of-sight obstruction",
    ],
    "body": [
        "frame/body damage after impact event",
        "body panel deformation with alignment drift",
        "chassis flex and mounting crack near rear section",
        "body support fatigue and visible structural bend",
        "loader mount alignment issue with frame stress marks",
    ],
}

NOISE_ISSUES = [
    "minor electrical warning, likely unrelated to primary failure",
    "operator reported unusual noise after long shift",
    "intermittent warning lamp observed during startup",
    "general handling issue noticed on uneven ground",
    "service advisor noted intermittent concern with low reproducibility",
]


@dataclass(frozen=True)
class GenerationConfig:
    n_cases: int
    seed: int
    demo_cases: int
    output_dir: Path


def model_part_bias(tractor_model: str) -> Dict[str, float]:
    bias = {
        "wheel": 1.0,
        "body": 1.0,
        "windshield": 1.0,
        "engine": 1.0,
    }
    if tractor_model in {"L2501", "L3902"}:
        bias["wheel"] = 1.12
        bias["body"] = 0.88
    elif tractor_model in {"MX5400"}:
        bias["wheel"] = 1.07
        bias["engine"] = 1.05
    elif tractor_model in {"M5-111", "M6-141"}:
        bias["engine"] = 1.18
        bias["body"] = 1.10
    return bias


def severity_multiplier(severity: int) -> float:
    return float(np.clip(0.65 + 0.18 * severity, 0.75, 1.65))


def sample_issue_description(rng: np.random.Generator, primary_part: str) -> str:
    base = rng.choice(ISSUE_LIBRARY[primary_part])
    if rng.random() < 0.20:
        return f"{base}; {rng.choice(NOISE_ISSUES)}"
    if rng.random() < 0.08:
        off_part = rng.choice([p for p in ISSUE_LIBRARY.keys() if p != primary_part])
        return f"{base}; also noted: {rng.choice(ISSUE_LIBRARY[off_part])}"
    return base


def generate_tables(config: GenerationConfig) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    today = date.today()
    out = {}

    parts = pd.DataFrame(PARTS)
    tractor_models = pd.DataFrame(TRACTOR_MODELS)
    part_ids = [p["part_id"] for p in PARTS]
    part_prior = np.array([0.42, 0.14, 0.20, 0.24])

    dealer_cases_rows: List[dict] = []
    parts_ordered_rows: List[dict] = []

    demo_case_indices = set(
        rng.choice(
            np.arange(1, config.n_cases + 1),
            size=min(config.demo_cases, config.n_cases),
            replace=False,
        ).tolist()
    )

    for i in range(1, config.n_cases + 1):
        case_id = f"C{i:06d}"
        primary_part = str(rng.choice(part_ids, p=part_prior))
        model_row = tractor_models.iloc[int(rng.integers(0, len(tractor_models)))]
        tractor_model = str(model_row["tractor_model"])
        bias = model_part_bias(tractor_model)
        opened_date = today - timedelta(days=int(rng.integers(1, 720)))
        severity = int(np.clip(rng.normal(3.2, 1.0), 1, 5))
        is_demo = i in demo_case_indices

        dealer_cases_rows.append(
            {
                "case_id": case_id,
                "dealer_id": f"D{int(rng.integers(1, 40)):03d}",
                "tractor_model": tractor_model,
                "opened_date": opened_date.isoformat(),
                "issue_description": sample_issue_description(rng, primary_part),
                "severity": severity,
                "is_demo": bool(is_demo),
            }
        )

        for part_id in part_ids:
            base_signal = float(parts.loc[parts["part_id"] == part_id, "base_failure_signal"].iloc[0])
            lam = base_signal * severity_multiplier(severity) * bias[part_id]
            if part_id == primary_part:
                lam *= 1.9

            qty = int(rng.poisson(lam))
            if qty <= 0 and rng.random() < 0.05 * severity_multiplier(severity):
                qty = 1
            if qty > 0:
                parts_ordered_rows.append(
                    {"case_id": case_id, "part_id": part_id, "ordered_qty": int(qty)}
                )

    out["dealer_cases"] = pd.DataFrame(dealer_cases_rows)
    out["parts"] = parts
    out["parts_ordered"] = pd.DataFrame(parts_ordered_rows)
    return out


def write_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dealer-diagnosis CSV datasets.")
    parser.add_argument("--n_cases", type=int, default=2000, help="Number of dealer cases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--demo_cases",
        type=int,
        default=50,
        help="Number of cases marked as demo slice.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "oracle_mock",
        help="Output directory for CSV tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        n_cases=args.n_cases,
        seed=args.seed,
        demo_cases=args.demo_cases,
        output_dir=args.output_dir,
    )
    tables = generate_tables(config)
    write_tables(tables, config.output_dir)
    print(
        f"Generated {len(tables)} tables in {config.output_dir} "
        f"(n_cases={config.n_cases}, seed={config.seed})."
    )


if __name__ == "__main__":
    main()
