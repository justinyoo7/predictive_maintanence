#!/usr/bin/env python3
"""Generate reproducible Oracle-mock CSV tables for demo/training."""

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
        "replacement_time_hours": 1.5,
        "expedite_ship_cost": 180.0,
        "satisfaction_penalty_per_day": 140.0,
    },
    {
        "part_id": "body",
        "part_name": "Body Frame",
        "unit_cost": 1800.0,
        "replacement_time_hours": 6.5,
        "expedite_ship_cost": 650.0,
        "satisfaction_penalty_per_day": 260.0,
    },
    {
        "part_id": "windshield",
        "part_name": "Windshield Panel",
        "unit_cost": 240.0,
        "replacement_time_hours": 1.0,
        "expedite_ship_cost": 120.0,
        "satisfaction_penalty_per_day": 120.0,
    },
    {
        "part_id": "engine",
        "part_name": "Engine Core",
        "unit_cost": 5200.0,
        "replacement_time_hours": 12.0,
        "expedite_ship_cost": 950.0,
        "satisfaction_penalty_per_day": 520.0,
    },
]

TASKS = [
    {"task_type": "planting", "intensity_scalar": 0.85},
    {"task_type": "clearing_land", "intensity_scalar": 1.15},
    {"task_type": "material_handling", "intensity_scalar": 1.45},
]

TASK_PART_STRAIN = [
    ("planting", "wheel", 0.95),
    ("planting", "body", 0.80),
    ("planting", "windshield", 0.70),
    ("planting", "engine", 0.90),
    ("clearing_land", "wheel", 1.20),
    ("clearing_land", "body", 1.25),
    ("clearing_land", "windshield", 0.95),
    ("clearing_land", "engine", 1.15),
    ("material_handling", "wheel", 1.35),
    ("material_handling", "body", 1.10),
    ("material_handling", "windshield", 1.05),
    ("material_handling", "engine", 1.30),
]

REGIONS = ["north", "south", "east", "west"]

SYMPTOM_LIBRARY = {
    "engine": [
        "engine knocking and power loss under heavy load",
        "intermittent stalling with low torque output",
        "smoke from engine bay and reduced acceleration",
        "hard starting with rough idle and vibration",
    ],
    "wheel": [
        "tire puncture and low pressure warning",
        "uneven wheel wear with repeated air loss",
        "wheel wobble and traction reduction on turns",
        "rim impact damage after rough terrain operation",
    ],
    "windshield": [
        "cracked glass spreading across windshield",
        "front glass chips with limited driver visibility",
        "windshield vibration noise and seal gap",
        "spider crack pattern after debris strike",
    ],
    "body": [
        "frame/body damage after impact event",
        "body panel deformation with alignment drift",
        "chassis flex and mounting crack near rear section",
        "body support fatigue and visible structural bend",
    ],
}

NOISE_SYMPTOMS = [
    "minor electrical warning, likely unrelated to primary failure",
    "operator reported unusual noise after long shift",
    "intermittent warning lamp observed during startup",
    "general handling issue noticed on uneven ground",
]


@dataclass(frozen=True)
class GenerationConfig:
    n_events: int
    seed: int
    demo_events: int
    output_dir: Path


def age_factor(age_days: int) -> float:
    return float(np.clip(0.75 + (age_days / 3650.0), 0.6, 2.2))


def load_environment_factor(avg_load_factor: float, environment_score: float) -> float:
    load_term = 0.85 + 0.7 * avg_load_factor
    env_term = 0.8 + 0.6 * environment_score
    return float(np.clip(load_term * env_term, 0.5, 2.8))


def build_assets(rng: np.random.Generator, n_assets: int) -> pd.DataFrame:
    model_families = ["KX-Compact", "M-Series", "L-Utility", "MX-HeavyDuty"]
    equipment_types = ["tractor_like"]
    rows = []
    for idx in range(1, n_assets + 1):
        rows.append(
            {
                "asset_id": f"A{idx:05d}",
                "model_family": rng.choice(model_families),
                "equipment_type": rng.choice(equipment_types),
                "age_days": int(rng.integers(90, 3650)),
                "usage_hours_total": int(rng.integers(80, 12000)),
            }
        )
    return pd.DataFrame(rows)


def sample_symptom(rng: np.random.Generator, primary_part: str) -> str:
    base = rng.choice(SYMPTOM_LIBRARY[primary_part])
    if rng.random() < 0.20:
        return f"{base}; {rng.choice(NOISE_SYMPTOMS)}"
    if rng.random() < 0.08:
        off_part = rng.choice([p for p in SYMPTOM_LIBRARY.keys() if p != primary_part])
        return f"{base}; also noted: {rng.choice(SYMPTOM_LIBRARY[off_part])}"
    return base


def generate_tables(config: GenerationConfig) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    today = date.today()
    out = {}

    assets = build_assets(rng, n_assets=max(250, config.n_events // 3))
    parts = pd.DataFrame(PARTS)
    tasks = pd.DataFrame(TASKS)
    task_part_strain = pd.DataFrame(
        TASK_PART_STRAIN, columns=["task_type", "part_id", "strain_multiplier"]
    )

    base_lambda = {"wheel": 0.11, "body": 0.03, "windshield": 0.07, "engine": 0.05}
    part_prior = np.array([0.40, 0.14, 0.24, 0.22])
    part_ids = ["wheel", "body", "windshield", "engine"]

    dealer_cases_rows: List[dict] = []
    events_rows: List[dict] = []
    failures_rows: List[dict] = []
    parts_used_rows: List[dict] = []
    trip_rows: List[dict] = []

    demo_event_indices = set(
        rng.choice(np.arange(1, config.n_events + 1), size=min(config.demo_events, config.n_events), replace=False).tolist()
    )

    for i in range(1, config.n_events + 1):
        case_id = f"C{i:06d}"
        event_id = f"E{i:06d}"
        asset_row = assets.iloc[int(rng.integers(0, len(assets)))]
        task_row = tasks.iloc[int(rng.integers(0, len(tasks)))]
        primary_part = str(rng.choice(part_ids, p=part_prior))
        duration_days = int(rng.integers(1, 8))
        environment_score = float(np.round(rng.uniform(0.1, 1.0), 3))
        avg_load_factor = float(np.round(rng.uniform(0.45, 1.25), 3))
        opened_date = today - timedelta(days=int(rng.integers(1, 720)))
        region = str(rng.choice(REGIONS))
        severity = int(np.clip(rng.normal(3.2, 1.0), 1, 5))
        is_demo = i in demo_event_indices

        dealer_cases_rows.append(
            {
                "case_id": case_id,
                "dealer_id": f"D{int(rng.integers(1, 40)):03d}",
                "asset_id": asset_row["asset_id"],
                "opened_date": opened_date.isoformat(),
                "symptom_text": sample_symptom(rng, primary_part),
                "region": region,
                "severity": severity,
            }
        )

        events_rows.append(
            {
                "event_id": event_id,
                "case_id": case_id,
                "task_type": task_row["task_type"],
                "duration_days": duration_days,
                "environment_score": environment_score,
                "avg_load_factor": avg_load_factor,
                "is_demo": is_demo,
            }
        )

        total_used_by_part = {pid: 0 for pid in part_ids}
        had_stockout = False
        event_downtime_hours = 0.0
        replacement_cost = 0.0
        expedite_cost = 0.0

        for part_id in part_ids:
            strain = float(
                task_part_strain[
                    (task_part_strain["task_type"] == task_row["task_type"])
                    & (task_part_strain["part_id"] == part_id)
                ]["strain_multiplier"].iloc[0]
            )
            lam = (
                base_lambda[part_id]
                * age_factor(int(asset_row["age_days"]))
                * strain
                * load_environment_factor(avg_load_factor, environment_score)
                * float(np.clip(rng.lognormal(mean=0.0, sigma=0.18), 0.65, 1.45))
            )
            if part_id == primary_part:
                lam *= 1.45

            fail_count = int(rng.poisson(lam * duration_days))
            if fail_count > 0:
                day_weights = rng.random(duration_days)
                day_weights = day_weights / day_weights.sum()
                daily_counts = rng.multinomial(fail_count, day_weights)
                for day_idx, qty in enumerate(daily_counts, start=1):
                    if qty == 0:
                        continue
                    failures_rows.append(
                        {
                            "failure_id": f"F{i:06d}_{part_id}_{day_idx}",
                            "event_id": event_id,
                            "part_id": part_id,
                            "day_of_event": day_idx,
                            "failed_qty": int(qty),
                        }
                    )
                    total_used_by_part[part_id] += int(qty)

            # Small chance of preventive replacement despite no explicit failure rows.
            if fail_count == 0 and rng.random() < (0.06 * strain):
                total_used_by_part[part_id] += 1

        for part_id, used_qty in total_used_by_part.items():
            if used_qty <= 0:
                continue
            parts_used_rows.append(
                {"event_id": event_id, "part_id": part_id, "used_qty": int(used_qty)}
            )

            part_row = parts.loc[parts["part_id"] == part_id].iloc[0]
            replacement_cost += float(part_row["unit_cost"]) * used_qty
            local_capacity = int(rng.integers(0, 3))
            shortage = max(0, used_qty - local_capacity)
            if shortage > 0:
                had_stockout = True
                expedite_cost += shortage * float(part_row["expedite_ship_cost"])
                event_downtime_hours += shortage * (
                    float(part_row["replacement_time_hours"]) + 12.0
                )
            event_downtime_hours += min(used_qty, local_capacity) * float(
                part_row["replacement_time_hours"]
            )

        extra_trip_required = had_stockout and (rng.random() < 0.78)
        carrying_cost = 40.0 + 22.0 * duration_days
        total_cost = carrying_cost + replacement_cost + expedite_cost
        trip_rows.append(
            {
                "event_id": event_id,
                "had_stockout": bool(had_stockout),
                "extra_trip_required": bool(extra_trip_required),
                "downtime_hours": round(event_downtime_hours, 3),
                "total_cost": round(total_cost, 2),
            }
        )

    inventory_rows = []
    for region in REGIONS:
        for part in PARTS:
            part_id = part["part_id"]
            base_stock = {"wheel": 140, "body": 28, "windshield": 75, "engine": 20}[part_id]
            inventory_rows.append(
                {
                    "region": region,
                    "part_id": part_id,
                    "on_hand_qty": int(
                        max(0, np.round(rng.normal(base_stock, base_stock * 0.2)))
                    ),
                    "last_updated": today.isoformat(),
                }
            )

    out["dealer_cases"] = pd.DataFrame(dealer_cases_rows)
    out["assets"] = assets
    out["parts"] = parts
    out["tasks"] = tasks
    out["task_part_strain"] = task_part_strain
    out["events"] = pd.DataFrame(events_rows)
    out["failures"] = pd.DataFrame(failures_rows)
    out["parts_used"] = pd.DataFrame(parts_used_rows)
    out["trip_outcomes"] = pd.DataFrame(trip_rows)
    out["inventory_snapshot"] = pd.DataFrame(inventory_rows)
    return out


def write_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Oracle-mock CSV datasets.")
    parser.add_argument("--n_events", type=int, default=2000, help="Number of events.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--demo_events",
        type=int,
        default=50,
        help="Number of events marked as demo slice.",
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
        n_events=args.n_events,
        seed=args.seed,
        demo_events=args.demo_events,
        output_dir=args.output_dir,
    )
    tables = generate_tables(config)
    write_tables(tables, config.output_dir)
    print(
        f"Generated {len(tables)} tables in {config.output_dir} "
        f"(n_events={config.n_events}, seed={config.seed})."
    )


if __name__ == "__main__":
    main()
