"""Scenario simulation and executive summary KPIs."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from app.modeling import build_features_from_event, predict_quantities
from app.oracle_client import OracleMockClient


def _part_catalog(client: OracleMockClient) -> Dict[str, dict]:
    return {p["part_id"]: p for p in client.fetch_parts()}


def recommend_spares(expected: Dict[str, float], safety_buffer: float = 0.35) -> Dict[str, int]:
    return {part_id: int(math.ceil(max(0.0, qty) + safety_buffer)) for part_id, qty in expected.items()}


def _actual_used_from_event(client: OracleMockClient, event_id: str, fallback_expected: Dict[str, float]) -> Dict[str, int]:
    event = client.fetch_event(event_id)
    if not event:
        return {k: int(math.ceil(v)) for k, v in fallback_expected.items()}
    used_rows = event.get("parts_used", [])
    actual = {k: 0 for k in fallback_expected.keys()}
    for row in used_rows:
        actual[row["part_id"]] = int(row["used_qty"])
    return actual


def _evaluate_scenario(
    bring_qty: Dict[str, int],
    actual_used_qty: Dict[str, int],
    part_map: Dict[str, dict],
    w1_money: float,
    w2_downtime: float,
    w3_satisfaction: float,
    shipping_delay_hours: float,
) -> Dict[str, float]:
    carrying_cost = 0.0
    replacement_cost = 0.0
    expedite_cost = 0.0
    downtime_hours = 0.0
    satisfaction_penalty = 0.0
    stockout_parts = 0

    for part_id, actual_qty in actual_used_qty.items():
        part = part_map[part_id]
        brought = bring_qty.get(part_id, 0)
        shortage = max(0, actual_qty - brought)

        carrying_cost += brought * float(part["unit_cost"]) * 0.03
        replacement_cost += actual_qty * float(part["unit_cost"])
        expedite_cost += shortage * float(part["expedite_ship_cost"])

        if shortage > 0:
            stockout_parts += 1
        spare_qty = min(actual_qty, brought)
        downtime_part = (
            spare_qty * float(part["replacement_time_hours"])
            + shortage * (float(part["replacement_time_hours"]) + shipping_delay_hours)
        )
        downtime_hours += downtime_part
        satisfaction_penalty += (
            downtime_part / 24.0
        ) * float(part["satisfaction_penalty_per_day"])

    money_cost = carrying_cost + replacement_cost + expedite_cost
    combined_loss = (
        w1_money * money_cost + w2_downtime * downtime_hours + w3_satisfaction * satisfaction_penalty
    )
    return {
        "money_cost": round(money_cost, 3),
        "downtime_hours": round(downtime_hours, 3),
        "satisfaction_penalty": round(satisfaction_penalty, 3),
        "combined_loss": round(combined_loss, 3),
        "stockout_parts": stockout_parts,
    }


def run_simulation(
    client: OracleMockClient,
    request_features: Dict[str, object],
    safety_buffer: float = 0.35,
    w1_money: float = 1.0,
    w2_downtime: float = 40.0,
    w3_satisfaction: float = 1.0,
    shipping_delay_hours: float = 12.0,
    artifact_id: Optional[str] = None,
) -> Dict[str, object]:
    event_id = request_features.get("event_id")
    if event_id:
        features = build_features_from_event(client, str(event_id))
    else:
        features = request_features

    resolved_artifact_id, expected = predict_quantities(features, artifact_id=artifact_id)
    recommended = recommend_spares(expected, safety_buffer=safety_buffer)
    part_map = _part_catalog(client)
    actual_used_qty = (
        _actual_used_from_event(client, str(event_id), expected)
        if event_id
        else {part_id: int(math.ceil(qty)) for part_id, qty in expected.items()}
    )

    scenario_a = _evaluate_scenario(
        bring_qty={part_id: 0 for part_id in expected.keys()},
        actual_used_qty=actual_used_qty,
        part_map=part_map,
        w1_money=w1_money,
        w2_downtime=w2_downtime,
        w3_satisfaction=w3_satisfaction,
        shipping_delay_hours=shipping_delay_hours,
    )
    critical_parts = {part_id: 1 for part_id, qty in expected.items() if qty >= 0.4}
    scenario_b = _evaluate_scenario(
        bring_qty=critical_parts,
        actual_used_qty=actual_used_qty,
        part_map=part_map,
        w1_money=w1_money,
        w2_downtime=w2_downtime,
        w3_satisfaction=w3_satisfaction,
        shipping_delay_hours=shipping_delay_hours,
    )
    scenario_c = _evaluate_scenario(
        bring_qty=recommended,
        actual_used_qty=actual_used_qty,
        part_map=part_map,
        w1_money=w1_money,
        w2_downtime=w2_downtime,
        w3_satisfaction=w3_satisfaction,
        shipping_delay_hours=shipping_delay_hours,
    )

    rec_rows = [
        {
            "part_id": part_id,
            "expected_qty": round(expected_qty, 4),
            "recommended_qty": recommended[part_id],
        }
        for part_id, expected_qty in expected.items()
    ]
    return {
        "artifact_id": resolved_artifact_id,
        "recommendations": rec_rows,
        "actual_used_qty": actual_used_qty,
        "scenarios": [
            {"scenario": "A_bring_0", **scenario_a},
            {"scenario": "B_bring_1_critical", **scenario_b},
            {"scenario": "C_bring_recommended", **scenario_c},
        ],
    }


def compute_exec_summary(
    client: OracleMockClient,
    safety_buffer: float = 0.35,
    w1_money: float = 1.0,
    w2_downtime: float = 40.0,
    w3_satisfaction: float = 1.0,
    shipping_delay_hours: float = 12.0,
    artifact_id: Optional[str] = None,
) -> Dict[str, object]:
    events = client.fetch_events(only_demo=True)
    if not events:
        events = client.fetch_events(limit=50)

    baseline_totals = {"money_cost": 0.0, "downtime_hours": 0.0, "satisfaction_penalty": 0.0, "combined_loss": 0.0}
    rec_totals = {"money_cost": 0.0, "downtime_hours": 0.0, "satisfaction_penalty": 0.0, "combined_loss": 0.0}

    for event in events:
        sim = run_simulation(
            client,
            {"event_id": event["event_id"]},
            safety_buffer=safety_buffer,
            w1_money=w1_money,
            w2_downtime=w2_downtime,
            w3_satisfaction=w3_satisfaction,
            shipping_delay_hours=shipping_delay_hours,
            artifact_id=artifact_id,
        )
        a = sim["scenarios"][0]
        c = sim["scenarios"][2]
        for key in baseline_totals:
            baseline_totals[key] += float(a[key])
            rec_totals[key] += float(c[key])

    deltas = {key: round(rec_totals[key] - baseline_totals[key], 3) for key in baseline_totals}
    baseline_totals = {k: round(v, 3) for k, v in baseline_totals.items()}
    rec_totals = {k: round(v, 3) for k, v in rec_totals.items()}
    return {
        "events_evaluated": len(events),
        "baseline": baseline_totals,
        "recommended": rec_totals,
        "delta_vs_baseline": deltas,
    }
