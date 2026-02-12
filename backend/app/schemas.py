from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class GenerateDataRequest(BaseModel):
    n_events: int = 2000
    seed: int = 42
    demo_events: int = 50


class TrainModelRequest(BaseModel):
    artifact_name: Optional[str] = None


class RecommendPartsRequest(BaseModel):
    event_id: Optional[str] = None
    symptom_text: Optional[str] = None
    task_type: Optional[str] = None
    duration_days: Optional[int] = None
    age_days: Optional[int] = None
    avg_load_factor: Optional[float] = None
    environment_score: Optional[float] = None
    region: Optional[str] = None
    safety_buffer: float = 0.35
    artifact_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_input(self) -> "RecommendPartsRequest":
        if self.event_id:
            return self
        required = [
            self.symptom_text,
            self.task_type,
            self.duration_days,
            self.age_days,
            self.avg_load_factor,
            self.environment_score,
            self.region,
        ]
        if any(v is None for v in required):
            raise ValueError(
                "Provide event_id or all direct features for recommendation."
            )
        return self


class PartPrediction(BaseModel):
    part_id: str
    expected_qty: float
    recommended_qty: int


class RecommendPartsResponse(BaseModel):
    artifact_id: str
    recommendations: List[PartPrediction]
    features: Dict[str, object]


class SimulateRequest(BaseModel):
    event_id: Optional[str] = None
    symptom_text: Optional[str] = None
    task_type: Optional[str] = None
    duration_days: Optional[int] = None
    age_days: Optional[int] = None
    avg_load_factor: Optional[float] = None
    environment_score: Optional[float] = None
    region: Optional[str] = None
    safety_buffer: float = 0.35
    w1_money: float = 1.0
    w2_downtime: float = 40.0
    w3_satisfaction: float = 1.0
    shipping_delay_hours: float = 12.0
    artifact_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_input(self) -> "SimulateRequest":
        if self.event_id:
            return self
        required = [
            self.symptom_text,
            self.task_type,
            self.duration_days,
            self.age_days,
            self.avg_load_factor,
            self.environment_score,
            self.region,
        ]
        if any(v is None for v in required):
            raise ValueError("Provide event_id or all direct features for simulation.")
        return self


class ScenarioKPI(BaseModel):
    scenario: str
    money_cost: float
    downtime_hours: float
    satisfaction_penalty: float
    combined_loss: float
    stockout_parts: int = Field(default=0)


class SimulateResponse(BaseModel):
    artifact_id: str
    recommendations: List[PartPrediction]
    actual_used_qty: Dict[str, int]
    scenarios: List[ScenarioKPI]


class ExecSummaryResponse(BaseModel):
    events_evaluated: int
    baseline: Dict[str, float]
    recommended: Dict[str, float]
    delta_vs_baseline: Dict[str, float]
