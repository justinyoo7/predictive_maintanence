from __future__ import annotations

from typing import Dict, List, Optional
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GenerateDataRequest(BaseModel):
    n_cases: int = 2000
    seed: int = 42
    demo_cases: int = 50
    seed_mode: Literal["fixed", "random"] = "fixed"


class TrainModelRequest(BaseModel):
    artifact_name: Optional[str] = None


class RecommendPartsRequest(BaseModel):
    tractor_model: str
    issue_description: str
    severity: int = Field(default=3, ge=1, le=5)
    top_k: int = Field(default=5, ge=1, le=10)
    artifact_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_input(self) -> "RecommendPartsRequest":
        if not self.tractor_model.strip():
            raise ValueError("tractor_model is required")
        if not self.issue_description.strip():
            raise ValueError("issue_description is required")
        return self


class PartPrediction(BaseModel):
    part_id: str
    score: float
    recommended_qty: int


class RecommendPartsResponse(BaseModel):
    artifact_id: str
    recommendations: List[PartPrediction]
    input_features: Dict[str, object]
    explanation: Dict[str, object]


class ModelInsightsResponse(BaseModel):
    artifact_id: str
    n_cases: int
    part_metrics: Dict[str, Dict[str, float | str]]
    global_feature_importance: List[Dict[str, object]]
