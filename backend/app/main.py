from __future__ import annotations

import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.modeling import get_model_insights, predict_parts, train_and_save_model
from app.oracle_client import oracle_client
from app.schemas import (
    GenerateDataRequest,
    ModelInsightsResponse,
    RecommendPartsRequest,
    RecommendPartsResponse,
    TrainModelRequest,
)
from scripts.generate_data import GenerationConfig, generate_tables, write_tables

app = FastAPI(title="Kubota Dealer Diagnosis API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate-demo-data")
def generate_demo_data(request: GenerateDataRequest) -> dict:
    seed = request.seed
    if request.seed_mode == "random":
        seed = random.randint(1, 1_000_000)
    config = GenerationConfig(
        n_cases=request.n_cases,
        seed=seed,
        demo_cases=request.demo_cases,
        output_dir=oracle_client.data_dir,
    )
    tables = generate_tables(config)
    write_tables(tables, config.output_dir)
    oracle_client.reload()
    return {
        "status": "generated",
        "output_dir": str(config.output_dir),
        "seed_used": seed,
        "table_counts": {name: len(df) for name, df in tables.items()},
    }


@app.post("/train-model")
def train_model(request: TrainModelRequest) -> dict:
    oracle_client.reload()
    result = train_and_save_model(oracle_client, artifact_name=request.artifact_name)
    return {"status": "trained", **result}


@app.post("/recommend-parts", response_model=RecommendPartsResponse)
def recommend_parts(payload: RecommendPartsRequest) -> RecommendPartsResponse:
    try:
        features = {
            "tractor_model": payload.tractor_model,
            "issue_description": payload.issue_description,
            "severity": payload.severity,
        }
        artifact_id, expected, explanation = predict_parts(
            features, artifact_id=payload.artifact_id
        )
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    top_items = sorted(expected.items(), key=lambda x: x[1], reverse=True)[: payload.top_k]
    rows = [
        {
            "part_id": part_id,
            "score": round(qty, 4),
            "recommended_qty": int(round(max(0.0, qty))),
        }
        for part_id, qty in top_items
    ]
    return RecommendPartsResponse(
        artifact_id=artifact_id,
        recommendations=rows,
        input_features=features,
        explanation=explanation,
    )


@app.get("/model-insights", response_model=ModelInsightsResponse)
def model_insights(artifact_id: str | None = None) -> ModelInsightsResponse:
    try:
        result = get_model_insights(artifact_id=artifact_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelInsightsResponse(**result)


@app.get("/oracle/tables")
def oracle_tables() -> dict:
    return {"tables": oracle_client.table_names()}


@app.get("/oracle/table/{name}")
def oracle_table_preview(name: str, limit: int = Query(default=50, ge=1, le=200)) -> dict:
    try:
        rows = oracle_client.preview_table(name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"table": name, "rows": rows, "limit": limit}
