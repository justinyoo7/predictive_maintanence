from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.modeling import build_features_from_event, predict_quantities, train_and_save_model
from app.oracle_client import oracle_client
from app.schemas import (
    ExecSummaryResponse,
    GenerateDataRequest,
    RecommendPartsRequest,
    RecommendPartsResponse,
    SimulateRequest,
    SimulateResponse,
    TrainModelRequest,
)
from app.simulation import compute_exec_summary, recommend_spares, run_simulation
from scripts.generate_data import GenerationConfig, generate_tables, write_tables

app = FastAPI(title="Predictive Parts Recommendation API", version="0.1.0")

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
    config = GenerationConfig(
        n_events=request.n_events,
        seed=request.seed,
        demo_events=request.demo_events,
        output_dir=oracle_client.data_dir,
    )
    tables = generate_tables(config)
    write_tables(tables, config.output_dir)
    oracle_client.reload()
    return {
        "status": "generated",
        "output_dir": str(config.output_dir),
        "table_counts": {name: len(df) for name, df in tables.items()},
    }


@app.post("/train-model")
def train_model(request: TrainModelRequest) -> dict:
    oracle_client.reload()
    result = train_and_save_model(oracle_client, artifact_name=request.artifact_name)
    return {"status": "trained", **result}


def _extract_features(payload: RecommendPartsRequest) -> dict:
    if payload.event_id:
        return build_features_from_event(oracle_client, payload.event_id)
    return {
        "symptom_text": payload.symptom_text,
        "task_type": payload.task_type,
        "duration_days": payload.duration_days,
        "age_days": payload.age_days,
        "avg_load_factor": payload.avg_load_factor,
        "environment_score": payload.environment_score,
        "region": payload.region,
    }


@app.post("/recommend-parts", response_model=RecommendPartsResponse)
def recommend_parts(payload: RecommendPartsRequest) -> RecommendPartsResponse:
    try:
        features = _extract_features(payload)
        artifact_id, expected = predict_quantities(
            features, artifact_id=payload.artifact_id
        )
        recommended = recommend_spares(expected, safety_buffer=payload.safety_buffer)
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rows = [
        {
            "part_id": part_id,
            "expected_qty": round(qty, 4),
            "recommended_qty": recommended[part_id],
        }
        for part_id, qty in expected.items()
    ]
    return RecommendPartsResponse(
        artifact_id=artifact_id,
        recommendations=rows,
        features=features,
    )


@app.post("/simulate", response_model=SimulateResponse)
def simulate(payload: SimulateRequest) -> SimulateResponse:
    try:
        request_features = payload.model_dump(exclude_none=True)
        result = run_simulation(
            oracle_client,
            request_features=request_features,
            safety_buffer=payload.safety_buffer,
            w1_money=payload.w1_money,
            w2_downtime=payload.w2_downtime,
            w3_satisfaction=payload.w3_satisfaction,
            shipping_delay_hours=payload.shipping_delay_hours,
            artifact_id=payload.artifact_id,
        )
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SimulateResponse(**result)


@app.get("/exec-summary", response_model=ExecSummaryResponse)
def exec_summary(
    safety_buffer: float = 0.35,
    w1_money: float = 1.0,
    w2_downtime: float = 40.0,
    w3_satisfaction: float = 1.0,
    shipping_delay_hours: float = 12.0,
    artifact_id: str | None = None,
) -> ExecSummaryResponse:
    try:
        result = compute_exec_summary(
            oracle_client,
            safety_buffer=safety_buffer,
            w1_money=w1_money,
            w2_downtime=w2_downtime,
            w3_satisfaction=w3_satisfaction,
            shipping_delay_hours=shipping_delay_hours,
            artifact_id=artifact_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExecSummaryResponse(**result)


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
