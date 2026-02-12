"""Model training and inference for per-part quantity prediction."""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

from app.oracle_client import OracleMockClient


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
LATEST_POINTER = ARTIFACTS_DIR / "latest_artifact.json"

NUMERIC_COLS = ["duration_days", "age_days", "avg_load_factor", "environment_score"]
CAT_COLS = ["task_type", "region"]


def _prepare_training_frame(client: OracleMockClient) -> pd.DataFrame:
    events = client.get_table("events")
    cases = client.get_table("dealer_cases")[["case_id", "asset_id", "symptom_text", "region"]]
    assets = client.get_table("assets")[["asset_id", "age_days"]]
    used = client.get_table("parts_used")
    parts = client.get_table("parts")

    if events.empty or cases.empty or assets.empty or parts.empty:
        raise RuntimeError("Missing required tables for training. Run data generation first.")

    used_pivot = (
        used.pivot_table(index="event_id", columns="part_id", values="used_qty", aggfunc="sum")
        .fillna(0)
        .reset_index()
    )
    frame = events.merge(cases, on="case_id", how="left").merge(assets, on="asset_id", how="left")
    frame = frame.merge(used_pivot, on="event_id", how="left").fillna(0)

    for part_id in parts["part_id"].tolist():
        if part_id not in frame.columns:
            frame[part_id] = 0.0
    return frame


def _build_feature_matrix(
    frame: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
    encoder: Optional[OneHotEncoder] = None,
    fit: bool = False,
):
    text_data = frame["symptom_text"].fillna("")
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        x_text = vectorizer.fit_transform(text_data)
    else:
        x_text = vectorizer.transform(text_data)

    cat_data = frame[CAT_COLS].fillna("unknown").astype(str)
    if fit or encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        x_cat = encoder.fit_transform(cat_data)
    else:
        x_cat = encoder.transform(cat_data)

    num_data = frame[NUMERIC_COLS].fillna(0.0).astype(float).values
    x_num = csr_matrix(num_data)
    x_all = hstack([x_text, x_cat, x_num], format="csr")
    return x_all, vectorizer, encoder


def train_and_save_model(
    client: OracleMockClient,
    artifact_name: Optional[str] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, object]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    frame = _prepare_training_frame(client)
    x_train, vectorizer, encoder = _build_feature_matrix(frame, fit=True)
    part_ids = client.get_table("parts")["part_id"].tolist()

    models: Dict[str, object] = {}
    model_types: Dict[str, str] = {}
    for part_id in part_ids:
        y = np.maximum(frame[part_id].astype(float).values, 0.0)
        poisson = PoissonRegressor(alpha=0.0005, max_iter=500)
        try:
            poisson.fit(x_train, y)
            models[part_id] = poisson
            model_types[part_id] = "poisson"
        except Exception:
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(x_train.toarray(), y)
            models[part_id] = gbr
            model_types[part_id] = "gbr"

    artifact_id = (
        artifact_name
        if artifact_name
        else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    )
    artifact_path = artifacts_dir / f"{artifact_id}.pkl"
    payload = {
        "artifact_id": artifact_id,
        "part_ids": part_ids,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "models": models,
        "model_types": model_types,
        "numeric_cols": NUMERIC_COLS,
        "cat_cols": CAT_COLS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with artifact_path.open("wb") as f:
        pickle.dump(payload, f)

    with LATEST_POINTER.open("w", encoding="utf-8") as f:
        json.dump({"artifact_id": artifact_id, "path": str(artifact_path)}, f, indent=2)
    return {"artifact_id": artifact_id, "path": str(artifact_path), "n_rows": len(frame)}


def _resolve_artifact_id(artifact_id: Optional[str], artifacts_dir: Path) -> str:
    if artifact_id:
        return artifact_id
    if not LATEST_POINTER.exists():
        raise RuntimeError("No model artifact found. Train a model first.")
    data = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
    return data["artifact_id"]


def load_artifact(artifact_id: Optional[str] = None, artifacts_dir: Path = ARTIFACTS_DIR):
    resolved_id = _resolve_artifact_id(artifact_id, artifacts_dir)
    artifact_path = artifacts_dir / f"{resolved_id}.pkl"
    if not artifact_path.exists():
        raise RuntimeError(f"Artifact '{resolved_id}' not found.")
    with artifact_path.open("rb") as f:
        payload = pickle.load(f)
    return payload


def build_features_from_event(client: OracleMockClient, event_id: str) -> Dict[str, object]:
    event = client.fetch_event(event_id)
    if not event:
        raise ValueError(f"Unknown event_id '{event_id}'.")
    case = event.get("dealer_case", {})
    return {
        "symptom_text": str(case.get("symptom_text", "")),
        "task_type": str(event.get("task_type", "")),
        "duration_days": int(event.get("duration_days", 1)),
        "age_days": int(client.get_table("assets").set_index("asset_id").loc[case.get("asset_id"), "age_days"]),
        "avg_load_factor": float(event.get("avg_load_factor", 1.0)),
        "environment_score": float(event.get("environment_score", 0.5)),
        "region": str(case.get("region", "north")),
    }


def predict_quantities(
    features: Dict[str, object],
    artifact_id: Optional[str] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Tuple[str, Dict[str, float]]:
    artifact = load_artifact(artifact_id, artifacts_dir=artifacts_dir)
    feature_df = pd.DataFrame([features])
    x_pred, _, _ = _build_feature_matrix(
        feature_df,
        vectorizer=artifact["vectorizer"],
        encoder=artifact["encoder"],
        fit=False,
    )

    result: Dict[str, float] = {}
    for part_id in artifact["part_ids"]:
        model = artifact["models"][part_id]
        model_type = artifact["model_types"][part_id]
        if model_type == "gbr":
            pred = model.predict(x_pred.toarray())[0]
        else:
            pred = model.predict(x_pred)[0]
        result[part_id] = float(max(0.0, pred))

    return artifact["artifact_id"], result
