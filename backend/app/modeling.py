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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from app.oracle_client import OracleMockClient


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
LATEST_POINTER = ARTIFACTS_DIR / "latest_artifact.json"

NUMERIC_COLS = ["duration_days", "age_days", "avg_load_factor", "environment_score"]
CAT_COLS = ["task_type", "region"]

SYMPTOM_HINTS = {
    "wheel": ["tire", "wheel", "rim", "pressure", "puncture", "traction"],
    "body": ["body", "frame", "chassis", "panel", "structural"],
    "windshield": ["glass", "windshield", "crack", "visibility", "chip"],
    "engine": ["engine", "knocking", "stall", "torque", "smoke", "power loss"],
}


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
    x_train_dense = x_train.toarray()
    part_ids = client.get_table("parts")["part_id"].tolist()

    models: Dict[str, object] = {}
    model_types: Dict[str, str] = {}
    model_mse: Dict[str, Dict[str, float]] = {}
    for part_id in part_ids:
        y = np.maximum(frame[part_id].astype(float).values, 0.0)
        poisson = PoissonRegressor(alpha=0.0001, max_iter=800)
        poisson_mse = float("inf")
        try:
            poisson.fit(x_train, y)
            poisson_mse = mean_squared_error(y, np.maximum(0.0, poisson.predict(x_train)))
        except Exception:
            poisson = None

        gbr = GradientBoostingRegressor(
            random_state=42,
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            min_samples_leaf=4,
        )
        gbr.fit(x_train_dense, y)
        gbr_mse = mean_squared_error(y, np.maximum(0.0, gbr.predict(x_train_dense)))

        # Pick the better in-sample fit for this synthetic demo.
        if poisson is not None and poisson_mse <= gbr_mse * 0.98:
            models[part_id] = poisson
            model_types[part_id] = "poisson"
        else:
            models[part_id] = gbr
            model_types[part_id] = "gbr"
        model_mse[part_id] = {
            "poisson_mse": float(poisson_mse) if np.isfinite(poisson_mse) else -1.0,
            "gbr_mse": float(gbr_mse),
        }

    feature_medians = {
        "duration_days": float(frame["duration_days"].median()),
        "age_days": float(frame["age_days"].median()),
        "avg_load_factor": float(frame["avg_load_factor"].median()),
        "environment_score": float(frame["environment_score"].median()),
    }
    tps = client.get_table("task_part_strain")
    task_part_strain: Dict[str, Dict[str, float]] = {}
    if not tps.empty:
        for _, row in tps.iterrows():
            task = str(row["task_type"])
            part = str(row["part_id"])
            task_part_strain.setdefault(task, {})[part] = float(row["strain_multiplier"])

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
        "model_mse": model_mse,
        "numeric_cols": NUMERIC_COLS,
        "cat_cols": CAT_COLS,
        "feature_medians": feature_medians,
        "task_part_strain": task_part_strain,
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
    prepared = {
        "symptom_text": str(features.get("symptom_text") or ""),
        "task_type": str(features.get("task_type") or "unknown"),
        "duration_days": float(features.get("duration_days") or 0.0),
        "age_days": float(features.get("age_days") or 0.0),
        "avg_load_factor": float(features.get("avg_load_factor") or 0.0),
        "environment_score": float(features.get("environment_score") or 0.0),
        "region": str(features.get("region") or "unknown"),
    }
    feature_df = pd.DataFrame([prepared])
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
        adjusted = float(max(0.0, pred)) * _calibrate_prediction(artifact, prepared, part_id)
        result[part_id] = float(max(0.0, adjusted))

    return artifact["artifact_id"], result


def _calibrate_prediction(artifact: Dict[str, object], features: Dict[str, object], part_id: str) -> float:
    med = artifact.get("feature_medians", {}) or {}
    task_map = artifact.get("task_part_strain", {}) or {}
    duration_med = max(float(med.get("duration_days", 3.0)), 1.0)
    age_med = max(float(med.get("age_days", 1200.0)), 1.0)
    load_med = float(med.get("avg_load_factor", 0.85))
    env_med = float(med.get("environment_score", 0.55))

    duration = float(features.get("duration_days", duration_med))
    age_days = float(features.get("age_days", age_med))
    load = float(features.get("avg_load_factor", load_med))
    env = float(features.get("environment_score", env_med))
    task_type = str(features.get("task_type", ""))
    symptom_text = str(features.get("symptom_text", "")).lower()

    duration_delta = (duration - duration_med) / duration_med
    age_delta = (age_days - age_med) / age_med
    load_delta = load - load_med
    env_delta = env - env_med

    task_multiplier = float(task_map.get(task_type, {}).get(part_id, 1.0))
    task_delta = task_multiplier - 1.0

    symptom_boost = 0.0
    for token in SYMPTOM_HINTS.get(part_id, []):
        if token in symptom_text:
            symptom_boost = 0.30
            break

    calibrated = (
        1.0
        + 0.45 * duration_delta
        + 0.20 * age_delta
        + 0.40 * load_delta
        + 0.30 * env_delta
        + 0.70 * task_delta
        + symptom_boost
    )
    return float(np.clip(calibrated, 0.35, 3.2))
