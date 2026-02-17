"""Model training and inference for dealer-side diagnosis."""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

NUMERIC_COLS = ["severity"]
CAT_COLS = ["tractor_model"]
TEXT_COL = "issue_description"


def _prepare_training_frame(client: OracleMockClient) -> pd.DataFrame:
    cases = client.get_table("dealer_cases")[
        ["case_id", "tractor_model", "issue_description", "severity"]
    ]
    ordered = client.get_table("parts_ordered")
    parts = client.get_table("parts")

    if cases.empty or ordered.empty or parts.empty:
        raise RuntimeError("Missing required tables for training. Run data generation first.")

    ordered_pivot = (
        ordered.pivot_table(
            index="case_id", columns="part_id", values="ordered_qty", aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )
    frame = cases.merge(ordered_pivot, on="case_id", how="left").fillna(0)

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
    text_data = frame[TEXT_COL].fillna("")
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
    x_dense = x_train.toarray()
    part_ids = client.get_table("parts")["part_id"].tolist()

    models: Dict[str, object] = {}
    model_types: Dict[str, str] = {}
    model_rmse: Dict[str, Dict[str, float | str]] = {}
    split_idx = max(1, int(len(frame) * 0.8))
    x_fit = x_train[:split_idx]
    x_eval = x_train[split_idx:] if split_idx < len(frame) else x_train[:1]
    x_fit_dense = x_dense[:split_idx]
    x_eval_dense = x_dense[split_idx:] if split_idx < len(frame) else x_dense[:1]

    for part_id in part_ids:
        y = np.maximum(frame[part_id].astype(float).values, 0.0)
        y_fit = y[:split_idx]
        y_eval = y[split_idx:] if split_idx < len(frame) else y[:1]
        poisson = PoissonRegressor(alpha=0.0001, max_iter=800)
        poisson_rmse = float("inf")
        try:
            poisson.fit(x_fit, y_fit)
            poisson_pred = np.maximum(0.0, poisson.predict(x_eval))
            poisson_rmse = float(np.sqrt(mean_squared_error(y_eval, poisson_pred)))
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
        gbr.fit(x_fit_dense, y_fit)
        gbr_pred = np.maximum(0.0, gbr.predict(x_eval_dense))
        gbr_rmse = float(np.sqrt(mean_squared_error(y_eval, gbr_pred)))

        # Pick the better holdout fit for this synthetic demo.
        if poisson is not None and poisson_rmse <= gbr_rmse * 0.98:
            models[part_id] = poisson
            model_types[part_id] = "poisson"
            chosen_rmse = poisson_rmse
        else:
            models[part_id] = gbr
            model_types[part_id] = "gbr"
            chosen_rmse = gbr_rmse
        model_rmse[part_id] = {
            "poisson_rmse": float(poisson_rmse) if np.isfinite(poisson_rmse) else -1.0,
            "gbr_rmse": float(gbr_rmse),
            "selected_model": model_types[part_id],
            "selected_rmse": float(chosen_rmse),
        }

    feature_names = _feature_names(vectorizer, encoder, NUMERIC_COLS)
    global_importance = _compute_global_importance(
        models=models,
        model_types=model_types,
        feature_names=feature_names,
        text_feature_count=len(vectorizer.get_feature_names_out()),
    )

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
        "model_rmse": model_rmse,
        "numeric_cols": NUMERIC_COLS,
        "cat_cols": CAT_COLS,
        "text_col": TEXT_COL,
        "global_feature_importance": global_importance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": int(len(frame)),
    }
    with artifact_path.open("wb") as f:
        pickle.dump(payload, f)

    with LATEST_POINTER.open("w", encoding="utf-8") as f:
        json.dump({"artifact_id": artifact_id, "path": str(artifact_path)}, f, indent=2)
    return {
        "artifact_id": artifact_id,
        "path": str(artifact_path),
        "n_rows": len(frame),
        "part_metrics": model_rmse,
        "global_feature_importance": global_importance,
    }


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


def predict_parts(
    features: Dict[str, object],
    artifact_id: Optional[str] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Tuple[str, Dict[str, float], Dict[str, object]]:
    artifact = load_artifact(artifact_id, artifacts_dir=artifacts_dir)
    prepared = {
        "issue_description": str(features.get("issue_description") or ""),
        "tractor_model": str(features.get("tractor_model") or "unknown"),
        "severity": float(features.get("severity") or 3.0),
    }
    feature_df = pd.DataFrame([prepared])
    x_pred, _, _ = _build_feature_matrix(
        feature_df,
        vectorizer=artifact["vectorizer"],
        encoder=artifact["encoder"],
        fit=False,
    )

    x_pred_dense = x_pred.toarray()
    result: Dict[str, float] = {}
    token_contribs: Dict[str, List[Dict[str, float | str]]] = {}
    for part_id in artifact["part_ids"]:
        model = artifact["models"][part_id]
        model_type = artifact["model_types"][part_id]
        if model_type == "gbr":
            pred = model.predict(x_pred_dense)[0]
        else:
            pred = model.predict(x_pred)[0]
        result[part_id] = float(max(0.0, pred))
        token_contribs[part_id] = _text_contributions_for_part(
            part_id=part_id,
            artifact=artifact,
            x_pred=x_pred,
        )

    explanation = {
        "top_terms_by_part": token_contribs,
        "global_feature_importance": artifact.get("global_feature_importance", []),
    }
    return artifact["artifact_id"], result, explanation


def get_model_insights(
    artifact_id: Optional[str] = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, object]:
    artifact = load_artifact(artifact_id, artifacts_dir=artifacts_dir)
    return {
        "artifact_id": artifact["artifact_id"],
        "n_cases": int(artifact.get("n_cases", 0)),
        "part_metrics": artifact.get("model_rmse", {}),
        "global_feature_importance": artifact.get("global_feature_importance", []),
    }


def _feature_names(
    vectorizer: TfidfVectorizer,
    encoder: OneHotEncoder,
    numeric_cols: List[str],
) -> List[str]:
    text_names = [f"text::{n}" for n in vectorizer.get_feature_names_out()]
    cat_names = [f"cat::{n}" for n in encoder.get_feature_names_out(CAT_COLS)]
    num_names = [f"num::{c}" for c in numeric_cols]
    return text_names + cat_names + num_names


def _compute_global_importance(
    models: Dict[str, object],
    model_types: Dict[str, str],
    feature_names: List[str],
    text_feature_count: int,
) -> List[Dict[str, object]]:
    accum = np.zeros(len(feature_names), dtype=float)
    for part_id, model in models.items():
        model_type = model_types[part_id]
        if model_type == "poisson" and hasattr(model, "coef_"):
            importances = np.abs(np.asarray(model.coef_))
        elif model_type == "gbr" and hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        else:
            continue
        if importances.shape[0] != accum.shape[0]:
            continue
        accum += importances

    if len(models) > 0:
        accum = accum / float(len(models))

    top_idx = np.argsort(accum)[::-1][:12]
    rows: List[Dict[str, object]] = []
    for idx in top_idx:
        if accum[idx] <= 0:
            continue
        feature_name = feature_names[idx]
        bucket = "text" if idx < text_feature_count else "meta"
        rows.append(
            {
                "feature": feature_name,
                "importance": round(float(accum[idx]), 6),
                "bucket": bucket,
            }
        )
    return rows


def _text_contributions_for_part(
    part_id: str,
    artifact: Dict[str, object],
    x_pred: csr_matrix,
) -> List[Dict[str, float | str]]:
    vectorizer = artifact["vectorizer"]
    text_features = vectorizer.get_feature_names_out()
    text_n = len(text_features)
    row = x_pred.toarray()[0]
    text_values = row[:text_n]
    model = artifact["models"][part_id]
    model_type = artifact["model_types"][part_id]

    if model_type == "poisson" and hasattr(model, "coef_"):
        signal = text_values * np.asarray(model.coef_[:text_n])
    elif model_type == "gbr" and hasattr(model, "feature_importances_"):
        signal = text_values * np.asarray(model.feature_importances_[:text_n])
    else:
        signal = text_values

    top_idx = np.argsort(np.abs(signal))[::-1][:5]
    rows: List[Dict[str, float | str]] = []
    for idx in top_idx:
        val = float(signal[idx])
        if abs(val) <= 1e-9:
            continue
        rows.append({"term": str(text_features[idx]), "contribution": round(val, 6)})
    return rows
