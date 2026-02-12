# Kubota-Style Predictive Parts Recommendation Demo

Monorepo structure:

- `backend/` FastAPI + synthetic Oracle-mock dataset + modeling/simulation
- `frontend/` Vite + React + TypeScript demo UI

## 5-minute demo script

1. **Start backend**
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. **Start frontend** (new terminal)
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. In the browser, open frontend and click **Generate Demo Data**.
4. Click **Train Model**.
5. Enter/adjust a symptom in **Case Intake**, run recommendation, then click **Run Simulation**.
6. Click **Load Executive Summary** to view aggregate KPI deltas.

## Backend quick commands

From repo root:

```bash
python3 backend/scripts/generate_data.py --n_events 2000 --seed 42
python3 backend/scripts/train_model.py
```

## Local API base URL for frontend

Frontend reads `VITE_API_BASE_URL` and defaults to:

`http://localhost:8000`
# Predictive Maintenance Sales Demo

End-to-end demo that simulates ingesting data from Oracle, predicts 14-day asset failure risk, translates risk into SKU demand impact, generates inventory recommendations, and reports **model accuracy** as the primary KPI.

- **Oracle** is simulated via CSV extracts in `oracle_exports/`. In production, these would be real Oracle exports.
- **Model accuracy** is computed on a synthetic holdout set (deterministic 80/20 split). In production, labels would come from real failure events.

## Tech Stack

- **Frontend**: React + Vite. Run with `npm run dev`. Set `VITE_API_BASE_URL` for API base URL.
- **Backend**: Python 3.11, FastAPI, SQLite (local). Railway-ready (binds to `0.0.0.0` and uses `PORT` from env).

## Local Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Local Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL=http://localhost:8000` in `.env` if the API is on a different host/port.

## Railway Deploy

1. From project root: `railway init` (or link existing project).
2. Set **root directory** to `backend` in Railway dashboard (or run commands from `backend`).
3. Deploy: `railway up` (or connect GitHub and auto-deploy).
4. In Railway dashboard, set environment variables:
   - `PORT` (usually set automatically by Railway)
   - `DATABASE_URL` (optional; defaults to local SQLite file)
   - `ENVIRONMENT` (optional; default `development`)

Start command is provided by `Procfile`: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/demo/refresh` | Run full pipeline (generate data if needed, ingest, score, recommendations, metrics) |
| GET | `/demo/status` | Last ingestion timestamp, run status |
| GET | `/demo/failure-risk?region=` | Failure risk table (optional region filter) |
| GET | `/demo/recommendations?region=` | Inventory recommendations (optional region filter) |
| GET | `/demo/metrics?region=&threshold=` | Model metrics (accuracy, AUC, precision, recall, F1, confusion matrix) |
| GET | `/demo/kpis?region=` | Aggregated KPIs for UI cards |

## Project Structure

```
/
  backend/          # FastAPI app, SQLite, pipeline, metrics
  frontend/         # React + Vite app
  oracle_exports/   # Auto-generated CSVs (mock Oracle data)
```
