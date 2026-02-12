# Backend (FastAPI)

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py --n_events 2000 --seed 42
python scripts/train_model.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API
- `GET /health`
- `POST /generate-demo-data`
- `POST /train-model`
- `POST /recommend-parts`
- `POST /simulate`
- `GET /exec-summary`
- `GET /oracle/tables`
- `GET /oracle/table/{name}?limit=50`
