# Grievance Triage

A modular BiLSTM with additive attention for predicting grievance category and urgency.

## Layout

- `src/grievance_triage/core/`: configuration, settings, and shared utilities
- `src/grievance_triage/database/`: Neon session and PostgreSQL/pgvector models
- `src/grievance_triage/ml/`: BiLSTM model, data, training, inference, embeddings, and MLflow
- `src/grievance_triage/services/`: complaint grouping, priority/SLA, and Gemini RAG
- `src/grievance_triage/api/`: FastAPI application, routes, and Pydantic schemas
- `src/grievance_triage/cli.py`: model training command-line entry point
- `data/`: place `train.csv` and `test.csv` here
- `notebooks/`: experiments and demonstrations
- `mlflow.db`: local MLflow tracking database
- `.env`: local DagsHub MLflow configuration (not committed)
- `streamlit_app.py`: initial citizen submission dashboard
- `docker-compose.yml`: API and dashboard services

## Run

```powershell
pip install -r requirements.txt
pip install -e .
python -m grievance_triage.cli
```

The command automatically finds `data/train.csv` and `data/test.csv`, even when launched from the `src` directory. It writes the submission to `outputs/submission.csv` and the matching model checkpoint to `outputs/submission_model.pt`. Logs include each epoch's loss and validation metrics.

Each run is tracked in the DagsHub MLflow experiment `grievance-triage`. DagsHub is initialized from `.env`:

```dotenv
DAGSHUB_REPO_OWNER=ankurrssingh110
DAGSHUB_REPO_NAME=grievance-triage
MLFLOW_EXPERIMENT_NAME=grievance-triage
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
```

Use a DagsHub access token as the password. Keep the username and token only in `.env` or environment variables. MLflow records the raw input CSVs and SHA-256 hashes, cleaned CSVs, configuration parameters, row counts, vocabulary size, epoch metrics, the trained checkpoint, and predictions.

Remote tracking is required by default with `MLFLOW_FALLBACK_LOCAL=false`. If DagsHub is unavailable or credentials are invalid, the command stops before training instead of creating a local MLflow run.

View the runs in DagsHub:

```text
https://dagshub.com/ankurrssingh110/grievance-triage.mlflow
```

Use `--epochs 1` for a quick smoke test, or pass custom paths when needed:

```powershell
python -m grievance_triage.cli --epochs 30
python -m grievance_triage.cli --train path/to/train.csv --test path/to/test.csv --out outputs/custom.csv
```

## Grievance Management System

The existing BiLSTM remains the classifier. The service layer adds:

- Neon PostgreSQL storage using the `NEON_DB` connection string
- raw `train.csv` and `test.csv` rows in the Neon `training_records` table
- pgvector complaint embeddings with department-filtered cosine similarity
- duplicate/related issue grouping and configurable similarity threshold
- issue groups, status history, SLA deadlines, priorities, and officer feedback
- Gemini officer assistance grounded in department policies and resolved cases

Required `.env` values:

```dotenv
NEON_DB=postgresql://user:password@host/database?sslmode=require
GOOGLE_API_KEY=your-google-api-key
MODEL_CHECKPOINT=outputs/submission_model.pt
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
COMPLAINT_SIMILARITY_THRESHOLD=0.85
```

`EMBEDDING_DIMENSION` is validated against the selected Sentence Transformer model. The `0.85` similarity threshold is an initial tunable value, not a production claim; calibrate it with labeled duplicate data.

Train the BiLSTM first so `outputs/submission_model.pt` exists, then start the API:

```powershell
pip install -e .
python -m grievance_triage.api.main
```

On startup the API enables the Neon `vector` extension and creates the application tables and HNSW cosine indexes. Open `http://localhost:8000/docs` for the API contract.

Running the training command also uploads the raw training and test rows to Neon. Each row stores its split, source filename, file hash, row number, and original JSON data. File hashes make the upload idempotent, so rerunning training does not duplicate an unchanged dataset. District is preserved in the raw database row when present, but remains excluded from BiLSTM text, inference, and embeddings.

Start the initial Streamlit citizen interface separately:

```powershell
streamlit run streamlit_app.py
```

Important endpoints:

```text
POST  /complaints
GET   /complaints/{id}
GET   /complaints/{id}/status
GET   /complaints/{id}/related
PATCH /complaints/{id}/status
POST  /complaints/{id}/assist
POST  /knowledge-documents
POST  /officer/feedback
GET   /departments/{department}/complaints
GET   /departments/{department}/dashboard
GET   /analytics/issues
```

The Gemini assistant is not used for duplicate detection. Duplicate detection uses pgvector within the predicted department; Gemini is used only for officer assistance and is instructed to cite retrieved policy/case sources.
