# Grievance Triage

A modular BiLSTM with additive attention for predicting grievance category and urgency.

## Layout

- `src/grievance_triage/config.py`: labels and hyperparameters
- `src/grievance_triage/data.py`: tokenization, vocabulary, and datasets
- `src/grievance_triage/model.py`: model and checkpoint save/load
- `src/grievance_triage/training.py`: training loop, metrics, and early stopping
- `src/grievance_triage/pipeline.py`: CSV orchestration and prediction
- `src/grievance_triage/cli.py`: command-line entry point
- `data/`: place `train.csv` and `test.csv` here
- `notebooks/`: experiments and demonstrations
- `mlflow.db`: local MLflow tracking database
- `.env`: local DagsHub MLflow configuration (not committed)

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
