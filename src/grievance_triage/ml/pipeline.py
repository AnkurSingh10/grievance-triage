import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import dagshub
import pandas as pd
import torch
import mlflow
from mlflow.exceptions import MlflowException
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..core.config import CATEGORY_LABELS, URGENCY_LABELS, Config
from ..database.training_data import upload_training_data
from .data import TestDataset, TextProcessor, build_complaint_text
from .model import BiLSTMAttn, save_checkpoint
from .training import set_seed, train_model
from .tracking import log_config, log_input_file

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(path: str | None, default: str) -> Path:
    resolved = Path(path or default)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _prepare(frame, processor, labeled):
    frame = frame.copy()
    frame["text"] = build_complaint_text(frame)
    frame["x"] = frame["text"].map(processor.encode)
    if labeled:
        frame["category"] = frame["label"].str.split("|").str[0]
        frame["urgency"] = frame["label"].str.split("|").str[1]
        frame["y_cat"] = frame["category"].map({v: i for i, v in enumerate(CATEGORY_LABELS)})
        frame["y_urg"] = frame["urgency"].map({v: i for i, v in enumerate(URGENCY_LABELS)})
        if frame[["y_cat", "y_urg"]].isna().any().any():
            raise ValueError("train.csv contains a label outside the configured category or urgency labels")
        frame[["y_cat", "y_urg"]] = frame[["y_cat", "y_urg"]].astype(int)
    return frame


def train_and_predict(train_path: str | None = None, test_path: str | None = None,
                      submission_path: str | None = None, config: Config | None = None):
    config = config or Config()
    set_seed(config.seed)
    train_path = _project_path(train_path, "data/train.csv")
    test_path = _project_path(test_path, "data/test.csv")
    submission_path = _project_path(submission_path, "outputs/submission.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_REPO_OWNER", "ankurrssingh110"),
        repo_name=os.getenv("DAGSHUB_REPO_NAME", "grievance-triage"),
        mlflow=True,
    )
    tracking_uri = mlflow.get_tracking_uri()
    logger.info("mlflow_tracking_uri=%s", tracking_uri)
    try:
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "grievance-triage"))
    except MlflowException as error:
        raise RuntimeError(
            "DagsHub MLflow tracking is unavailable. Confirm that the repository "
            "ankurrssingh110/grievance-triage exists, then add "
            "MLFLOW_TRACKING_USERNAME and a DagsHub access token as "
            "MLFLOW_TRACKING_PASSWORD in .env."
        ) from error
    with mlflow.start_run() as run:
        mlflow.set_tag("project", "grievance-triage")
        mlflow.set_tag("pipeline", "cleaning-training-prediction")
        log_config(config)
        logger.info("mlflow_run_id=%s", run.info.run_id)
        logger.info("loading train=%s test=%s", train_path, test_path)
        log_input_file(train_path)
        log_input_file(test_path)
        train_frame = pd.read_csv(train_path)
        test_frame = pd.read_csv(test_path)
        uploaded_train, uploaded_test = upload_training_data(train_frame, test_frame, train_path, test_path)
        logger.info("Neon training data upload: train_rows=%d test_rows=%d", uploaded_train, uploaded_test)
        mlflow.log_metrics({"raw_train_rows": len(train_frame), "raw_test_rows": len(test_frame)})
        if "label" not in train_frame.columns:
            raise ValueError("train.csv must contain a 'label' column")

        raw_text = build_complaint_text(train_frame)
        processor = TextProcessor(config)
        processor.fit(raw_text)
        train_frame = _prepare(train_frame, processor, labeled=True)
        test_frame = _prepare(test_frame, processor, labeled=False)
        train_frame.drop(columns=["x"], errors="ignore").to_csv(PROJECT_ROOT / "outputs" / "cleaned_train.csv", index=False)
        test_frame.drop(columns=["x"], errors="ignore").to_csv(PROJECT_ROOT / "outputs" / "cleaned_test.csv", index=False)
        mlflow.log_artifact(str(PROJECT_ROOT / "outputs" / "cleaned_train.csv"), "cleaned_data")
        mlflow.log_artifact(str(PROJECT_ROOT / "outputs" / "cleaned_test.csv"), "cleaned_data")
        mlflow.log_metrics({"clean_train_rows": len(train_frame), "clean_test_rows": len(test_frame), "vocabulary_size": len(processor.itos)})
        train_split, validation_split = train_test_split(train_frame, test_size=config.val_size, random_state=config.seed, stratify=train_frame["label"])
        model = BiLSTMAttn(config, len(processor.itos), len(CATEGORY_LABELS), len(URGENCY_LABELS)).to(config.device)
        logger.info("training %d rows, validating %d rows, vocabulary=%d, device=%s", len(train_split), len(validation_split), len(processor.itos), config.device)
        model = train_model(model, train_split, validation_split, config)

        checkpoint_path = submission_path.with_suffix("").with_name(submission_path.stem + "_model.pt")
        save_checkpoint(model, processor, config, CATEGORY_LABELS, URGENCY_LABELS, checkpoint_path)
        test_loader = DataLoader(TestDataset(test_frame), batch_size=config.batch_size)
        categories, urgencies = [], []
        with torch.no_grad():
            for inputs in test_loader:
                category_logits, urgency_logits, _ = model(inputs.to(config.device))
                categories.extend(category_logits.argmax(1).cpu().tolist())
                urgencies.extend(urgency_logits.argmax(1).cpu().tolist())
        labels = [f"{CATEGORY_LABELS[c]}|{URGENCY_LABELS[u]}" for c, u in zip(categories, urgencies)]
        submission = pd.DataFrame({"id": test_frame["id"], "label": labels})
        submission.to_csv(submission_path, index=False)
        mlflow.log_artifact(str(checkpoint_path), "model")
        mlflow.log_artifact(str(submission_path), "predictions")
        mlflow.set_tag("checkpoint", checkpoint_path.name)
        logger.info("saved submission=%s checkpoint=%s", submission_path, checkpoint_path)
        return model, processor, config
