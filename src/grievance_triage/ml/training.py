import logging
import random
import json

import numpy as np
import torch
import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from ..core.config import CATEGORY_LABELS, URGENCY_LABELS
from .data import GrievanceDataset

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def urgency_class_weights(frame, config):
    counts = frame["y_urg"].value_counts().reindex(range(len(URGENCY_LABELS))).fillna(1).values.astype(float)
    weights = counts.sum() / (len(URGENCY_LABELS) * counts)
    weights[URGENCY_LABELS.index("critical")] *= config.critical_boost
    return torch.tensor(weights, dtype=torch.float, device=config.device)


def build_optimizer(model, config):
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(opt, cfg):
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    return None


def classification_metrics(true_values, predicted_values, labels):
    report = classification_report(
        true_values,
        predicted_values,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(true_values, predicted_values),
        "precision_macro": precision_score(true_values, predicted_values, average="macro", zero_division=0),
        "recall_macro": recall_score(true_values, predicted_values, average="macro", zero_division=0),
        "f1_macro": f1_score(true_values, predicted_values, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_values, predicted_values, average="weighted", zero_division=0),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(
            true_values,
            predicted_values,
            labels=list(range(len(labels))),
        ).tolist(),
    }


def log_validation_metrics(true_urgency, predicted_urgency, true_category, predicted_category, epoch):
    department = classification_metrics(true_category, predicted_category, CATEGORY_LABELS)
    urgency = classification_metrics(true_urgency, predicted_urgency, URGENCY_LABELS)
    metrics = {
        "department_accuracy": department["accuracy"],
        "department_precision_macro": department["precision_macro"],
        "department_recall_macro": department["recall_macro"],
        "department_macro_f1": department["f1_macro"],
        "department_weighted_f1": department["f1_weighted"],
        "urgency_accuracy": urgency["accuracy"],
        "urgency_precision_macro": urgency["precision_macro"],
        "urgency_recall_macro": urgency["recall_macro"],
        "urgency_macro_f1": urgency["f1_macro"],
        "urgency_weighted_f1": urgency["f1_weighted"],
    }
    for task_name, result in (("department", department), ("urgency", urgency)):
        for class_name, class_metrics in result["per_class"].items():
            metric_name = class_name.replace("_", "-")
            metrics[f"{task_name}_{metric_name}_precision"] = class_metrics["precision"]
            metrics[f"{task_name}_{metric_name}_recall"] = class_metrics["recall"]
            metrics[f"{task_name}_{metric_name}_f1"] = class_metrics["f1"]
    if mlflow.active_run():
        mlflow.log_metrics(metrics, step=epoch)
    return department, urgency


def log_final_reports(department, urgency):
    if not mlflow.active_run():
        return
    mlflow.log_text(
        json.dumps({"department": department, "urgency": urgency}, indent=2),
        "metrics/classification_report.json",
    )
    for task_name, result in (("department", department), ("urgency", urgency)):
        labels = CATEGORY_LABELS if task_name == "department" else URGENCY_LABELS
        rows = [",".join(["actual/predicted"] + labels)]
        rows.extend(",".join([labels[index]] + [str(value) for value in row])
                    for index, row in enumerate(result["confusion_matrix"]))
        mlflow.log_text("\n".join(rows), f"metrics/{task_name}_confusion_matrix.csv")

def epoch_pass(model, loader, config, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, true_urgency, predicted_urgency = 0.0, [], []
    true_category, predicted_category = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, categories, urgencies in loader:
            inputs, categories, urgencies = (value.to(config.device) for value in (inputs, categories, urgencies))
            if training:
                optimizer.zero_grad()
            category_logits, urgency_logits, _ = model(inputs)
            loss = loss_fn(category_logits, urgency_logits, categories, urgencies)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            total_loss += loss.item()
            true_urgency.extend(urgencies.cpu().numpy())
            predicted_urgency.extend(urgency_logits.argmax(1).cpu().numpy())
            true_category.extend(categories.cpu().numpy())
            predicted_category.extend(category_logits.argmax(1).cpu().numpy())
    return (total_loss / max(len(loader), 1), np.array(true_urgency), np.array(predicted_urgency),
            np.array(true_category), np.array(predicted_category))


def train_model(model, train_frame, validation_frame, config):
    train_loader = DataLoader(GrievanceDataset(train_frame), batch_size=config.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(GrievanceDataset(validation_frame), batch_size=config.batch_size)
    weights = urgency_class_weights(train_frame, config) if config.use_class_weights else None
    category_loss = torch.nn.CrossEntropyLoss()
    urgency_loss = torch.nn.CrossEntropyLoss(weight=weights)

    def total_loss(category_logits, urgency_logits, categories, urgencies):
        return config.w_category * category_loss(category_logits, categories) + config.w_urgency * urgency_loss(urgency_logits, urgencies)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    best_key, best_state, best_reports, stale_epochs = (-1.0, -1.0), None, None, 0

    for epoch in range(1, config.epochs + 1):
        train_loss, *_ = epoch_pass(model, train_loader, config, total_loss, optimizer)
        validation_loss, true_u, predicted_u, true_c, predicted_c = epoch_pass(model, validation_loader, config, total_loss)
        department_metrics, urgency_metrics = log_validation_metrics(
            true_u, predicted_u, true_c, predicted_c, epoch
        )
        critical_recall = urgency_metrics["per_class"]["critical"]["recall"]
        macro_f1 = urgency_metrics["f1_macro"]
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f urgency_accuracy=%.4f urgency_macro_f1=%.4f critical_recall=%.4f category_accuracy=%.4f",
                    epoch, train_loss, validation_loss, accuracy_score(true_u, predicted_u), macro_f1,
                    critical_recall, accuracy_score(true_c, predicted_c))
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "critical_recall": critical_recall,
                "department_accuracy": department_metrics["accuracy"],
            }, step=epoch)
        key = (critical_recall, macro_f1)
        if key > best_key:
            best_key, best_state, best_reports, stale_epochs = (
                key,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                (department_metrics, urgency_metrics),
                0,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("early stopping at epoch %d", epoch)
                break
        if scheduler:
            scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
        log_final_reports(*best_reports)
    return model
