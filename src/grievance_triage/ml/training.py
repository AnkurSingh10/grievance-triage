import json
import logging
import random

import mlflow
import numpy as np
import torch
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
from .data import GrievanceDataset, MuRILDataset

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def urgency_class_weights(y_urg_array, config):
    if hasattr(y_urg_array, "value_counts"):
        counts = y_urg_array.value_counts().reindex(range(len(URGENCY_LABELS))).fillna(1).values.astype(float)
    else:
        counts = np.bincount(y_urg_array, minlength=len(URGENCY_LABELS)).astype(float)
        counts[counts == 0] = 1.0
    weights = counts.sum() / (len(URGENCY_LABELS) * counts)
    weights[URGENCY_LABELS.index("critical")] *= config.critical_boost
    return torch.tensor(weights, dtype=torch.float, device=config.device)


def build_optimizer(model, config):
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def build_muril_optimizer(model, config):
    encoder_params = list(model.encoder.encoder.layer.parameters())
    head_params = list(model.head_cat.parameters()) + list(model.head_urg.parameters())
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": getattr(config, "lr_encoder", 2e-5)},
            {"params": head_params, "lr": getattr(config, "lr_heads", 1e-3)},
        ],
        weight_decay=config.weight_decay,
    )


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
        rows.extend(
            ",".join([labels[index]] + [str(value) for value in row])
            for index, row in enumerate(result["confusion_matrix"])
        )
        mlflow.log_text("\n".join(rows), f"metrics/{task_name}_confusion_matrix.csv")


def epoch_pass(model, loader, config, loss_fn, optimizer=None):
    """Legacy epoch pass for BiLSTM."""
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
    return (
        total_loss / max(len(loader), 1),
        np.array(true_urgency),
        np.array(predicted_urgency),
        np.array(true_category),
        np.array(predicted_category),
    )


def epoch_pass_muril(model, loader, config, loss_fn, optimizer=None):
    """Epoch pass for MuRIL transformer model."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, ys_u, ps_u, ys_c, ps_c = 0.0, [], [], [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            yc = batch["y_cat"].to(config.device)
            yu = batch["y_urg"].to(config.device)

            if training:
                optimizer.zero_grad()
            cat_logits, urg_logits = model(input_ids, attention_mask)
            loss = loss_fn(cat_logits, urg_logits, yc, yu)
            if training:
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], config.grad_clip
                    )
                optimizer.step()

            total_loss += loss.item()
            ys_u.extend(yu.cpu().numpy())
            ps_u.extend(urg_logits.argmax(1).cpu().numpy())
            ys_c.extend(yc.cpu().numpy())
            ps_c.extend(cat_logits.argmax(1).cpu().numpy())

            if training and ((i + 1) % 25 == 0 or (i + 1) == len(loader)):
                logger.info("  Batch %d/%d - loss: %.4f", i + 1, len(loader), loss.item())
    return (
        total_loss / max(len(loader), 1),
        np.array(ys_u),
        np.array(ps_u),
        np.array(ys_c),
        np.array(ps_c),
    )


def train_model(model, train_frame, validation_frame, config):
    """Legacy BiLSTM training function."""
    train_loader = DataLoader(GrievanceDataset(train_frame), batch_size=config.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(GrievanceDataset(validation_frame), batch_size=config.batch_size)
    weights = urgency_class_weights(train_frame["y_urg"], config) if config.use_class_weights else None
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
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f urgency_accuracy=%.4f urgency_macro_f1=%.4f critical_recall=%.4f category_accuracy=%.4f",
            epoch, train_loss, validation_loss, accuracy_score(true_u, predicted_u), macro_f1,
            critical_recall, accuracy_score(true_c, predicted_c)
        )
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


def train_muril_model(model, tr_df, va_df, tokenizer, config):
    """Fine-tune MuRIL transformer model with progressive layer unfreezing."""
    train_ds = MuRILDataset(tr_df["text"], tr_df["y_cat"].values, tr_df["y_urg"].values, tokenizer, config.max_len)
    val_ds = MuRILDataset(va_df["text"], va_df["y_cat"].values, va_df["y_urg"].values, tokenizer, config.max_len)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size * 2)

    class_w = urgency_class_weights(tr_df["y_urg"].values, config)
    logger.info("Urgency class weights (%s): %s", URGENCY_LABELS, class_w.cpu().numpy().round(3).tolist())

    ce_cat = torch.nn.CrossEntropyLoss()
    ce_urg = torch.nn.CrossEntropyLoss(weight=class_w if config.use_class_weights else None)

    def total_loss(cat_logits, urg_logits, yc, yu):
        return config.w_category * ce_cat(cat_logits, yc) + config.w_urgency * ce_urg(urg_logits, yu)

    optimizer = build_muril_optimizer(model, config)
    crit_idx = URGENCY_LABELS.index("critical")
    best_key, best_state, best_reports, epochs_no_improve = (-1.0, -1.0), None, None, 0

    for epoch in range(config.epochs):
        n_unfrozen = model.set_unfrozen_layers(
            config.initial_unfrozen_layers + config.unfreeze_step_per_epoch * epoch
        )

        train_loss, _, _, _, _ = epoch_pass_muril(model, train_dl, config, total_loss, optimizer)
        val_loss, va_y, va_p, va_yc, va_pc = epoch_pass_muril(model, val_dl, config, total_loss)

        department_metrics, urgency_metrics = log_validation_metrics(
            va_y, va_p, va_yc, va_pc, epoch + 1
        )

        val_acc = accuracy_score(va_y, va_p)
        val_f1 = f1_score(va_y, va_p, average="macro", zero_division=0)
        val_crit_recall = recall_score(va_y, va_p, labels=[crit_idx], average="macro", zero_division=0)
        val_cat_acc = accuracy_score(va_yc, va_pc)

        logger.info(
            "Epoch %2d (unfrozen=%d/%d): train_loss=%.4f val_loss=%.4f val_urg_acc=%.4f val_urg_macroF1=%.4f val_crit_recall=%.4f val_cat_acc=%.4f",
            epoch + 1, n_unfrozen, model.n_encoder_layers, train_loss, val_loss, val_acc, val_f1, val_crit_recall, val_cat_acc
        )

        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_urg_acc": val_acc,
                "val_urg_macroF1": val_f1,
                "val_crit_recall": val_crit_recall,
                "val_cat_acc": val_cat_acc,
            }, step=epoch + 1)

        key = (val_crit_recall, val_f1)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_reports = (department_metrics, urgency_metrics)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                logger.info("Early stopping after %d epochs.", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        log_final_reports(*best_reports)

    return model
