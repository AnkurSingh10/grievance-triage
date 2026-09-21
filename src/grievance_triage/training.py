import logging
import random

import numpy as np
import torch
import mlflow
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from .config import CATEGORY_LABELS, URGENCY_LABELS
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
    critical_index = URGENCY_LABELS.index("critical")
    best_key, best_state, stale_epochs = (-1.0, -1.0), None, 0

    for epoch in range(1, config.epochs + 1):
        train_loss, *_ = epoch_pass(model, train_loader, config, total_loss, optimizer)
        validation_loss, true_u, predicted_u, true_c, predicted_c = epoch_pass(model, validation_loader, config, total_loss)
        critical_recall = recall_score(true_u, predicted_u, labels=[critical_index], average="macro", zero_division=0)
        macro_f1 = f1_score(true_u, predicted_u, average="macro", zero_division=0)
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f urgency_accuracy=%.4f urgency_macro_f1=%.4f critical_recall=%.4f category_accuracy=%.4f",
                    epoch, train_loss, validation_loss, accuracy_score(true_u, predicted_u), macro_f1,
                    critical_recall, accuracy_score(true_c, predicted_c))
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "urgency_accuracy": accuracy_score(true_u, predicted_u),
                "urgency_macro_f1": macro_f1,
                "critical_recall": critical_recall,
                "category_accuracy": accuracy_score(true_c, predicted_c),
            }, step=epoch)
        key = (critical_recall, macro_f1)
        if key > best_key:
            best_key, best_state, stale_epochs = key, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("early stopping at epoch %d", epoch)
                break
        if scheduler:
            scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
