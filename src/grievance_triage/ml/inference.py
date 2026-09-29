from functools import lru_cache

import torch

from .model import load_trained_model
from ..core.settings import get_settings


@lru_cache
def get_classifier():
    settings = get_settings()
    return load_trained_model(str(settings.checkpoint_path))


def predict_complaint(subject: str, body: str, channel: str = "web", complaint_history: list | None = None) -> dict:
    model, processor_or_tokenizer, config, departments, urgencies = get_classifier()
    history_text = " ".join(str(item) for item in (complaint_history or []))
    text = f"subject: {subject} body: {body} channel: {channel} history: {history_text}"

    model_type = getattr(config, "model_type", "muril")

    if model_type == "muril":
        tokenizer = processor_or_tokenizer
        max_len = getattr(config, "max_len", 200)
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(config.device)
        attention_mask = inputs["attention_mask"].to(config.device)
        with torch.no_grad():
            department_logits, urgency_logits = model(input_ids, attention_mask)
            department_probabilities = torch.softmax(department_logits, dim=1)[0]
            urgency_probabilities = torch.softmax(urgency_logits, dim=1)[0]
    else:
        processor = processor_or_tokenizer
        inputs = torch.tensor([processor.encode(text)], dtype=torch.long, device=config.device)
        with torch.no_grad():
            department_logits, urgency_logits, _ = model(inputs)
            department_probabilities = torch.softmax(department_logits, dim=1)[0]
            urgency_probabilities = torch.softmax(urgency_logits, dim=1)[0]

    department_index = int(department_probabilities.argmax())
    urgency_index = int(urgency_probabilities.argmax())
    return {
        "department": departments[department_index],
        "department_confidence": float(department_probabilities[department_index]),
        "urgency": urgencies[urgency_index],
        "urgency_confidence": float(urgency_probabilities[urgency_index]),
    }
