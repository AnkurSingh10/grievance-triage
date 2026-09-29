from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


def build_complaint_text(frame):
    """Build the classifier input from the complaint's text and context fields."""
    def column(name):
        return frame[name].fillna("").astype(str) if name in frame else ""

    return (
        "subject: " + column("subject")
        + " body: " + column("body")
        + " channel: " + column("channel")
        + " history: " + column("complaint_history")
    )


class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.itos = ["<PAD>", "<UNK>"]
        self.stoi = {"<PAD>": 0, "<UNK>": 1}

    def tokenize(self, text: str) -> list[str]:
        text = text.lower() if self.config.lowercase else text
        return list(text) if self.config.token_level == "char" else text.split()

    def fit(self, texts) -> None:
        counter = Counter(token for text in texts for token in self.tokenize(text))
        self.itos += [token for token, count in counter.most_common(self.config.max_vocab)
                      if count >= self.config.min_freq]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(token, 1) for token in self.tokenize(text)]
        ids = ids[: self.config.max_len]
        return ids + [0] * (self.config.max_len - len(ids))


class GrievanceDataset(Dataset):
    """Legacy Dataset for BiLSTM model."""
    def __init__(self, frame):
        self.x = torch.tensor(np.stack(frame["x"].values), dtype=torch.long)
        self.y_category = torch.tensor(frame["y_cat"].values, dtype=torch.long)
        self.y_urgency = torch.tensor(frame["y_urg"].values, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y_category[index], self.y_urgency[index]


class TestDataset(Dataset):
    """Legacy TestDataset for BiLSTM model."""
    def __init__(self, frame):
        self.x = torch.tensor(np.stack(frame["x"].values), dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


class MuRILDataset(Dataset):
    """Dataset for MuRIL transformer model."""

    def __init__(self, texts, y_cat, y_urg, tokenizer, max_len=200):
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors=None,
        )
        self.y_cat = y_cat
        self.y_urg = y_urg

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {
            "input_ids": torch.tensor(self.enc["input_ids"][i], dtype=torch.long),
            "attention_mask": torch.tensor(self.enc["attention_mask"][i], dtype=torch.long),
        }
        if self.y_cat is not None:
            item["y_cat"] = torch.tensor(int(self.y_cat[i]), dtype=torch.long)
            item["y_urg"] = torch.tensor(int(self.y_urg[i]), dtype=torch.long)
        return item
