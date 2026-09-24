from dataclasses import asdict

import torch
import torch.nn as nn
from ..core.config import Config
import torch.nn.functional as F

class MultiLayerFCNN(nn.Module):
    """MLP head: 2 hidden layers, BatchNorm + ReLU + dropout."""

    def __init__(self, in_features, out_features, hidden=128, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc_out = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc_out(x)


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM outputs."""

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H, mask):
        e = self.v(torch.tanh(self.W(H))).squeeze(-1)
        e = e.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return context, alpha


class BiLSTMAttn(nn.Module):
    def __init__(self, cfg: Config, vocab_size, n_cat, n_urg):
        super().__init__()
        H = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.embedding = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(cfg.emb_dim, cfg.hidden_dim, num_layers=cfg.num_layers,
                             batch_first=True, bidirectional=cfg.bidirectional,
                             dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.attn = Attention(H, cfg.attn_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_cat = MultiLayerFCNN(H, n_cat, hidden=cfg.head_hidden, dropout_rate=cfg.dropout)
        self.head_urg = MultiLayerFCNN(H, n_urg, hidden=cfg.head_hidden, dropout_rate=cfg.dropout)

    def forward(self, x):
        mask = (x != 0).long()
        emb = self.dropout(self.embedding(x))
        H, _ = self.lstm(emb)
        context, alpha = self.attn(H, mask)
        context = self.dropout(context)
        return self.head_cat(context), self.head_urg(context), alpha


def save_checkpoint(model, processor, config, category_labels, urgency_labels, path):
    torch.save({
        "model_state_dict": model.state_dict(), "itos": processor.itos,
        "config": asdict(config), "category_labels": category_labels,
        "urgency_labels": urgency_labels,
    }, path)


def load_trained_model(path, device=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from ..core.config import Config
    from .data import TextProcessor
    config = Config(**checkpoint["config"])
    config.device = device or config.device
    model = BiLSTMAttn(config, len(checkpoint["itos"]), len(checkpoint["category_labels"]), len(checkpoint["urgency_labels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device).eval()
    processor = TextProcessor(config)
    processor.itos = checkpoint["itos"]
    processor.stoi = {word: index for index, word in enumerate(processor.itos)}
    return model, processor, config, checkpoint["category_labels"], checkpoint["urgency_labels"]
