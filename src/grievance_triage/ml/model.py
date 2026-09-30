from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertConfig

from ..core.config import Config


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
    """Preserved legacy BiLSTM model with attention (kept for comparison)."""

    def __init__(self, cfg: Config, vocab_size, n_cat, n_urg):
        super().__init__()
        H = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.embedding = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            cfg.emb_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
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


class MuRILMultiTask(nn.Module):
    """Google MuRIL fine-tuning architecture for multi-task grievance triage."""

    def __init__(self, cfg: Config, n_cat: int, n_urg: int, from_pretrained: bool = True):
        super().__init__()
        self.model_name = getattr(cfg, "model_name", "google/muril-base-cased")

        if from_pretrained:
            # Training mode: download full pretrained weights from HuggingFace
            self.encoder = AutoModel.from_pretrained(self.model_name)
        else:
            # Inference mode: create empty model shell using known MuRIL config
            # (no network call, no 500MB download — all weights come from checkpoint)
            encoder_config = BertConfig(
                vocab_size=197285,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512,
            )
            self.encoder = AutoModel.from_config(encoder_config)

        H = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(cfg.dropout)
        self.head_cat = nn.Sequential(
            nn.Linear(H, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, n_cat),
        )
        self.head_urg = nn.Sequential(
            nn.Linear(H, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, n_urg),
        )

        self.n_encoder_layers = len(self.encoder.encoder.layer)
        self.freeze_encoder_embeddings()
        self.set_unfrozen_layers(getattr(cfg, "initial_unfrozen_layers", 4))

    def freeze_encoder_embeddings(self):
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

    def set_unfrozen_layers(self, n_unfrozen: int):
        n_unfrozen = min(n_unfrozen, self.n_encoder_layers)
        for i, layer in enumerate(self.encoder.encoder.layer):
            unfreeze = i >= (self.n_encoder_layers - n_unfrozen)
            for p in layer.parameters():
                p.requires_grad = unfreeze
        return n_unfrozen

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.head_cat(cls), self.head_urg(cls)


def save_checkpoint(model, processor, config, category_labels, urgency_labels, path):
    checkpoint = {
        "model_type": getattr(config, "model_type", "muril"),
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "category_labels": category_labels,
        "urgency_labels": urgency_labels,
    }
    if hasattr(processor, "itos"):
        checkpoint["itos"] = processor.itos
    torch.save(checkpoint, path)


def load_trained_model(path, device=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from ..core.config import Config
    from .data import TextProcessor

    config = Config(**checkpoint["config"])
    if device:
        config.device = device
    elif not torch.cuda.is_available():
        config.device = "cpu"
    model_type = checkpoint.get("model_type", getattr(config, "model_type", "muril"))

    if model_type == "muril":
        # Use from_pretrained=False to avoid downloading 500MB base model;
        # all weights come from the checkpoint instead.
        model = MuRILMultiTask(config, len(checkpoint["category_labels"]), len(checkpoint["urgency_labels"]), from_pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # Ensure float32 for CPU inference (needed if checkpoint was saved in fp16)
        model.float().to(config.device).eval()
        outputs_dir = Path(path).parent
        if (outputs_dir / "tokenizer.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(outputs_dir))
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        return model, tokenizer, config, checkpoint["category_labels"], checkpoint["urgency_labels"]
    else:
        model = BiLSTMAttn(config, len(checkpoint["itos"]), len(checkpoint["category_labels"]), len(checkpoint["urgency_labels"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.device).eval()
        processor = TextProcessor(config)
        processor.itos = checkpoint["itos"]
        processor.stoi = {word: index for index, word in enumerate(processor.itos)}
        return model, processor, config, checkpoint["category_labels"], checkpoint["urgency_labels"]
