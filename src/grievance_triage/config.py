from dataclasses import dataclass

import torch

CATEGORY_LABELS = [
    "roads_transport", "water_supply", "electricity", "healthcare",
    "education", "welfare_schemes", "law_and_order", "agriculture_irrigation",
]
URGENCY_LABELS = ["routine", "high", "critical"]


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    token_level: str = "char"
    max_vocab: int = 4000
    min_freq: int = 2
    max_len: int = 300
    lowercase: bool = True
    val_size: float = 0.12
    emb_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    attn_dim: int = 64
    dropout: float = 0.3
    pad_idx: int = 0
    head_hidden: int = 128
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    grad_clip: float = 5.0
    patience: int = 8
    w_category: float = 1.0
    w_urgency: float = 1.0
    use_class_weights: bool = True
    critical_boost: float = 3.0
