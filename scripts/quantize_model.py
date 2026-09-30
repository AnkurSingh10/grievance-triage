"""
Quantize the MuRIL model to INT8 to reduce size from ~950MB to ~250MB.
This uses PyTorch dynamic quantization which converts Linear layers to INT8
while keeping the accuracy nearly identical.

Usage:
    python scripts/quantize_model.py
"""
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grievance_triage.ml.model import load_trained_model


def quantize_model():
    checkpoint_path = PROJECT_ROOT / "outputs" / "submission_muril_model.pt"
    output_path = PROJECT_ROOT / "outputs" / "submission_muril_model_quantized.pt"

    print(f"Loading original model from: {checkpoint_path}")
    print(f"Original size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Load the original checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild the model
    from grievance_triage.core.config import Config
    config = Config(**checkpoint["config"])
    config.device = "cpu"

    from grievance_triage.ml.model import MuRILMultiTask
    model = MuRILMultiTask(config, len(checkpoint["category_labels"]), len(checkpoint["urgency_labels"]), from_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Apply dynamic quantization (INT8) to all Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8,
    )

    # Save quantized checkpoint
    quantized_checkpoint = {
        "model_type": checkpoint.get("model_type", "muril"),
        "model_state_dict": quantized_model.state_dict(),
        "config": checkpoint["config"],
        "category_labels": checkpoint["category_labels"],
        "urgency_labels": checkpoint["urgency_labels"],
        "quantized": True,
    }
    torch.save(quantized_checkpoint, output_path)

    print(f"\nQuantized size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Reduction: {(1 - output_path.stat().st_size / checkpoint_path.stat().st_size) * 100:.1f}%")
    print(f"Saved to: {output_path}")

    # Quick sanity check
    print("\nRunning sanity check...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "outputs"))

    test_text = "subject: Road pothole body: Big pothole on main road causing accidents channel: web history: "
    inputs = tokenizer(test_text, truncation=True, padding="max_length", max_length=200, return_tensors="pt")

    with torch.no_grad():
        dept_logits, urg_logits = quantized_model(inputs["input_ids"], inputs["attention_mask"])
        dept_probs = torch.softmax(dept_logits, dim=1)[0]
        urg_probs = torch.softmax(urg_logits, dim=1)[0]

    departments = checkpoint["category_labels"]
    urgencies = checkpoint["urgency_labels"]

    print(f"Department: {departments[dept_probs.argmax()]} ({dept_probs.max():.4f})")
    print(f"Urgency: {urgencies[urg_probs.argmax()]} ({urg_probs.max():.4f})")
    print("\n✅ Quantization complete!")


if __name__ == "__main__":
    quantize_model()
