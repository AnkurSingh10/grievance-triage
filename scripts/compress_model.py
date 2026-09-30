"""
Convert the MuRIL checkpoint to float16 to halve its size.
float32 (950MB) -> float16 (~475MB)
No accuracy loss for inference since float16 is sufficient for forward pass.

Usage:
    python scripts/compress_model.py
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def compress_model():
    checkpoint_path = PROJECT_ROOT / "outputs" / "submission_muril_model.pt"
    output_path = PROJECT_ROOT / "outputs" / "submission_muril_model_fp16.pt"

    print(f"Loading: {checkpoint_path}")
    print(f"Original size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Convert all tensors in state_dict to float16
    fp16_state_dict = {}
    for key, tensor in checkpoint["model_state_dict"].items():
        if tensor.is_floating_point():
            fp16_state_dict[key] = tensor.half()
        else:
            fp16_state_dict[key] = tensor  # keep non-float tensors as-is

    # Save compressed checkpoint
    compressed = {
        "model_type": checkpoint.get("model_type", "muril"),
        "model_state_dict": fp16_state_dict,
        "config": checkpoint["config"],
        "category_labels": checkpoint["category_labels"],
        "urgency_labels": checkpoint["urgency_labels"],
        "fp16": True,
    }
    torch.save(compressed, output_path)

    new_size = output_path.stat().st_size / 1024 / 1024
    old_size = checkpoint_path.stat().st_size / 1024 / 1024
    print(f"Compressed size: {new_size:.1f} MB")
    print(f"Reduction: {(1 - new_size / old_size) * 100:.1f}%")
    print(f"Saved to: {output_path}")

    # Sanity check - load and run inference
    print("\nSanity check...")
    from grievance_triage.core.config import Config
    from grievance_triage.ml.model import MuRILMultiTask
    from transformers import AutoTokenizer

    config = Config(**compressed["config"])
    config.device = "cpu"

    model = MuRILMultiTask(config, len(compressed["category_labels"]), len(compressed["urgency_labels"]), from_pretrained=False)
    # Load fp16 weights, convert back to fp32 for CPU inference
    model.load_state_dict(compressed["model_state_dict"], strict=False)
    model.float().eval()  # convert to float32 for CPU computation

    tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "outputs"))
    test_text = "subject: Road pothole body: Big pothole on main road causing accidents channel: web history: "
    inputs = tokenizer(test_text, truncation=True, padding="max_length", max_length=200, return_tensors="pt")

    with torch.no_grad():
        dept_logits, urg_logits = model(inputs["input_ids"], inputs["attention_mask"])
        dept_probs = torch.softmax(dept_logits, dim=1)[0]
        urg_probs = torch.softmax(urg_logits, dim=1)[0]

    departments = compressed["category_labels"]
    urgencies = compressed["urgency_labels"]
    print(f"Department: {departments[dept_probs.argmax()]} ({dept_probs.max():.4f})")
    print(f"Urgency: {urgencies[urg_probs.argmax()]} ({urg_probs.max():.4f})")
    print("\nDone! Compression complete.")


if __name__ == "__main__":
    compress_model()
