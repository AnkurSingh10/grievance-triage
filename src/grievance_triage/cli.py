import argparse

from .core.config import Config
from .core.logging_utils import configure_logging
from .ml.pipeline import train_and_predict


def main():
    parser = argparse.ArgumentParser(description="Train the grievance triage model and create a submission")
    parser.add_argument("--train", default=None, help="Path to train.csv; defaults to data/train.csv")
    parser.add_argument("--test", default=None, help="Path to test.csv; defaults to data/test.csv")
    parser.add_argument("--out", default=None, help="Submission path; defaults to outputs/submission.csv")
    parser.add_argument("--model-type", choices=["muril", "bilstm"], default="muril", help="Model architecture: 'muril' or 'bilstm'")
    parser.add_argument("--model-name", default="google/muril-base-cased", help="HuggingFace Transformer model name")
    parser.add_argument("--token-level", choices=["char", "word"], default="char")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--critical-boost", type=float, default=3.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = Config(
        model_type=args.model_type,
        model_name=args.model_name,
        token_level=args.token_level,
        epochs=args.epochs,
        max_len=args.max_len,
        critical_boost=args.critical_boost,
    )
    train_and_predict(args.train, args.test, args.out, config)


if __name__ == "__main__":
    main()
