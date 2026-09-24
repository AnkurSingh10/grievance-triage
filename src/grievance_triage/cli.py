import argparse

from .core.config import Config
from .core.logging_utils import configure_logging
from .ml.pipeline import train_and_predict


def main():
    parser = argparse.ArgumentParser(description="Train the grievance triage model and create a submission")
    parser.add_argument("--train", default=None, help="Path to train.csv; defaults to data/train.csv")
    parser.add_argument("--test", default=None, help="Path to test.csv; defaults to data/test.csv")
    parser.add_argument("--out", default=None, help="Submission path; defaults to outputs/submission.csv")
    parser.add_argument("--token-level", choices=["char", "word"], default="char")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--critical-boost", type=float, default=3.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = Config(token_level=args.token_level, epochs=args.epochs, critical_boost=args.critical_boost)
    if args.max_len is not None:
        config.max_len = args.max_len
    elif args.token_level == "word":
        config.max_len = 60
    train_and_predict(args.train, args.test, args.out, config)


if __name__ == "__main__":
    main()
