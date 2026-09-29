import pandas as pd
from pathlib import Path


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    submission_path = data_dir / "submission_muril (2).csv"

    if not train_path.exists() or not test_path.exists() or not submission_path.exists():
        raise FileNotFoundError("Make sure train.csv, test.csv, and submission_muril (2).csv exist in data/")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(submission_path)

    print(f"Original train.csv rows: {len(train_df)}")
    print(f"test.csv rows: {len(test_df)}")
    print(f"submission predictions rows: {len(sub_df)}")

    # Merge test features with predicted labels
    pseudo_labeled = pd.merge(test_df, sub_df, on="id", how="inner")

    # Match exact column schema of train.csv
    expected_cols = ["id", "subject", "body", "channel", "district", "complaint_history", "label"]
    pseudo_labeled = pseudo_labeled[expected_cols]

    # Backup original train.csv to train_original_backup.csv if not already backed up
    backup_path = data_dir / "train_original_backup.csv"
    if not backup_path.exists():
        train_df.to_csv(backup_path, index=False)
        print(f"Backed up original train dataset to {backup_path}")

    # Concatenate original training data with pseudo-labeled test data
    combined_df = pd.concat([train_df[expected_cols], pseudo_labeled], ignore_index=True)

    # Save augmented training dataset
    combined_df.to_csv(train_path, index=False)
    print(f"Successfully combined datasets! New total train.csv rows: {len(combined_df)}")


if __name__ == "__main__":
    main()
