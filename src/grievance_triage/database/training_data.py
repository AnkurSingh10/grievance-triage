import hashlib
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import TrainingRecord


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload_training_frame(session: Session, frame: pd.DataFrame, path: Path, source_split: str) -> int:
    dataset_hash = _file_hash(path)
    already_uploaded = session.scalar(
        select(TrainingRecord.id)
        .where(TrainingRecord.dataset_hash == dataset_hash, TrainingRecord.source_split == source_split)
        .limit(1)
    )
    if already_uploaded:
        return 0

    records = frame.where(pd.notna(frame), None).to_json(orient="records", date_format="iso")
    rows = [
        TrainingRecord(
            dataset_hash=dataset_hash,
            source_split=source_split,
            source_file=path.name,
            row_number=row_number,
            raw_data=raw_data,
        )
        for row_number, raw_data in enumerate(__import__("json").loads(records), start=1)
    ]
    session.add_all(rows)
    session.commit()
    return len(rows)


def upload_training_data(train_frame: pd.DataFrame, test_frame: pd.DataFrame, train_path: Path, test_path: Path) -> tuple[int, int]:
    from .session import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        train_count = upload_training_frame(session, train_frame, train_path, "train")
        test_count = upload_training_frame(session, test_frame, test_path, "test")
        return train_count, test_count
    finally:
        session.close()