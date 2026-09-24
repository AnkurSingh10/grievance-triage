"""Seed Neon with searchable, embedded records from data/train.csv.

This uses the training labels as ground truth and never calls the BiLSTM.
District is stored as metadata but is excluded from the embedding text.
"""

import argparse
import hashlib
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from grievance_triage.database.models import Complaint, IssueGroup, StatusHistory
from grievance_triage.database.session import SessionLocal, init_db
from grievance_triage.ml.embeddings import generate_embeddings
from grievance_triage.services.priority import calculate_priority, sla_deadline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_URGENCY = {"routine", "high", "critical"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_training_complaints(path: Path) -> tuple[int, int]:
    frame = pd.read_csv(path)
    required = {"subject", "body", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {sorted(missing)}")
    dataset_hash = file_hash(path)
    init_db()
    session = SessionLocal()
    inserted = 0
    skipped = 0
    try:
        existing_rows = set(session.scalars(
            select(Complaint.source_row_number).where(
                Complaint.source_dataset_hash == dataset_hash,
                Complaint.source_split == "train",
            )
        ).all())
        pending = []
        for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
            if row_number in existing_rows:
                skipped += 1
                continue
            department, urgency = str(row["label"]).split("|", maxsplit=1)
            urgency = urgency.lower()
            if urgency not in VALID_URGENCY:
                raise ValueError(f"Unsupported urgency at row {row_number}: {urgency}")
            subject = "" if pd.isna(row.get("subject")) else str(row.get("subject"))
            body = "" if pd.isna(row.get("body")) else str(row.get("body"))
            district = "" if pd.isna(row.get("district")) else str(row.get("district"))
            channel = "web" if pd.isna(row.get("channel")) else str(row.get("channel"))
            history = row.get("complaint_history", [])
            if pd.isna(history):
                history = []
            pending.append((row_number, row, department, urgency, subject, body, district, channel, history))

        embeddings = generate_embeddings([f"{item[4]}\n{item[5]}" for item in pending]) if pending else []
        for batch_start in range(0, len(pending), 500):
            batch = list(zip(pending[batch_start:batch_start + 500], embeddings[batch_start:batch_start + 500]))
            issue_groups = []
            complaints = []
            histories = []
            for item, embedding in batch:
                row_number, row, department, urgency, subject, body, district, channel, history = item
                issue_group_id = uuid.uuid4()
                complaint_id = uuid.uuid4()
                issue_group = IssueGroup(
                    id=issue_group_id, department=department, title=subject, description=body,
                    status="RESOLVED", urgency=urgency, priority=calculate_priority(urgency),
                    complaint_count=1, assigned_department=department,
                )
                complaint = Complaint(
                    id=complaint_id, citizen_id=f"training-row-{row_number}", subject=subject, body=body,
                    district=district, channel=channel, complaint_history=history,
                    department=department, department_confidence=1.0,
                    urgency=urgency, urgency_confidence=1.0, status="RESOLVED",
                    priority=calculate_priority(urgency), issue_group_id=issue_group_id, embedding=embedding,
                    sla_deadline=sla_deadline(urgency), source_dataset_hash=dataset_hash,
                    source_split="train", source_row_number=row_number,
                )
                issue_groups.append(issue_group)
                complaints.append(complaint)
                histories.append(StatusHistory(
                    complaint_id=complaint_id, old_status="SUBMITTED", new_status="RESOLVED",
                    changed_by="training-seed", comment="Seeded from labeled training data",
                ))
            session.add_all(issue_groups)
            session.flush()
            session.add_all(complaints)
            session.add_all(histories)
            session.commit()
            inserted += len(batch)
        session.commit()
        return inserted, skipped
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Upload labeled training rows and embeddings to Neon")
    parser.add_argument("--train", default=str(PROJECT_ROOT / "data" / "train.csv"))
    args = parser.parse_args()
    inserted, skipped = seed_training_complaints(Path(args.train))
    print(f"Seed complete: inserted={inserted}, skipped_existing={skipped}")


if __name__ == "__main__":
    main()