import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base
from ..core.settings import get_settings


settings = get_settings()


class IssueGroup(Base):
    __tablename__ = "issue_groups"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    department: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default="OPEN", index=True)
    urgency: Mapped[str] = mapped_column(String(40), default="Low")
    priority: Mapped[str] = mapped_column(String(10), default="P3")
    complaint_count: Mapped[int] = mapped_column(Integer, default=0)
    first_reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    assigned_department: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Complaint(Base):
    __tablename__ = "complaints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    citizen_id: Mapped[str] = mapped_column(String(150), index=True)
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text)
    district: Mapped[str] = mapped_column(String(150), index=True)
    channel: Mapped[str] = mapped_column(String(80), default="web")
    complaint_history: Mapped[list] = mapped_column(JSONB, default=list)
    department: Mapped[str] = mapped_column(String(100), index=True)
    department_confidence: Mapped[float] = mapped_column(Float)
    urgency: Mapped[str] = mapped_column(String(40), index=True)
    urgency_confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(40), default="SUBMITTED", index=True)
    priority: Mapped[str] = mapped_column(String(10), default="P3", index=True)
    issue_group_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("issue_groups.id"), index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dimension))
    sla_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_dataset_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    source_split: Mapped[str | None] = mapped_column(String(20), index=True)
    source_row_number: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_complaints_embedding_hnsw", "embedding", postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("ix_complaints_training_source", "source_dataset_hash", "source_split", "source_row_number", unique=True),
    )


class TrainingRecord(Base):
    __tablename__ = "training_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_hash: Mapped[str] = mapped_column(String(64), index=True)
    source_split: Mapped[str] = mapped_column(String(20), index=True)
    source_file: Mapped[str] = mapped_column(String(255))
    row_number: Mapped[int] = mapped_column(Integer)
    raw_data: Mapped[dict] = mapped_column(JSONB)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_training_records_dataset_row", "dataset_hash", "source_split", "row_number", unique=True),
    )


class ComplaintRelation(Base):
    __tablename__ = "complaint_relations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_complaint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("complaints.id"), index=True)
    target_complaint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("complaints.id"), index=True)
    similarity_score: Mapped[float] = mapped_column(Float)
    relation_type: Mapped[str] = mapped_column(String(30), default="RELATED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StatusHistory(Base):
    __tablename__ = "status_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    complaint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("complaints.id"), index=True)
    old_status: Mapped[str | None] = mapped_column(String(40))
    new_status: Mapped[str] = mapped_column(String(40))
    changed_by: Mapped[str] = mapped_column(String(150))
    comment: Mapped[str | None] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    department: Mapped[str] = mapped_column(String(100), index=True)
    document_type: Mapped[str] = mapped_column(String(80))
    source: Mapped[str] = mapped_column(String(500))
    effective_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dimension))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_knowledge_embedding_hnsw", "embedding", postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}),
    )


class OfficerFeedback(Base):
    __tablename__ = "officer_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    complaint_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("complaints.id"), index=True)
    original_department: Mapped[str] = mapped_column(String(100))
    corrected_department: Mapped[str | None] = mapped_column(String(100))
    original_urgency: Mapped[str] = mapped_column(String(40))
    corrected_urgency: Mapped[str | None] = mapped_column(String(40))
    officer_id: Mapped[str] = mapped_column(String(150))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
