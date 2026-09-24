from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ComplaintCreate(BaseModel):
    citizen_id: str
    subject: str = Field(min_length=3, max_length=500)
    body: str = Field(min_length=3)
    district: str = Field(min_length=1, max_length=150)
    channel: str = "web"
    complaint_history: list = Field(default_factory=list)


class ComplaintResponse(BaseModel):
    id: UUID
    district: str
    department: str
    department_confidence: float
    urgency: str
    urgency_confidence: float
    status: str
    priority: str
    issue_group_id: UUID
    related_complaints: list[dict] = []


class StatusUpdate(BaseModel):
    status: str
    changed_by: str
    comment: str | None = None


class OfficerFeedbackCreate(BaseModel):
    complaint_id: UUID
    corrected_department: str | None = None
    corrected_urgency: str | None = None
    officer_id: str


class KnowledgeDocumentCreate(BaseModel):
    department: str
    document_type: str
    source: str
    content: str
    effective_date: datetime | None = None


class AssistanceResponse(BaseModel):
    summary: str
    relevant_policies: list[dict]
    similar_resolved_cases: list[dict]
    suggested_next_steps: list[str]
    citations: list[str]
