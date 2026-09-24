from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .schemas import (AssistanceResponse, ComplaintCreate, ComplaintResponse,
                      KnowledgeDocumentCreate, OfficerFeedbackCreate, StatusUpdate)
from ..services.complaint import create_complaint, update_status
from ..database.session import get_db, init_db
from ..database.models import Complaint, ComplaintRelation, KnowledgeDocument, OfficerFeedback, StatusHistory
from ..ml.embeddings import generate_embedding
from ..services.rag import officer_assistance

app = FastAPI(title="Government Grievance Management API", version="0.1.0")


@app.on_event("startup")
def startup():
    init_db()


def response_for(session: Session, complaint: Complaint) -> ComplaintResponse:
    relations = session.execute(select(ComplaintRelation).where(ComplaintRelation.source_complaint_id == complaint.id)).scalars().all()
    return ComplaintResponse(
        id=complaint.id, district=complaint.district, department=complaint.department, department_confidence=complaint.department_confidence,
        urgency=complaint.urgency, urgency_confidence=complaint.urgency_confidence, status=complaint.status,
        priority=complaint.priority, issue_group_id=complaint.issue_group_id,
        related_complaints=[{"complaint_id": str(item.target_complaint_id), "similarity_score": item.similarity_score, "relation_type": item.relation_type} for item in relations],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/complaints", response_model=ComplaintResponse, status_code=201)
def submit_complaint(payload: ComplaintCreate, session: Session = Depends(get_db)):
    return response_for(session, create_complaint(session, payload))


@app.get("/complaints/{complaint_id}", response_model=ComplaintResponse)
def get_complaint(complaint_id: UUID, session: Session = Depends(get_db)):
    complaint = session.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    return response_for(session, complaint)


@app.get("/complaints/{complaint_id}/related")
def related_complaints(complaint_id: UUID, session: Session = Depends(get_db)):
    complaint = session.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    rows = session.execute(select(ComplaintRelation).where(ComplaintRelation.source_complaint_id == complaint_id)).scalars().all()
    return [{"complaint_id": str(row.target_complaint_id), "similarity_score": row.similarity_score, "relation_type": row.relation_type} for row in rows]


@app.get("/complaints/{complaint_id}/status")
def complaint_status(complaint_id: UUID, session: Session = Depends(get_db)):
    complaint = session.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    history = session.execute(
        select(StatusHistory).where(StatusHistory.complaint_id == complaint_id).order_by(StatusHistory.timestamp)
    ).scalars().all()
    return {
        "complaint_id": str(complaint.id),
        "status": complaint.status,
        "department": complaint.department,
        "updated_at": complaint.updated_at,
        "history": [
            {"old_status": item.old_status, "new_status": item.new_status, "changed_by": item.changed_by, "comment": item.comment, "timestamp": item.timestamp}
            for item in history
        ],
    }


@app.patch("/complaints/{complaint_id}/status", response_model=ComplaintResponse)
def change_status(complaint_id: UUID, payload: StatusUpdate, session: Session = Depends(get_db)):
    try:
        return response_for(session, update_status(session, complaint_id, payload.status, payload.changed_by, payload.comment))
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/complaints/{complaint_id}/assist", response_model=AssistanceResponse)
def assist_officer(complaint_id: UUID, session: Session = Depends(get_db)):
    complaint = session.get(Complaint, complaint_id)
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    return officer_assistance(session, complaint)


@app.post("/officer/feedback")
def officer_feedback(payload: OfficerFeedbackCreate, session: Session = Depends(get_db)):
    complaint = session.get(Complaint, payload.complaint_id)
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    feedback = OfficerFeedback(
        complaint_id=complaint.id, original_department=complaint.department,
        corrected_department=payload.corrected_department, original_urgency=complaint.urgency,
        corrected_urgency=payload.corrected_urgency, officer_id=payload.officer_id,
    )
    session.add(feedback)
    session.commit()
    return {"feedback_id": str(feedback.id), "status": "recorded"}


@app.post("/knowledge-documents")
def add_knowledge_document(payload: KnowledgeDocumentCreate, session: Session = Depends(get_db)):
    document = KnowledgeDocument(**payload.model_dump(), embedding=generate_embedding(payload.content))
    session.add(document)
    session.commit()
    return {"id": str(document.id), "status": "stored"}


@app.get("/departments/{department}/complaints")
def department_complaints(department: str, session: Session = Depends(get_db)):
    complaints = session.execute(select(Complaint).where(Complaint.department == department).order_by(Complaint.created_at.desc())).scalars().all()
    return [
        {
            "id": str(item.id), "subject": item.subject, "department": item.department,
            "urgency": item.urgency, "priority": item.priority, "status": item.status,
            "issue_group_id": str(item.issue_group_id), "created_at": item.created_at,
            "sla_deadline": item.sla_deadline,
        }
        for item in complaints
    ]


@app.get("/departments/{department}/dashboard")
def department_dashboard(department: str, session: Session = Depends(get_db)):
    rows = session.execute(
        select(Complaint.status, Complaint.urgency, func.count(Complaint.id))
        .where(Complaint.department == department)
        .group_by(Complaint.status, Complaint.urgency)
    ).all()
    unresolved = {"SUBMITTED", "CLASSIFIED", "FORWARDED", "ASSIGNED", "UNDER_INVESTIGATION", "ACTION_TAKEN", "ESCALATED", "REOPENED"}
    summary = {"new_complaints": 0, "critical": 0, "high": 0, "medium": 0, "low": 0, "unresolved": 0}
    for status, urgency, count in rows:
        summary[urgency.lower()] = summary.get(urgency.lower(), 0) + count
        if status in unresolved:
            summary["unresolved"] += count
        if status == "SUBMITTED":
            summary["new_complaints"] += count
    return {"department": department, **summary}


@app.get("/analytics/issues")
def issue_analytics(session: Session = Depends(get_db)):
    rows = session.execute(
        select(Complaint.department, Complaint.issue_group_id, func.count(Complaint.id).label("complaint_count"))
        .group_by(Complaint.department, Complaint.issue_group_id)
        .order_by(func.count(Complaint.id).desc())
    ).all()
    return [{"department": department, "issue_group_id": str(issue_group_id), "complaint_count": count} for department, issue_group_id, count in rows]
