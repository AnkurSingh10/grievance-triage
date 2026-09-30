from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..api.schemas import ComplaintCreate
from ..database.models import Complaint, ComplaintRelation, IssueGroup, StatusHistory
from ..core.settings import get_settings
from .priority import calculate_priority, sla_deadline

UNRESOLVED = {"SUBMITTED", "CLASSIFIED", "FORWARDED", "ASSIGNED", "UNDER_INVESTIGATION", "ACTION_TAKEN", "ESCALATED", "REOPENED"}
VALID_STATUSES = UNRESOLVED | {"RESOLVED"}


def find_similar(session: Session, department: str, embedding: list[float], top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    distance = Complaint.embedding.cosine_distance(embedding)
    rows = session.execute(
        select(Complaint, (1 - distance).label("similarity"))
        .where(Complaint.department == department)
        .order_by(distance)
        .limit(top_k or settings.similarity_top_k)
    ).all()
    return [{"complaint": complaint, "similarity": float(similarity)} for complaint, similarity in rows]


def create_complaint(session: Session, payload: ComplaintCreate) -> Complaint:
    from ..ml.embeddings import generate_embedding
    from ..ml.inference import predict_complaint

    prediction = predict_complaint(payload.subject, payload.body, payload.channel, payload.complaint_history)
    embedding = generate_embedding(f"{payload.subject}\n{payload.body}")
    candidates = find_similar(session, prediction["department"], embedding)
    threshold = get_settings().complaint_similarity_threshold
    related = [item for item in candidates if item["similarity"] >= threshold]
    unresolved = next((item for item in related if item["complaint"].status in UNRESOLVED), None)
    reference = unresolved or (related[0] if related else None)
    if reference and unresolved:
        issue_group = session.get(IssueGroup, reference["complaint"].issue_group_id)
        issue_group.complaint_count += 1
        issue_group.last_reported_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    else:
        issue_group = IssueGroup(
            department=prediction["department"],
            title=payload.subject,
            description=payload.body,
            urgency=prediction["urgency"],
            priority=calculate_priority(prediction["urgency"], len(related)),
            complaint_count=1,
            assigned_department=prediction["department"],
        )
        session.add(issue_group)
        session.flush()
    complaint = Complaint(
        citizen_id=payload.citizen_id, subject=payload.subject, body=payload.body,
        district=payload.district,
        channel=payload.channel, complaint_history=payload.complaint_history,
        department=prediction["department"], department_confidence=prediction["department_confidence"],
        urgency=prediction["urgency"], urgency_confidence=prediction["urgency_confidence"],
        status="CLASSIFIED", priority=issue_group.priority, issue_group_id=issue_group.id,
        embedding=embedding, sla_deadline=sla_deadline(prediction["urgency"]),
    )
    session.add(complaint)
    session.flush()
    session.add(StatusHistory(complaint_id=complaint.id, old_status="SUBMITTED", new_status="CLASSIFIED", changed_by="system", comment="BiLSTM classification completed"))
    for item in related:
        session.add(ComplaintRelation(source_complaint_id=complaint.id, target_complaint_id=item["complaint"].id, similarity_score=item["similarity"], relation_type="DUPLICATE" if item is reference else "RELATED"))
    session.commit()
    session.refresh(complaint)
    return complaint


def update_status(session: Session, complaint_id: UUID, status: str, changed_by: str, comment: str | None) -> Complaint:
    if status not in VALID_STATUSES:
        raise ValueError(f"Unsupported complaint status: {status}")
    complaint = session.get(Complaint, complaint_id)
    if not complaint:
        raise LookupError("Complaint not found")
    old_status = complaint.status
    complaint.status = status
    session.add(StatusHistory(complaint_id=complaint.id, old_status=old_status, new_status=status, changed_by=changed_by, comment=comment))
    if status == "RESOLVED":
        complaint.resolved_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    session.commit()
    session.refresh(complaint)
    return complaint


def update_complaint_assignment(
    session: Session,
    complaint_id: UUID,
    department: str,
    urgency: str,
    priority: str,
    changed_by: str,
    comment: str | None,
) -> Complaint:
    complaint = session.get(Complaint, complaint_id)
    if not complaint:
        raise LookupError("Complaint not found")
    if not department.strip():
        raise ValueError("Department is required")
    if urgency.lower() not in {"routine", "low", "medium", "high", "critical"}:
        raise ValueError("Unsupported urgency")
    if priority not in {"P0", "P1", "P2", "P3"}:
        raise ValueError("Unsupported priority")

    old_status = complaint.status
    old_department = complaint.department
    old_urgency = complaint.urgency
    old_priority = complaint.priority
    complaint.department = department.strip()
    complaint.urgency = urgency.lower()
    complaint.priority = priority
    if old_status == "CLASSIFIED":
        complaint.status = "FORWARDED"
    session.add(StatusHistory(
        complaint_id=complaint.id,
        old_status=old_status,
        new_status=complaint.status,
        changed_by=changed_by,
        comment=(
            f"Assignment edited: department {old_department}->{complaint.department}, "
            f"urgency {old_urgency}->{complaint.urgency}, priority {old_priority}->{priority}. "
            f"{comment or ''}"
        ).strip(),
    ))
    session.commit()
    session.refresh(complaint)
    return complaint
