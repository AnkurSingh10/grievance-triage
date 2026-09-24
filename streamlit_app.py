import pandas as pd
import streamlit as st
from sqlalchemy import func, select
from uuid import UUID

from grievance_triage.api.schemas import ComplaintCreate, KnowledgeDocumentCreate
from grievance_triage.database.models import Complaint, ComplaintRelation, IssueGroup, KnowledgeDocument, StatusHistory
from grievance_triage.database.session import SessionLocal, init_db
from grievance_triage.ml.embeddings import generate_embedding
from grievance_triage.services.complaint import create_complaint, update_complaint_assignment, update_status
from grievance_triage.services.rag import officer_assistance

init_db()

st.set_page_config(page_title="Grievance Command Centre", page_icon="G", layout="wide")
st.title("Grievance Command Centre")
st.caption("Citizen intake, department operations, issue groups, and grounded officer assistance")


def complaint_summary(complaint):
    return {
        "Complaint ID": str(complaint.id),
        "District": complaint.district,
        "Department": complaint.department,
        "Department confidence": round(complaint.department_confidence, 3),
        "Urgency": complaint.urgency,
        "Urgency confidence": round(complaint.urgency_confidence, 3),
        "Priority": complaint.priority,
        "Status": complaint.status,
        "Issue group": str(complaint.issue_group_id),
        "SLA deadline": complaint.sla_deadline,
    }


def load_complaint(session, complaint_id):
    try:
        return session.get(Complaint, UUID(str(complaint_id).strip()))
    except (ValueError, TypeError):
        return None


citizen_tab, officer_tab, dashboard_tab, analytics_tab, knowledge_tab = st.tabs([
    "Citizen Portal", "Officer Queue", "Department Dashboard", "Issue Analytics", "Knowledge Base",
])

with citizen_tab:
    st.subheader("Submit a complaint")
    with st.form("complaint"):
        citizen_id = st.text_input("Citizen ID")
        subject = st.text_input("Subject")
        body = st.text_area("Complaint", height=140)
        district = st.text_input("District")
        channel = st.selectbox("Channel", ["web", "mobile", "call-centre", "office"])
        submitted = st.form_submit_button("Submit complaint", type="primary")
    if submitted:
        if not citizen_id or not subject or not body or not district:
            st.error("Citizen ID, subject, complaint, and district are required.")
        else:
            session = SessionLocal()
            try:
                complaint = create_complaint(session, ComplaintCreate(
                    citizen_id=citizen_id, subject=subject, body=body,
                    district=district, channel=channel,
                ))
                st.session_state["last_complaint_id"] = str(complaint.id)
                st.success(f"Complaint submitted: {complaint.id}")
                st.dataframe(pd.DataFrame([complaint_summary(complaint)]), use_container_width=True, hide_index=True)
            finally:
                session.close()

    st.divider()
    st.subheader("Track complaint")
    tracking_id = st.text_input("Complaint ID", value=st.session_state.get("last_complaint_id", ""), key="citizen_tracking_id")
    if st.button("Load status", key="load_citizen_status") and tracking_id:
        session = SessionLocal()
        try:
            try:
                UUID(tracking_id.strip())
            except ValueError:
                st.error("Enter the full Complaint ID (UUID) shown after submission, not a row number such as 45.")
                st.stop()
            complaint = load_complaint(session, tracking_id)
            if not complaint:
                st.error("Complaint not found.")
            else:
                st.dataframe(pd.DataFrame([complaint_summary(complaint)]), use_container_width=True, hide_index=True)
                history = session.execute(select(StatusHistory).where(StatusHistory.complaint_id == complaint.id).order_by(StatusHistory.timestamp)).scalars().all()
                st.dataframe(pd.DataFrame([{
                    "Previous": item.old_status, "Current": item.new_status,
                    "Changed by": item.changed_by, "Comment": item.comment, "Time": item.timestamp,
                } for item in history]), use_container_width=True, hide_index=True)
        finally:
            session.close()

with officer_tab:
    st.subheader("Officer complaint queue")
    session = SessionLocal()
    try:
        departments = session.execute(select(Complaint.department).distinct().order_by(Complaint.department)).scalars().all()
        selected_department = st.selectbox("Department", departments or ["No departments yet"])
        all_statuses = ["SUBMITTED", "CLASSIFIED", "FORWARDED", "ASSIGNED", "UNDER_INVESTIGATION", "ACTION_TAKEN", "ESCALATED", "REOPENED", "RESOLVED"]
        status_filter = st.multiselect("Status", all_statuses, default=all_statuses)
        query = select(Complaint).where(Complaint.department == selected_department).order_by(Complaint.created_at.desc())
        complaints = session.execute(query).scalars().all() if departments else []
        complaints = [item for item in complaints if not status_filter or item.status in status_filter]
        st.metric("Open queue", len(complaints))
        st.dataframe(pd.DataFrame([{
            "ID": str(item.id), "Subject": item.subject, "District": item.district,
            "Urgency": item.urgency, "Priority": item.priority, "Status": item.status,
            "Created": item.created_at, "SLA": item.sla_deadline,
        } for item in complaints]), use_container_width=True, hide_index=True)
        selected_id = st.text_input("Complaint ID to review", key="officer_complaint_id")
        complaint = load_complaint(session, selected_id) if selected_id else None
        if selected_id and not complaint:
            st.warning("Enter a valid full Complaint ID (UUID) from the queue.")
        if complaint:
            st.write(complaint_summary(complaint))
            relation_rows = session.execute(select(ComplaintRelation).where(ComplaintRelation.source_complaint_id == complaint.id)).scalars().all()
            st.write("Related complaints")
            st.dataframe(pd.DataFrame([{
                "Complaint": str(row.target_complaint_id), "Similarity": round(row.similarity_score, 3), "Relation": row.relation_type,
            } for row in relation_rows]), use_container_width=True, hide_index=True)
            st.subheader("Edit assignment and workflow")
            department_options = sorted(set(departments + [complaint.department]))
            urgency_options = ["routine", "low", "medium", "high", "critical"]
            priority_options = ["P0", "P1", "P2", "P3"]
            with st.form("complaint_edit"):
                edited_department = st.selectbox("Assigned department", department_options, index=department_options.index(complaint.department))
                edited_urgency = st.selectbox("Urgency", urgency_options, index=urgency_options.index(complaint.urgency.lower()) if complaint.urgency.lower() in urgency_options else 0)
                edited_priority = st.selectbox("Priority", priority_options, index=priority_options.index(complaint.priority) if complaint.priority in priority_options else 3)
                edited_status = st.selectbox("Status", all_statuses, index=all_statuses.index(complaint.status))
                comment = st.text_input("Officer comment")
                save_edit = st.form_submit_button("Save complaint changes", type="primary")
            if save_edit:
                try:
                    update_complaint_assignment(session, complaint.id, edited_department, edited_urgency, edited_priority, "streamlit-officer", comment)
                    if edited_status != complaint.status:
                        update_status(session, complaint.id, edited_status, "streamlit-officer", comment)
                    st.success("Department, urgency, priority, and status updated.")
                    st.rerun()
                except (LookupError, ValueError) as error:
                    st.error(str(error))
            if st.button("Generate Gemini assistance", key="assist_button"):
                with st.spinner("Retrieving policies and resolved cases..."):
                    st.session_state["assistance"] = officer_assistance(session, complaint)
            if st.session_state.get("assistance"):
                st.subheader("Grounded officer assistance")
                assistance = st.session_state["assistance"]
                st.markdown(assistance.get("summary", "No summary available."))
                next_steps = assistance.get("suggested_next_steps", [])
                if next_steps:
                    st.markdown("**Suggested next steps**")
                    for step in next_steps:
                        st.markdown(f"- {step}")
                policies = assistance.get("relevant_policies", [])
                if policies:
                    with st.expander("Relevant policies"):
                        st.dataframe(pd.DataFrame(policies), use_container_width=True, hide_index=True)
                cases = assistance.get("similar_resolved_cases", [])
                if cases:
                    with st.expander("Similar resolved cases"):
                        st.dataframe(pd.DataFrame(cases), use_container_width=True, hide_index=True)
                citations = assistance.get("citations", [])
                if citations:
                    st.caption("Citations: " + ", ".join(str(citation) for citation in citations))
    finally:
        session.close()

with dashboard_tab:
    st.subheader("Department dashboard")
    session = SessionLocal()
    try:
        departments = session.execute(select(Complaint.department).distinct().order_by(Complaint.department)).scalars().all()
        selected_department = st.selectbox("Department", departments or ["No data"], key="dashboard_department")
        rows = session.execute(select(Complaint.status, Complaint.urgency, func.count(Complaint.id)).where(Complaint.department == selected_department).group_by(Complaint.status, Complaint.urgency)).all() if departments else []
        unresolved = {"SUBMITTED", "CLASSIFIED", "FORWARDED", "ASSIGNED", "UNDER_INVESTIGATION", "ACTION_TAKEN", "ESCALATED", "REOPENED"}
        counts = {"New": 0, "Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Unresolved": 0}
        for status, urgency, count in rows:
            counts[urgency.title()] = counts.get(urgency.title(), 0) + count
            counts["Unresolved"] += count if status in unresolved else 0
            counts["New"] += count if status == "SUBMITTED" else 0
        columns = st.columns(6)
        for column, (label, value) in zip(columns, counts.items()):
            column.metric(label, value)
        st.bar_chart(pd.DataFrame({"Urgency": [counts.get("Critical", 0), counts.get("High", 0), counts.get("Medium", 0), counts.get("Low", 0)]}, index=["Critical", "High", "Medium", "Low"]))
    finally:
        session.close()

with analytics_tab:
    st.subheader("Issue groups and recurring complaints")
    session = SessionLocal()
    try:
        groups = session.execute(select(IssueGroup).order_by(IssueGroup.complaint_count.desc())).scalars().all()
        st.dataframe(pd.DataFrame([{
            "Issue group": str(group.id), "Department": group.department, "Title": group.title,
            "Complaints": group.complaint_count, "Urgency": group.urgency, "Priority": group.priority, "Status": group.status,
        } for group in groups]), use_container_width=True, hide_index=True)
    finally:
        session.close()

with knowledge_tab:
    st.subheader("Department knowledge base")
    with st.form("knowledge_document"):
        knowledge_department = st.text_input("Department", key="knowledge_department")
        document_type = st.selectbox("Document type", ["policy", "procedure", "service-guideline", "FAQ", "circular"])
        source = st.text_input("Source / document reference")
        content = st.text_area("Document content", height=180)
        add_document = st.form_submit_button("Store and embed document")
    if add_document:
        if not knowledge_department or not source or not content:
            st.error("Department, source, and content are required.")
        else:
            session = SessionLocal()
            try:
                document = KnowledgeDocument(**KnowledgeDocumentCreate(
                    department=knowledge_department, document_type=document_type,
                    source=source, content=content,
                ).model_dump(), embedding=generate_embedding(content))
                session.add(document)
                session.commit()
                st.success(f"Stored knowledge document {document.id} in Neon.")
            finally:
                session.close()
    session = SessionLocal()
    try:
        documents = session.execute(select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())).scalars().all()
        st.dataframe(pd.DataFrame([{
            "Department": item.department, "Type": item.document_type, "Source": item.source, "Created": item.created_at,
        } for item in documents]), use_container_width=True, hide_index=True)
    finally:
        session.close()
