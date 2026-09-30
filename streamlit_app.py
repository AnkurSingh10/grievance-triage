import os
import uuid
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Grievance Command Centre",
    page_icon="🏛️",
    layout="wide",
)

# API Configuration
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://3.236.127.3:8000").rstrip("/")

with st.sidebar:
    st.title("⚙️ System Status")
    api_url = st.text_input("Backend API URL", value=DEFAULT_API_URL)
    api_url = api_url.rstrip("/")
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=3)
        if health_resp.status_code == 200:
            st.success("🟢 FastAPI Backend Online")
        else:
            st.warning(f"🟡 Backend Status: {health_resp.status_code}")
    except Exception as ex:
        st.error(f"🔴 Backend Offline / Unreachable: {ex}")

st.title("🏛️ Grievance Command Centre")
st.caption("Citizen intake, department operations, issue groups, and grounded officer assistance (Powered by FastAPI & MuRIL/Gemini)")

KNOWN_DEPARTMENTS = [
    "agriculture_irrigation",
    "education",
    "electricity",
    "healthcare",
    "law_and_order",
    "roads_transport",
    "water_supply",
    "welfare_schemes",
]

ALL_STATUSES = [
    "SUBMITTED",
    "CLASSIFIED",
    "FORWARDED",
    "ASSIGNED",
    "UNDER_INVESTIGATION",
    "ACTION_TAKEN",
    "ESCALATED",
    "REOPENED",
    "RESOLVED",
]

citizen_tab, officer_tab, dashboard_tab, analytics_tab, knowledge_tab = st.tabs([
    "Citizen Portal", "Officer Queue", "Department Dashboard", "Issue Analytics", "Knowledge Base",
])

# ==========================================
# 1. CITIZEN PORTAL
# ==========================================
with citizen_tab:
    st.subheader("Submit a Complaint")
    with st.form("complaint_form"):
        citizen_id = st.text_input("Citizen ID", placeholder="e.g. CITIZEN-98765")
        subject = st.text_input("Subject", placeholder="Brief description of the grievance")
        body = st.text_area("Complaint Details", height=140, placeholder="Provide full details of your grievance...")
        district = st.text_input("District", placeholder="e.g. Lucknow, Varanasi, Bangalore")
        channel = st.selectbox("Channel", ["web", "mobile", "call-centre", "office"])
        submitted = st.form_submit_button("Submit Complaint", type="primary")

    if submitted:
        if not citizen_id or not subject or not body or not district:
            st.error("Citizen ID, subject, complaint, and district are all required.")
        else:
            with st.spinner("Classifying with MuRIL FP16 & saving to database..."):
                try:
                    payload = {
                        "citizen_id": citizen_id,
                        "subject": subject,
                        "body": body,
                        "district": district,
                        "channel": channel,
                        "complaint_history": [],
                    }
                    resp = requests.post(f"{api_url}/complaints", json=payload, timeout=60)
                    if resp.status_code == 201:
                        data = resp.json()
                        st.session_state["last_complaint_id"] = data["id"]
                        st.success(f"Complaint successfully submitted! ID: {data['id']}")
                        
                        summary_df = pd.DataFrame([{
                            "Complaint ID": data["id"],
                            "District": data.get("district"),
                            "Department": data.get("department"),
                            "Department Conf.": round(data.get("department_confidence", 0), 3),
                            "Urgency": data.get("urgency"),
                            "Urgency Conf.": round(data.get("urgency_confidence", 0), 3),
                            "Priority": data.get("priority"),
                            "Status": data.get("status"),
                            "Issue Group": str(data.get("issue_group_id", "")),
                        }])
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as ex:
                    st.error(f"Failed to connect to FastAPI backend: {ex}")

    st.divider()
    st.subheader("Track Complaint")
    tracking_id = st.text_input(
        "Enter Complaint ID (UUID)",
        value=st.session_state.get("last_complaint_id", ""),
        key="citizen_tracking_id",
    )
    if st.button("Track Status", key="load_citizen_status") and tracking_id:
        try:
            uuid.UUID(tracking_id.strip())
        except ValueError:
            st.error("Please enter a valid UUID format (e.g. 5a07ad5a-0111-46c6-8ab5-8a61c0e8357b).")
            st.stop()

        with st.spinner("Fetching complaint status..."):
            try:
                stat_resp = requests.get(f"{api_url}/complaints/{tracking_id.strip()}/status", timeout=15)
                comp_resp = requests.get(f"{api_url}/complaints/{tracking_id.strip()}", timeout=15)

                if stat_resp.status_code == 200 and comp_resp.status_code == 200:
                    stat_data = stat_resp.json()
                    comp_data = comp_resp.json()

                    st.markdown("#### Complaint Details")
                    st.dataframe(pd.DataFrame([{
                        "Complaint ID": comp_data.get("id"),
                        "District": comp_data.get("district"),
                        "Department": comp_data.get("department"),
                        "Department Conf.": round(comp_data.get("department_confidence", 0), 3),
                        "Urgency": comp_data.get("urgency"),
                        "Priority": comp_data.get("priority"),
                        "Current Status": stat_data.get("status"),
                        "Updated At": stat_data.get("updated_at"),
                    }]), use_container_width=True, hide_index=True)

                    st.markdown("#### Status History")
                    history = stat_data.get("history", [])
                    if history:
                        hist_df = pd.DataFrame([{
                            "Previous Status": item.get("old_status"),
                            "New Status": item.get("new_status"),
                            "Changed By": item.get("changed_by"),
                            "Comment": item.get("comment"),
                            "Timestamp": item.get("timestamp"),
                        } for item in history])
                        st.dataframe(hist_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No status transition history recorded yet.")
                elif stat_resp.status_code == 404 or comp_resp.status_code == 404:
                    st.error("Complaint ID not found in database.")
                else:
                    st.error(f"Error ({stat_resp.status_code}): {stat_resp.text}")
            except Exception as ex:
                st.error(f"Failed to fetch tracking data: {ex}")

# ==========================================
# 2. OFFICER QUEUE
# ==========================================
with officer_tab:
    st.subheader("Officer Complaint Queue")
    
    # Fetch active departments dynamically if possible, or fallback
    selected_dept = st.selectbox("Select Department", KNOWN_DEPARTMENTS, index=6)  # defaults to water_supply
    status_filter = st.multiselect("Filter by Status", ALL_STATUSES, default=ALL_STATUSES)

    complaints = []
    try:
        queue_resp = requests.get(f"{api_url}/departments/{selected_dept}/complaints", timeout=30)
        if queue_resp.status_code == 200:
            complaints = queue_resp.json()
            if status_filter:
                complaints = [c for c in complaints if c.get("status") in status_filter]
        else:
            st.error(f"Failed to load queue ({queue_resp.status_code}): {queue_resp.text}")
    except Exception as ex:
        st.error(f"Error fetching department queue: {ex}")

    st.metric("Open Complaints in Queue", len(complaints))
    if complaints:
        queue_df = pd.DataFrame([{
            "ID": item.get("id"),
            "Subject": item.get("subject"),
            "Department": item.get("department"),
            "Urgency": item.get("urgency"),
            "Priority": item.get("priority"),
            "Status": item.get("status"),
            "Created": item.get("created_at"),
            "SLA Deadline": item.get("sla_deadline"),
        } for item in complaints])
        st.dataframe(queue_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Review & Action Complaint")
    selected_id = st.text_input("Enter Complaint ID to Review", key="officer_complaint_id", placeholder="Paste Complaint UUID here...")

    if selected_id:
        try:
            uuid.UUID(selected_id.strip())
            valid_uuid = True
        except ValueError:
            valid_uuid = False
            st.warning("Please enter a valid UUID from the queue.")

        if valid_uuid:
            try:
                comp_resp = requests.get(f"{api_url}/complaints/{selected_id.strip()}", timeout=15)
                rel_resp = requests.get(f"{api_url}/complaints/{selected_id.strip()}/related", timeout=15)

                if comp_resp.status_code == 200:
                    complaint = comp_resp.json()
                    st.write("### Complaint Overview")
                    st.dataframe(pd.DataFrame([{
                        "Complaint ID": complaint.get("id"),
                        "District": complaint.get("district"),
                        "Department": complaint.get("department"),
                        "Urgency": complaint.get("urgency"),
                        "Priority": complaint.get("priority"),
                        "Status": complaint.get("status"),
                        "Issue Group": str(complaint.get("issue_group_id", "")),
                    }]), use_container_width=True, hide_index=True)

                    if rel_resp.status_code == 200:
                        related = rel_resp.json()
                        st.markdown("#### Similar / Related Complaints")
                        if related:
                            st.dataframe(pd.DataFrame([{
                                "Complaint ID": r.get("complaint_id"),
                                "Similarity Score": round(r.get("similarity_score", 0), 3),
                                "Relation Type": r.get("relation_type"),
                            } for r in related]), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No related complaints found.")

                    st.markdown("#### Update Workflow Status & Feedback")
                    urgency_options = ["routine", "low", "medium", "high", "critical"]
                    curr_dept = complaint.get("department", KNOWN_DEPARTMENTS[0])
                    curr_urgency = (complaint.get("urgency") or "routine").lower()
                    curr_status = complaint.get("status", "SUBMITTED")

                    with st.form("officer_action_form"):
                        edited_dept = st.selectbox(
                            "Department",
                            KNOWN_DEPARTMENTS,
                            index=KNOWN_DEPARTMENTS.index(curr_dept) if curr_dept in KNOWN_DEPARTMENTS else 0,
                        )
                        edited_urgency = st.selectbox(
                            "Urgency",
                            urgency_options,
                            index=urgency_options.index(curr_urgency) if curr_urgency in urgency_options else 0,
                        )
                        edited_status = st.selectbox(
                            "Status",
                            ALL_STATUSES,
                            index=ALL_STATUSES.index(curr_status) if curr_status in ALL_STATUSES else 0,
                        )
                        officer_id = st.text_input("Officer ID", value="officer-01")
                        officer_comment = st.text_input("Officer Comment / Resolution Note")
                        save_btn = st.form_submit_button("Save Updates", type="primary")

                    if save_btn:
                        # 1. Update Status
                        status_payload = {
                            "status": edited_status,
                            "changed_by": officer_id,
                            "comment": officer_comment or "Status updated by officer",
                        }
                        patch_resp = requests.patch(f"{api_url}/complaints/{selected_id.strip()}/status", json=status_payload, timeout=15)
                        
                        # 2. Record feedback if department or urgency changed
                        if edited_dept != curr_dept or edited_urgency != curr_urgency:
                            fb_payload = {
                                "complaint_id": selected_id.strip(),
                                "corrected_department": edited_dept,
                                "corrected_urgency": edited_urgency,
                                "officer_id": officer_id,
                            }
                            requests.post(f"{api_url}/officer/feedback", json=fb_payload, timeout=15)

                        if patch_resp.status_code == 200:
                            st.success("Complaint status and officer feedback successfully recorded!")
                            st.rerun()
                        else:
                            st.error(f"Failed to update status ({patch_resp.status_code}): {patch_resp.text}")

                    # Grounded Gemini RAG Assistance
                    st.markdown("#### 🤖 Grounded AI Assistance (Gemini RAG)")
                    if st.button("Generate Officer Assistance & Policy Guidance", key="rag_assist_btn"):
                        with st.spinner("Retrieving departmental policies and historical resolved cases via Gemini..."):
                            try:
                                assist_resp = requests.post(f"{api_url}/complaints/{selected_id.strip()}/assist", timeout=60)
                                if assist_resp.status_code == 200:
                                    st.session_state["assistance"] = assist_resp.json()
                                else:
                                    st.error(f"Assistance Error ({assist_resp.status_code}): {assist_resp.text}")
                            except Exception as ex:
                                st.error(f"Failed to call assistance endpoint: {ex}")

                    if st.session_state.get("assistance"):
                        assistance = st.session_state["assistance"]
                        st.subheader("💡 Suggested Next Steps & Strategy")
                        st.markdown(assistance.get("summary", "No summary generated."))
                        
                        next_steps = assistance.get("suggested_next_steps", [])
                        if next_steps:
                            st.markdown("**Actionable Steps:**")
                            for s in next_steps:
                                st.markdown(f"- {s}")

                        policies = assistance.get("relevant_policies", [])
                        if policies:
                            with st.expander("📚 Relevant Government Policies & Guidelines"):
                                st.dataframe(pd.DataFrame(policies), use_container_width=True, hide_index=True)

                        cases = assistance.get("similar_resolved_cases", [])
                        if cases:
                            with st.expander("📂 Similar Precedent Cases"):
                                st.dataframe(pd.DataFrame(cases), use_container_width=True, hide_index=True)

                        citations = assistance.get("citations", [])
                        if citations:
                            st.caption("Citations: " + ", ".join(str(c) for c in citations))

                elif comp_resp.status_code == 404:
                    st.error("Complaint ID not found.")
                else:
                    st.error(f"Error ({comp_resp.status_code}): {comp_resp.text}")
            except Exception as ex:
                st.error(f"Error querying complaint: {ex}")

# ==========================================
# 3. DEPARTMENT DASHBOARD
# ==========================================
with dashboard_tab:
    st.subheader("Department Analytics Dashboard")
    dash_dept = st.selectbox("Select Department for Dashboard", KNOWN_DEPARTMENTS, index=6, key="dash_dept")

    try:
        d_resp = requests.get(f"{api_url}/departments/{dash_dept}/dashboard", timeout=15)
        if d_resp.status_code == 200:
            stats = d_resp.json()
            
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("New Complaints", stats.get("new_complaints", 0))
            c2.metric("Critical", stats.get("critical", 0))
            c3.metric("High", stats.get("high", 0))
            c4.metric("Medium", stats.get("medium", 0))
            c5.metric("Low", stats.get("low", 0))
            c6.metric("Unresolved", stats.get("unresolved", 0))

            urgency_chart_data = pd.DataFrame({
                "Urgency": ["Critical", "High", "Medium", "Low", "Routine"],
                "Count": [
                    stats.get("critical", 0),
                    stats.get("high", 0),
                    stats.get("medium", 0),
                    stats.get("low", 0),
                    stats.get("routine", 0),
                ]
            }).set_index("Urgency")
            
            st.markdown("#### Urgency Distribution")
            st.bar_chart(urgency_chart_data)
        else:
            st.error(f"Failed to load dashboard metrics ({d_resp.status_code}): {d_resp.text}")
    except Exception as ex:
        st.error(f"Error fetching dashboard data: {ex}")

# ==========================================
# 4. ISSUE ANALYTICS
# ==========================================
with analytics_tab:
    st.subheader("Recurring Issue Clusters & Analytics")
    try:
        issues_resp = requests.get(f"{api_url}/analytics/issues", timeout=20)
        if issues_resp.status_code == 200:
            issue_list = issues_resp.json()
            if issue_list:
                idf = pd.DataFrame(issue_list)
                st.dataframe(
                    idf.rename(columns={
                        "department": "Department",
                        "issue_group_id": "Issue Group ID",
                        "complaint_count": "Total Complaints in Cluster",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No issue clusters aggregated yet.")
        else:
            st.error(f"Failed to fetch issue clusters ({issues_resp.status_code}): {issues_resp.text}")
    except Exception as ex:
        st.error(f"Error fetching issue analytics: {ex}")

# ==========================================
# 5. KNOWLEDGE BASE
# ==========================================
with knowledge_tab:
    st.subheader("Department Knowledge Base & Policy Repository")
    st.markdown("Upload official government circulars, SOPs, and service policies to be vectorized for Gemini RAG grounding.")
    
    with st.form("knowledge_upload_form"):
        knowledge_department = st.selectbox("Department", KNOWN_DEPARTMENTS, key="kb_dept")
        document_type = st.selectbox("Document Type", ["policy", "procedure", "service-guideline", "FAQ", "circular"])
        source = st.text_input("Source / Reference (e.g. GR-2026/PWD/09)")
        content = st.text_area("Document Content / Policy Text", height=180, placeholder="Paste policy details or standard resolution SOPs...")
        add_doc = st.form_submit_button("Store & Vectorize Document", type="primary")

    if add_doc:
        if not knowledge_department or not source or not content:
            st.error("Department, source reference, and document content are required.")
        else:
            with st.spinner("Embedding document via SentenceTransformers & storing in Neon pgvector..."):
                try:
                    payload = {
                        "department": knowledge_department,
                        "document_type": document_type,
                        "source": source,
                        "content": content,
                    }
                    resp = requests.post(f"{api_url}/knowledge-documents", json=payload, timeout=60)
                    if resp.status_code == 200:
                        doc_id = resp.json().get("id")
                        st.success(f"✅ Successfully stored and embedded knowledge document! Document ID: {doc_id}")
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as ex:
                    st.error(f"Failed to connect to API: {ex}")
