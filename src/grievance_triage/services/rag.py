import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database.models import Complaint, KnowledgeDocument
from ..core.settings import get_settings


def officer_assistance(session: Session, complaint: Complaint) -> dict:
    from ..ml.embeddings import generate_embedding

    settings = get_settings()
    query_embedding = generate_embedding(f"{complaint.subject}\n{complaint.body}")
    distance = KnowledgeDocument.embedding.cosine_distance(query_embedding)
    policies = session.execute(
        select(KnowledgeDocument, (1 - distance).label("similarity"))
        .where(KnowledgeDocument.department == complaint.department)
        .order_by(distance).limit(5)
    ).all()
    resolved = session.execute(
        select(Complaint).where(Complaint.department == complaint.department, Complaint.status == "RESOLVED").limit(5)
    ).scalars().all()
    policy_context = "\n\n".join(f"[{doc.source}] {doc.content}" for doc, _ in policies)
    case_context = "\n\n".join(f"[{case.id}] {case.subject}: {case.body}" for case in resolved)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You assist a government officer. Use only the supplied sources. Do not invent policy. Return valid JSON with summary, suggested_next_steps, and citations."),
        ("human", "Complaint:\n{complaint}\n\nPolicies:\n{policies}\n\nResolved cases:\n{cases}"),
    ])
    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.google_api_key, temperature=settings.gemini_temperature)
    response = llm.invoke(prompt.format_messages(complaint=f"{complaint.subject}\n{complaint.body}", policies=policy_context or "No policy documents found.", cases=case_context or "No resolved cases found."))
    try:
        result = json.loads(response.content)
    except (TypeError, json.JSONDecodeError):
        result = {"summary": str(response.content), "suggested_next_steps": [], "citations": []}

    nested_summary = result.get("summary")
    if isinstance(nested_summary, str):
        fenced_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", nested_summary, re.DOTALL)
        candidate = fenced_json.group(1) if fenced_json else nested_summary.strip()
        try:
            nested_result = json.loads(candidate)
        except json.JSONDecodeError:
            nested_result = None
        if isinstance(nested_result, dict):
            result = {**result, **nested_result}

    if isinstance(result.get("summary"), str):
        result["summary"] = result["summary"].replace("```json", "").replace("```", "").strip()
    result.setdefault("relevant_policies", [{"source": doc.source, "similarity": float(score)} for doc, score in policies])
    result.setdefault("similar_resolved_cases", [{"id": str(case.id), "subject": case.subject} for case in resolved])
    result.setdefault("suggested_next_steps", [])
    result.setdefault("citations", [doc.source for doc, _ in policies])
    return result
