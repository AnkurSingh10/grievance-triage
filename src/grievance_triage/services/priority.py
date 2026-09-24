from datetime import datetime, timedelta, timezone

URGENCY_PRIORITY = {"critical": "P0", "high": "P1", "medium": "P2", "low": "P3", "routine": "P3"}
SLA_HOURS = {"critical": 12, "high": 48, "medium": 96, "low": 168, "routine": 168}


def calculate_priority(urgency: str, previous_complaints: int = 0, unresolved_complaints: int = 0, complaint_age_hours: float = 0) -> str:
    priority = URGENCY_PRIORITY.get(urgency.lower(), "P3")
    if priority not in {"P0", "P1"} and (previous_complaints >= 10 or unresolved_complaints >= 5 or complaint_age_hours >= 72):
        return "P1" if urgency.lower() == "high" else "P2"
    return priority


def sla_deadline(urgency: str) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=SLA_HOURS.get(urgency.lower(), 168))
