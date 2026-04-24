from pydantic import BaseModel, Field
from typing import List


# Output Models 

class TicketCategory(BaseModel):
    department: str  # Billing / Technical / Account / Other
    urgency: str     # Critical / High / Normal / Low


class TicketSummary(BaseModel):
    issue_summary: str
    root_cause: str
    suggested_action: str
    sentiment: str   # Angry / Neutral / Satisfied


class DraftReply(BaseModel):
    subject: str
    body: str
class AgentState(BaseModel):
    csv_path: str = "tickets.csv"
    raw_tickets: List[dict] = Field(default_factory=list)
    ticket_categories: List[dict] = Field(default_factory=list)
    ticket_summaries: List[dict] = Field(default_factory=list)
    critical_replies: List[dict] = Field(default_factory=list)
