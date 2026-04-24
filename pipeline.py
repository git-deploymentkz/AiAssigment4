import csv
import json
import os
from typing import Any
from pydantic import ValidationError

# from google import genai                  
# from google.genai import types             
from openai import OpenAI
from langgraph.graph import StateGraph, END

from models import AgentState, TicketCategory, TicketSummary, DraftReply
from prompts import CLASSIFY_TICKET_PROMPT, ANALYZE_TICKET_PROMPT, DRAFT_REPLY_PROMPT

try:
    from langsmith import traceable
except ImportError:
    def traceable(func=None, **_):  
        return func if func is not None else lambda f: f


MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

_client: OpenAI | None = None


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object from a possibly noisy model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


@traceable(name="openrouter_structured_call")
def call_structured(prompt: str, response_model: type) -> Any:
    """Call OpenRouter with JSON structured output and parse into a Pydantic model.

    Retries when JSON is malformed/truncated, which can happen on some model responses.
    """
    schema_instruction = (
        f"Respond ONLY with valid JSON that matches this schema exactly:\n"
        f"{json.dumps(response_model.model_json_schema(), indent=2)}"
    )

    client = get_client()
    base_messages = [
        {"role": "system", "content": schema_instruction},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    last_content = ""

    for attempt in range(1, 4):
        response = client.chat.completions.create(
            model=MODEL,
            messages=base_messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        candidate_json = _extract_json_object(content)

        try:
            return response_model.model_validate_json(candidate_json)
        except ValidationError as exc:
            last_error = exc
            last_content = content
            if attempt == 3:
                break

            # Ask the model to self-correct using the previous invalid output.
            base_messages = [
                {"role": "system", "content": schema_instruction},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid JSON or did not match the schema. "
                        "Return only a corrected JSON object that matches the schema exactly."
                    ),
                },
            ]

    raise RuntimeError(
        f"Failed to parse {response_model.__name__} after 3 attempts. "
        f"Last model output: {last_content!r}. Error: {last_error}"
    )


# ── Node 1: Ingest ────────────────────────────────────────────────────────────

def ingest_tickets(state: AgentState) -> dict:
    """Read tickets.csv and load raw ticket dicts into state."""
    tickets = []
    with open(state.csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tickets.append({
                "id": row["id"].strip(),
                "subject": row["subject"].strip(),
                "body": row["body"].strip(),
            })
    print(f"[ingest] Loaded {len(tickets)} tickets from '{state.csv_path}'")
    return {"raw_tickets": tickets}


# ── Node 2: Classify (LLM call #1) ───────────────────────────────────────────

def classify_tickets(state: AgentState) -> dict:
    """LLM call — classify each ticket into department + urgency."""
    categories = []
    for ticket in state.raw_tickets:
        prompt = CLASSIFY_TICKET_PROMPT.format(
            ticket_id=ticket["id"],
            subject=ticket["subject"],
            body=ticket["body"],
        )
        result: TicketCategory = call_structured(prompt, TicketCategory)
        categories.append({
            "ticket_id": ticket["id"],
            "department": result.department,
            "urgency": result.urgency,
        })
        print(f"[classify] {ticket['id']} → {result.department} / {result.urgency}")
    return {"ticket_categories": categories}


# ── Node 3: Analyze (LLM call #2) ────────────────────────────────────────────

def analyze_tickets(state: AgentState) -> dict:
    """LLM call — summarize each ticket (root cause, action, sentiment)."""
    cat_map = {c["ticket_id"]: c for c in state.ticket_categories}
    summaries = []
    for ticket in state.raw_tickets:
        cat = cat_map.get(ticket["id"], {})
        prompt = ANALYZE_TICKET_PROMPT.format(
            ticket_id=ticket["id"],
            subject=ticket["subject"],
            body=ticket["body"],
            department=cat.get("department", "Unknown"),
            urgency=cat.get("urgency", "Unknown"),
        )
        result: TicketSummary = call_structured(prompt, TicketSummary)
        summaries.append({
            "ticket_id": ticket["id"],
            "issue_summary": result.issue_summary,
            "root_cause": result.root_cause,
            "suggested_action": result.suggested_action,
            "sentiment": result.sentiment,
        })
        print(f"[analyze]  {ticket['id']} → {result.sentiment} | {result.issue_summary[:60]}…")
    return {"ticket_summaries": summaries}


# ── Node 4: Draft Critical Replies (LLM call #3, bonus node) ─────────────────

def generate_critical_replies(state: AgentState) -> dict:
    """LLM call — draft a reply for every Critical-urgency ticket."""
    cat_map = {c["ticket_id"]: c for c in state.ticket_categories}
    sum_map = {s["ticket_id"]: s for s in state.ticket_summaries}
    replies = []

    critical_tickets = [
        t for t in state.raw_tickets
        if cat_map.get(t["id"], {}).get("urgency") == "Critical"
    ]
    print(f"[draft]    {len(critical_tickets)} Critical ticket(s) found")

    for ticket in critical_tickets:
        summary = sum_map.get(ticket["id"], {})
        prompt = DRAFT_REPLY_PROMPT.format(
            ticket_id=ticket["id"],
            subject=ticket["subject"],
            body=ticket["body"],
            issue_summary=summary.get("issue_summary", ""),
            suggested_action=summary.get("suggested_action", ""),
        )
        result: DraftReply = call_structured(prompt, DraftReply)
        replies.append({
            "ticket_id": ticket["id"],
            "reply_subject": result.subject,
            "reply_body": result.body,
        })
        print(f"[draft]    {ticket['id']} → reply drafted ✓")

    return {"critical_replies": replies}


# ── Build grap

def build_pipeline():
    graph = StateGraph(AgentState)

    graph.add_node("ingest", ingest_tickets)
    graph.add_node("classify", classify_tickets)
    graph.add_node("analyze", analyze_tickets)
    graph.add_node("draft_replies", generate_critical_replies)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "classify")
    graph.add_edge("classify", "analyze")
    graph.add_edge("analyze", "draft_replies")
    graph.add_edge("draft_replies", END)

    return graph.compile()
