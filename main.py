import json
import os

from dotenv import load_dotenv

load_dotenv()

from pipeline import build_pipeline
from models import AgentState


URGENCY_ORDER = {"Critical": 0, "High": 1, "Normal": 2, "Low": 3}


def main():
    pipeline = build_pipeline()

    print("\n" + "=" * 60)
    print("  SUPPORT TICKET TRIAGE PIPELINE")
    print("=" * 60 + "\n")

    result = pipeline.invoke(AgentState(csv_path="tickets.csv"))

    # Build enriched output records
    cat_map = {c["ticket_id"]: c for c in result["ticket_categories"]}
    sum_map = {s["ticket_id"]: s for s in result["ticket_summaries"]}
    rep_map = {r["ticket_id"]: r for r in result["critical_replies"]}

    output = []
    for ticket in result["raw_tickets"]:
        tid = ticket["id"]
        record = {
            "ticket_id": tid,
            "subject": ticket["subject"],
            "body": ticket["body"],
            **{k: v for k, v in cat_map.get(tid, {}).items() if k != "ticket_id"},
            **{k: v for k, v in sum_map.get(tid, {}).items() if k != "ticket_id"},
        }
        if tid in rep_map:
            record["draft_reply"] = {
                "subject": rep_map[tid]["reply_subject"],
                "body": rep_map[tid]["reply_body"],
            }
        output.append(record)

    # Sort by urgency
    output.sort(key=lambda x: URGENCY_ORDER.get(x.get("urgency", "Low"), 99))

    os.makedirs("output", exist_ok=True)
    out_path = "output/results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


    print("\n" + "=" * 60)
    print("  TRIAGE RESULTS")
    print("=" * 60)
    for rec in output:
        urgency_tag = f"[{rec.get('urgency', '?'):8}]"
        print(f"\n{urgency_tag} {rec['ticket_id']} — {rec['subject']}")
        print(f"  Dept     : {rec.get('department', 'N/A')}")
        print(f"  Sentiment: {rec.get('sentiment', 'N/A')}")
        print(f"  Summary  : {rec.get('issue_summary', 'N/A')}")
        print(f"  Action   : {rec.get('suggested_action', 'N/A')}")
        if "draft_reply" in rec:
            print(f"  *** DRAFT REPLY GENERATED ***")
            print(f"      Subject: {rec['draft_reply']['subject']}")

    print(f"\n\nFull results saved to: {out_path}")
    print(f"Tickets processed: {len(output)}")
    print(f"Critical tickets : {sum(1 for r in output if r.get('urgency') == 'Critical')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
