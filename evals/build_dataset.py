"""Build an eval set whose ground truth comes straight from ClinicalTrials.gov.

No answers are hand-written: every expected value is read from the structured API
record, so the eval set can be regenerated for any therapeutic area and never goes
stale. Usage:

    python -m evals.build_dataset --conditions "psoriasis" "non-small cell lung cancer" \
        --per-condition 6 --out evals/data/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from trial_agent.ctgov import CTGovClient, Trial


def questions_for(trial: Trial) -> list[dict]:
    n = trial.nct_id
    items: list[dict] = []
    if trial.phases and trial.phases != ["NA"]:
        items.append({"type": "phase", "question": f"What phase is clinical trial {n}?",
                      "expected": trial.phases, "nct_id": n})
    if trial.enrollment:
        items.append({"type": "enrollment", "question": f"What is the enrollment for {n}?",
                      "expected": trial.enrollment, "nct_id": n})
    if trial.sponsor:
        items.append({"type": "sponsor", "question": f"Who is the lead sponsor of {n}?",
                      "expected": trial.sponsor, "nct_id": n})
    if trial.minimum_age:
        items.append({"type": "min_age", "question": f"What is the minimum participant age for {n}?",
                      "expected": trial.minimum_age, "nct_id": n})
    if trial.primary_outcomes:
        items.append({"type": "primary_outcome",
                      "question": f"What is the primary outcome measure of {n}?",
                      "expected": trial.primary_outcomes[0].measure, "nct_id": n})
    # Abstention probe: posted p-values are not exposed by the agent's tools,
    # so the correct behaviour is to say the data is not available.
    items.append({"type": "abstain",
                  "question": f"What was the exact p-value for the primary endpoint of {n}?",
                  "expected": None, "nct_id": n})
    return items


def discovery_questions(condition: str) -> list[dict]:
    return [{
        "type": "discovery",
        "question": f"Find a currently recruiting phase 3 trial for {condition}. Give its NCT ID.",
        "expected": {"status": "RECRUITING", "phase": "PHASE3"},
        "condition": condition,
    }]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--per-condition", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="evals/data/eval_set.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    client = CTGovClient()
    rows: list[dict] = []
    for cond in args.conditions:
        trials = client.search(condition=cond, page_size=max(args.per_condition * 2, 10))
        for t in random.sample(trials, k=min(args.per_condition, len(trials))):
            rows.extend(questions_for(t))
        rows.extend(discovery_questions(cond))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": i, **r}) + "\n")
    print(f"Wrote {len(rows)} eval items to {out}")


if __name__ == "__main__":
    main()
