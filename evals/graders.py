"""Graders: each returns a dict of named, independently interpretable checks.

We grade more than "is the answer right":
  * correct    - task-specific correctness against API ground truth
  * cited      - the answer cites the trial it is about
  * grounded   - every NCT ID in the answer was actually returned by a tool
                 (catches hallucinated / invented trial IDs)
"""
from __future__ import annotations

import re
from typing import Any

from trial_agent.agent import AgentResult
from trial_agent.ctgov import CTGovClient

ABSTAIN_PATTERNS = re.compile(
    r"(not (available|reported|provided|included|contain|have)|no (data|results|information)|"
    r"unable to|cannot (find|determine|provide)|don't have|do not have)", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def _numbers(text: str) -> set[int]:
    return {int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text)}


def _token_recall(expected: str, answer: str) -> float:
    exp = {t for t in _norm(expected).split() if len(t) > 3}
    if not exp:
        return 1.0
    ans = set(_norm(answer).split())
    return len(exp & ans) / len(exp)


def grade(item: dict[str, Any], result: AgentResult, client: CTGovClient | None = None) -> dict[str, Any]:
    a = result.answer
    kind = item["type"]
    exp = item.get("expected")
    correct: bool

    if kind == "phase":
        labels = {p: p.replace("PHASE", "phase ").strip() for p in exp}
        correct = all(p in a.upper().replace(" ", "") or labels[p] in a.lower() for p in exp)
    elif kind == "enrollment":
        correct = int(exp) in _numbers(a)
    elif kind == "sponsor":
        correct = _token_recall(exp, a) >= 0.6
    elif kind == "min_age":
        correct = _numbers(exp) <= _numbers(a)
    elif kind == "primary_outcome":
        correct = _token_recall(exp, a) >= 0.5
    elif kind == "abstain":
        correct = bool(ABSTAIN_PATTERNS.search(a))
    elif kind == "discovery":
        correct = False
        if client is not None:
            for nid in result.cited_nct_ids & result.retrieved_nct_ids:
                try:
                    t = client.get(nid)
                except Exception:  # noqa: BLE001
                    continue
                if t.status == exp["status"] and exp["phase"] in t.phases:
                    correct = True
                    break
    else:
        raise ValueError(f"unknown item type {kind}")

    target = item.get("nct_id")
    return {
        "correct": correct,
        "cited": (target in result.cited_nct_ids) if target else bool(result.cited_nct_ids),
        "grounded": result.cited_nct_ids <= result.retrieved_nct_ids,
        "tool_calls": len(result.steps),
        "hit_step_limit": result.stopped_reason == "max_steps",
    }
