"""Tools the agent can call. Each returns JSON-serialisable data with citations."""
from __future__ import annotations

import json
from typing import Any, Callable

from .ctgov import CTGovClient
from .retrieval import BM25Index, chunk_trial

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_trials",
        "description": (
            "Search ClinicalTrials.gov for studies. Use for discovery questions such as "
            "'recruiting phase 3 trials for psoriasis'. Returns compact overviews with NCT IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "Disease or condition, e.g. 'type 2 diabetes'"},
                "intervention": {"type": "string", "description": "Drug, device or procedure name"},
                "term": {"type": "string", "description": "Any other keyword"},
                "status": {"type": "string", "description": "e.g. RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING"},
                "phase": {"type": "string", "description": "e.g. PHASE1, PHASE2, PHASE3, PHASE4"},
                "page_size": {"type": "integer", "description": "Max results (1-20)", "default": 5},
            },
        },
    },
    {
        "name": "get_trial",
        "description": "Get the structured overview of one study by NCT ID (phase, status, enrollment, sponsor, outcomes, ages).",
        "input_schema": {
            "type": "object",
            "properties": {"nct_id": {"type": "string"}},
            "required": ["nct_id"],
        },
    },
    {
        "name": "search_trial_text",
        "description": (
            "Retrieve the most relevant passages from a study's eligibility criteria, outcomes and "
            "summary. Use for detailed questions like 'are patients with prior chemotherapy excluded?'. "
            "Each passage has a citation tag you must quote in the answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nct_id": {"type": "string"},
                "query": {"type": "string"},
                "section": {
                    "type": "string",
                    "enum": ["inclusion", "exclusion", "primary_outcome", "secondary_outcome", "summary"],
                    "description": "Optional section filter",
                },
                "k": {"type": "integer", "default": 5},
            },
            "required": ["nct_id", "query"],
        },
    },
    {
        "name": "compare_trials",
        "description": "Side-by-side comparison of 2-5 studies on design fields (phase, enrollment, status, outcomes, ages).",
        "input_schema": {
            "type": "object",
            "properties": {"nct_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5}},
            "required": ["nct_ids"],
        },
    },
]


class Toolbox:
    def __init__(self, client: CTGovClient):
        self.client = client
        self._indexes: dict[str, BM25Index] = {}
        self._handlers: dict[str, Callable[..., Any]] = {
            "search_trials": self.search_trials,
            "get_trial": self.get_trial,
            "search_trial_text": self.search_trial_text,
            "compare_trials": self.compare_trials,
        }

    # --- tool implementations -------------------------------------------------
    def search_trials(self, condition="", intervention="", term="", status="", phase="", page_size=5):
        trials = self.client.search(condition=condition, term=term, intervention=intervention,
                                    status=status, phase=phase, page_size=min(int(page_size), 20))
        return {"count": len(trials), "results": [t.overview() for t in trials]}

    def get_trial(self, nct_id: str):
        return self.client.get(nct_id).overview()

    def search_trial_text(self, nct_id: str, query: str, section: str | None = None, k: int = 5):
        trial = self.client.get(nct_id)
        if trial.nct_id not in self._indexes:
            self._indexes[trial.nct_id] = BM25Index(chunk_trial(trial))
        hits = self._indexes[trial.nct_id].search(query, k=int(k), section=section)
        return {
            "nct_id": trial.nct_id,
            "passages": [{"citation": p.citation, "section": p.section, "text": p.text,
                          "score": round(s, 3)} for p, s in hits],
        }

    def compare_trials(self, nct_ids: list[str]):
        rows = []
        for nid in nct_ids[:5]:
            t = self.client.get(nid)
            rows.append({
                "nct_id": t.nct_id, "title": t.title, "phases": t.phases, "status": t.status,
                "enrollment": t.enrollment, "sponsor": t.sponsor,
                "ages": f"{t.minimum_age or 'N/A'} - {t.maximum_age or 'N/A'}",
                "primary_outcomes": [o.measure for o in t.primary_outcomes],
            })
        return {"trials": rows}

    # --- dispatch -------------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool and always return a string (errors are returned, not raised,
        so the model can recover, e.g. by fixing a malformed NCT ID)."""
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(handler(**args), ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model on purpose
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
