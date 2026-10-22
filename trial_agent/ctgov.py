"""Thin client for the public ClinicalTrials.gov API v2.

Only the Python standard library is used so the agent runs anywhere.
Docs: https://clinicaltrials.gov/data-api/api
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
NCT_RE = re.compile(r"^NCT\d{8}$")


@dataclass
class Outcome:
    measure: str
    time_frame: str = ""
    description: str = ""


@dataclass
class Trial:
    """The subset of a study record the agent reasons over."""

    nct_id: str
    title: str
    status: str = ""
    phases: list[str] = field(default_factory=list)
    study_type: str = ""
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    sponsor: str = ""
    enrollment: int | None = None
    start_date: str = ""
    primary_completion_date: str = ""
    summary: str = ""
    eligibility_criteria: str = ""
    sex: str = ""
    minimum_age: str = ""
    maximum_age: str = ""
    primary_outcomes: list[Outcome] = field(default_factory=list)
    secondary_outcomes: list[Outcome] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    def overview(self) -> dict[str, Any]:
        """Compact, citation-ready view returned to the LLM by tools."""
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "status": self.status,
            "phases": self.phases,
            "conditions": self.conditions,
            "interventions": self.interventions,
            "sponsor": self.sponsor,
            "enrollment": self.enrollment,
            "start_date": self.start_date,
            "primary_completion_date": self.primary_completion_date,
            "sex": self.sex,
            "minimum_age": self.minimum_age,
            "maximum_age": self.maximum_age,
            "primary_outcomes": [o.measure for o in self.primary_outcomes],
            "url": self.url,
        }


def _outcomes(items: list[dict[str, Any]] | None) -> list[Outcome]:
    return [
        Outcome(
            measure=i.get("measure", ""),
            time_frame=i.get("timeFrame", ""),
            description=i.get("description", ""),
        )
        for i in (items or [])
    ]


def parse_study(study: dict[str, Any]) -> Trial:
    """Convert a raw API v2 study JSON object into a Trial."""
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    elig = p.get("eligibilityModule", {})
    outcomes = p.get("outcomesModule", {})
    arms = p.get("armsInterventionsModule", {})
    enrollment = design.get("enrollmentInfo", {}).get("count")
    return Trial(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle") or ident.get("officialTitle", ""),
        status=status.get("overallStatus", ""),
        phases=design.get("phases", []) or [],
        study_type=design.get("studyType", ""),
        conditions=p.get("conditionsModule", {}).get("conditions", []) or [],
        interventions=[
            f"{i.get('type', '').title()}: {i.get('name', '')}".strip(": ")
            for i in arms.get("interventions", []) or []
        ],
        sponsor=p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
        enrollment=int(enrollment) if enrollment is not None else None,
        start_date=status.get("startDateStruct", {}).get("date", ""),
        primary_completion_date=status.get("primaryCompletionDateStruct", {}).get("date", ""),
        summary=p.get("descriptionModule", {}).get("briefSummary", ""),
        eligibility_criteria=elig.get("eligibilityCriteria", ""),
        sex=elig.get("sex", ""),
        minimum_age=elig.get("minimumAge", ""),
        maximum_age=elig.get("maximumAge", ""),
        primary_outcomes=_outcomes(outcomes.get("primaryOutcomes")),
        secondary_outcomes=_outcomes(outcomes.get("secondaryOutcomes")),
    )


class CTGovClient:
    """Fetches studies over HTTP and caches them in memory for the session."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 20.0):
        self.base_url = base_url
        self.timeout = timeout
        self._cache: dict[str, Trial] = {}

    def _get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def search(
        self,
        condition: str = "",
        term: str = "",
        intervention: str = "",
        status: str = "",
        phase: str = "",
        page_size: int = 10,
    ) -> list[Trial]:
        params: dict[str, str] = {"pageSize": str(max(1, min(page_size, 50))), "format": "json"}
        if condition:
            params["query.cond"] = condition
        if term:
            params["query.term"] = term
        if intervention:
            params["query.intr"] = intervention
        if status:
            params["filter.overallStatus"] = status.upper()
        if phase:
            # Phase is filtered through the Essie expression syntax on AREA[Phase].
            expr = f"AREA[Phase]{phase.upper()}"
            params["filter.advanced"] = expr
        data = self._get(f"{self.base_url}?{urllib.parse.urlencode(params)}")
        trials = [parse_study(s) for s in data.get("studies", [])]
        for t in trials:
            self._cache[t.nct_id] = t
        return trials

    def get(self, nct_id: str) -> Trial:
        nct_id = nct_id.strip().upper()
        if not NCT_RE.match(nct_id):
            raise ValueError(f"Not a valid NCT ID: {nct_id!r}")
        if nct_id not in self._cache:
            self._cache[nct_id] = parse_study(self._get(f"{self.base_url}/{nct_id}"))
        return self._cache[nct_id]


class FixtureClient(CTGovClient):
    """Offline client backed by saved API responses (used for tests and replayable evals)."""

    def __init__(self, studies: list[dict[str, Any]]):
        super().__init__()
        self._trials = [parse_study(s) for s in studies]
        self._cache = {t.nct_id: t for t in self._trials}

    def search(self, condition="", term="", intervention="", status="", phase="", page_size=10):
        def hit(t: Trial) -> bool:
            hay = " ".join([t.title, t.summary, *t.conditions, *t.interventions]).lower()
            checks = [
                not condition or condition.lower() in " ".join(t.conditions + [t.title]).lower(),
                not term or term.lower() in hay,
                not intervention or intervention.lower() in " ".join(t.interventions).lower(),
                not status or status.upper() == t.status,
                not phase or phase.upper() in t.phases,
            ]
            return all(checks)

        return [t for t in self._trials if hit(t)][:page_size]

    def get(self, nct_id: str) -> Trial:
        nct_id = nct_id.strip().upper()
        if nct_id not in self._cache:
            raise KeyError(f"{nct_id} not in fixture set")
        return self._cache[nct_id]
