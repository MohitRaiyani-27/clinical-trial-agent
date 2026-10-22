"""Lightweight BM25 retrieval over the long free-text parts of a trial.

Eligibility criteria and outcome descriptions are often thousands of words.
Instead of stuffing the whole record into the prompt, the agent retrieves the
few passages that answer the question and cites them by section and line.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .ctgov import Trial

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "who", "must",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Passage:
    nct_id: str
    section: str  # e.g. "inclusion", "exclusion", "primary_outcome"
    index: int
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.nct_id}:{self.section}#{self.index}]"


def chunk_trial(trial: Trial) -> list[Passage]:
    """Split a trial into small, individually citable passages."""
    passages: list[Passage] = []
    section = "eligibility"
    idx = 0
    for raw in trial.eligibility_criteria.splitlines():
        line = raw.strip().lstrip("*-• ").strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("inclusion criteria"):
            section, idx = "inclusion", 0
            continue
        if low.startswith("exclusion criteria"):
            section, idx = "exclusion", 0
            continue
        idx += 1
        passages.append(Passage(trial.nct_id, section, idx, line))
    for kind, outcomes in (("primary_outcome", trial.primary_outcomes),
                           ("secondary_outcome", trial.secondary_outcomes)):
        for i, o in enumerate(outcomes, 1):
            text = f"{o.measure}. Time frame: {o.time_frame}. {o.description}".strip()
            passages.append(Passage(trial.nct_id, kind, i, text))
    if trial.summary:
        passages.append(Passage(trial.nct_id, "summary", 1, trial.summary))
    return passages


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [tokenize(p.text) for p in passages]
        self.avgdl = (sum(map(len, self.docs)) / len(self.docs)) if self.docs else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5, section: str | None = None) -> list[tuple[Passage, float]]:
        q = tokenize(query)
        scored: list[tuple[Passage, float]] = []
        for p, d in zip(self.passages, self.docs):
            if section and p.section != section:
                continue
            tf = Counter(d)
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1))
                score += self.idf.get(t, 0.0) * num / den
            if score > 0:
                scored.append((p, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
