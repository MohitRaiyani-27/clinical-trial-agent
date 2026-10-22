"""The agent loop: plan -> call tools -> observe -> answer with citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .llm import LLM
from .tools import TOOL_SPECS, Toolbox

SYSTEM_PROMPT = """You are a clinical development research assistant used by drug development teams.
You answer questions about clinical trials using ONLY data returned by your tools (ClinicalTrials.gov).

Rules:
1. Never answer from memory. If you have not retrieved it with a tool, you do not know it.
2. Cite every factual claim: use the study's NCT ID (e.g. NCT01234567) and, for eligibility or
   outcome details, the passage tag returned by search_trial_text (e.g. [NCT01234567:exclusion#3]).
3. If the tools do not contain the answer, say so plainly instead of guessing.
4. Prefer the fewest tool calls that fully answer the question.
5. Keep answers short and structured: a one-line direct answer first, then supporting detail."""

NCT_IN_TEXT = re.compile(r"NCT\d{8}")


@dataclass
class Step:
    tool: str
    input: dict[str, Any]
    output: str
    latency_s: float


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stopped_reason: str = "answered"

    @property
    def cited_nct_ids(self) -> set[str]:
        return set(NCT_IN_TEXT.findall(self.answer))

    @property
    def retrieved_nct_ids(self) -> set[str]:
        return {nid for s in self.steps for nid in NCT_IN_TEXT.findall(s.output)}


class TrialAgent:
    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 8, system: str = SYSTEM_PROMPT):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.system = system

    def run(self, question: str) -> AgentResult:
        start = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        result = AgentResult(question=question, answer="")
        for _ in range(self.max_steps):
            resp = self.llm.complete(self.system, messages, TOOL_SPECS)
            result.input_tokens += resp.input_tokens
            result.output_tokens += resp.output_tokens
            if not resp.tool_calls:
                result.answer = resp.text.strip()
                break
            messages.append({"role": "assistant", "content": resp.raw_content})
            tool_results = []
            for call in resp.tool_calls:
                t0 = time.perf_counter()
                output = self.toolbox.call(call.name, call.input)
                result.steps.append(Step(call.name, call.input, output, time.perf_counter() - t0))
                tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
            messages.append({"role": "user", "content": tool_results})
        else:
            result.stopped_reason = "max_steps"
            result.answer = result.answer or "I could not finish within the step budget."
        result.latency_s = time.perf_counter() - start
        return result
