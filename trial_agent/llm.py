"""Model backends. The agent only depends on the small `LLM` protocol below,
so swapping models (or replaying a scripted model in tests) is one line."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> LLMResponse: ...


class AnthropicLLM:
    """Claude via the Messages API with native tool use (stdlib HTTP, no SDK)."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_tokens: int = 1500, temperature: float = 0.0):
        self.model = model or os.environ.get("TRIAL_AGENT_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the live model.")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.url = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")

    def complete(self, system, messages, tools) -> LLMResponse:
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": system, "messages": messages, "tools": tools,
        }).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        out = LLMResponse(raw_content=data.get("content", []),
                          input_tokens=data.get("usage", {}).get("input_tokens", 0),
                          output_tokens=data.get("usage", {}).get("output_tokens", 0))
        for block in out.raw_content:
            if block.get("type") == "text":
                out.text += block.get("text", "")
            elif block.get("type") == "tool_use":
                out.tool_calls.append(ToolCall(block["id"], block["name"], block.get("input", {})))
        return out


class ScriptedLLM:
    """Replays a fixed list of responses. Used to unit-test the agent loop
    deterministically without network access or API cost."""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, system, messages, tools) -> LLMResponse:
        self.calls.append(json.loads(json.dumps(messages)))
        if not self.script:
            return LLMResponse(text="(script exhausted)")
        resp = self.script.pop(0)
        if not resp.raw_content:
            resp.raw_content = ([{"type": "text", "text": resp.text}] if resp.text else []) + [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls
            ]
        return resp
