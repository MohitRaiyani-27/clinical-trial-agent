"""CLI:  python -m trial_agent "Is NCT01234567 still recruiting, and who can enroll?" [--trace]"""
from __future__ import annotations

import argparse
import json

from .agent import TrialAgent
from .ctgov import CTGovClient
from .llm import AnthropicLLM
from .tools import Toolbox


def main() -> None:
    ap = argparse.ArgumentParser(description="Clinical trial research agent")
    ap.add_argument("question")
    ap.add_argument("--model", default=None)
    ap.add_argument("--trace", action="store_true", help="print every tool call")
    args = ap.parse_args()

    agent = TrialAgent(AnthropicLLM(model=args.model), Toolbox(CTGovClient()))
    res = agent.run(args.question)
    if args.trace:
        for i, s in enumerate(res.steps, 1):
            print(f"--- step {i}: {s.tool}({json.dumps(s.input)}) [{s.latency_s:.2f}s]")
            print(s.output[:800] + ("..." if len(s.output) > 800 else ""))
        print("---")
    print(res.answer)
    print(f"\n({len(res.steps)} tool calls, {res.input_tokens + res.output_tokens} tokens, {res.latency_s:.1f}s)")


if __name__ == "__main__":
    main()
