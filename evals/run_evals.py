"""Run the agent over an eval set and write a JSON + Markdown report.

    export ANTHROPIC_API_KEY=...
    python -m evals.run_evals --data evals/data/eval_set.jsonl --out evals/reports/latest

Compare two models or prompts by running twice with --model / --tag and diffing
the reports. Every per-item trace (tool calls, outputs, answer) is saved so a
failing case can be inspected instead of guessed at.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from trial_agent.agent import TrialAgent
from trial_agent.ctgov import CTGovClient
from trial_agent.llm import AnthropicLLM
from trial_agent.tools import Toolbox

from .graders import grade


def summarize(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["grade"])

    def agg(gs: list[dict]) -> dict:
        return {
            "n": len(gs),
            "accuracy": round(mean(g["correct"] for g in gs), 3),
            "citation_rate": round(mean(g["cited"] for g in gs), 3),
            "grounded_rate": round(mean(g["grounded"] for g in gs), 3),
            "avg_tool_calls": round(mean(g["tool_calls"] for g in gs), 2),
        }

    return {"overall": agg([r["grade"] for r in rows]), "by_type": {k: agg(v) for k, v in sorted(by_type.items())}}


def to_markdown(summary: dict, meta: dict) -> str:
    lines = [f"# Eval report: {meta['tag']}", "", f"Model: `{meta['model']}` | items: {summary['overall']['n']}", "",
             "| slice | n | accuracy | citation | grounded | avg tool calls |", "|---|---|---|---|---|---|"]
    for name, s in [("overall", summary["overall"]), *summary["by_type"].items()]:
        lines.append(f"| {name} | {s['n']} | {s['accuracy']:.0%} | {s['citation_rate']:.0%} | "
                     f"{s['grounded_rate']:.0%} | {s['avg_tool_calls']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="evals/data/eval_set.jsonl")
    ap.add_argument("--out", default="evals/reports/latest")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]

    client = CTGovClient()
    llm = AnthropicLLM(model=args.model)
    agent = TrialAgent(llm, Toolbox(client))

    rows = []
    for item in items:
        res = agent.run(item["question"])
        g = grade(item, res, client)
        rows.append({**item, "grade": g, "answer": res.answer, "latency_s": round(res.latency_s, 2),
                     "tokens": res.input_tokens + res.output_tokens,
                     "steps": [asdict(s) for s in res.steps]})
        print(f"[{item['id']:>3}] {item['type']:<16} correct={g['correct']!s:<5} "
              f"grounded={g['grounded']!s:<5} tools={g['tool_calls']}")

    summary = summarize(rows)
    meta = {"tag": args.tag, "model": llm.model}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2))
    (out / "report.md").write_text(to_markdown(summary, meta))
    print("\n" + to_markdown(summary, meta))


if __name__ == "__main__":
    main()
