# Clinical Trial Research Agent

An LLM agent that answers questions about clinical trials the way a clinical development analyst would: it searches **ClinicalTrials.gov**, pulls the exact eligibility or outcome passage it needs, and answers **with citations to the source record**. It ships with an **eval harness** that measures whether the answers are right, grounded and cited, so you know it works instead of hoping it does.

```
$ python -m trial_agent "Are patients previously treated with IL-17 inhibitors excluded from NCT0XXXXXXX?" --trace
--- step 1: search_trial_text({"nct_id": "NCT0XXXXXXX", "query": "prior IL-17 inhibitor"})
...
Yes. Prior IL-17 therapy is an exclusion criterion [NCT0XXXXXXX:exclusion#2].
```

## Why

Drug development teams spend hours reading trial registrations to benchmark designs, check eligibility overlap and scan the competitive landscape. The answers are in public, structured records, but they're spread across long free-text fields. This agent automates that lookup, and its eval suite holds it to the standard regulated work requires: **never invent a trial, always show the source, and say "not available" when the data isn't there.**

## How it works

```
question ──► Claude (tool use) ──► tools ──► ClinicalTrials.gov API v2
                 ▲                   │
                 └──── observations ◄┘     (loop, max 8 steps)
                 │
                 ▼
     answer with NCT IDs + passage citations
```

| Component | What it does |
|---|---|
| `trial_agent/ctgov.py` | Stdlib client for the ClinicalTrials.gov v2 API; parses study JSON into a typed `Trial` |
| `trial_agent/retrieval.py` | Splits eligibility criteria and outcomes into individually citable passages (`[NCT…:exclusion#3]`) and ranks them with BM25 |
| `trial_agent/tools.py` | 4 tools: `search_trials`, `get_trial`, `search_trial_text`, `compare_trials`. Errors go back to the model, not up the stack, so it can recover |
| `trial_agent/agent.py` | Tool-use loop with a step budget and a full trace (tool, input, output, latency, tokens) |
| `trial_agent/llm.py` | Claude via the Messages API with native tool use, plus a `ScriptedLLM` for deterministic tests. The model is swappable with `--model` |
| `evals/` | Dataset builder, graders and runner (below) |

**Design choices**
- **Retrieval over stuffing.** Eligibility sections can run to thousands of words. Retrieving the 3 to 5 relevant lines keeps prompts small and makes every claim traceable to a line.
- **Citations are enforced.** The system prompt requires them, and the evals measure them.
- **Zero dependencies.** Pure Python standard library, so it's easy to run, audit and deploy.

## Evals: how we know it works

`evals/build_dataset.py` **generates questions whose ground truth comes straight from the registry's structured fields**, so nothing is hand-labelled and the set can be rebuilt for any therapeutic area:

| Type | Example | Graded by |
|---|---|---|
| phase / enrollment / sponsor / min_age | "What is the enrollment for NCT…?" | exact or numeric match against the API record |
| primary_outcome | "What is the primary outcome measure of NCT…?" | token recall vs. registered measure |
| discovery | "Find a recruiting phase 3 trial for psoriasis" | **re-checks the cited trial against the registry** (status + phase) |
| abstain | "What was the exact p-value for …?" | must say the data isn't available (tests hallucination) |

Every item is also scored on:
- **cited**: does the answer cite the trial it's about?
- **grounded**: was every NCT ID in the answer actually returned by a tool? This catches invented trial IDs.
- **tool calls**: the efficiency of the trajectory.

```bash
export ANTHROPIC_API_KEY=sk-...
python -m evals.build_dataset --conditions "plaque psoriasis" "non-small cell lung cancer" "type 2 diabetes" --per-condition 5
python -m evals.run_evals --tag baseline
python -m evals.run_evals --tag other-model --model <model-id>   # compare models / prompts
```

Reports (`report.md`, `summary.json`, and per-item traces in `results.jsonl`) are written to `evals/reports/<tag>/`.

## Quickstart

```bash
git clone https://github.com/MohitRaiyani-27/clinical-trial-agent.git
cd clinical-trial-agent
python -m unittest -v                     # offline tests, no API key needed
export ANTHROPIC_API_KEY=sk-...
python -m trial_agent "Compare the primary endpoints of recruiting phase 3 psoriasis trials" --trace
```

Requires Python 3.10+.

## Tests

`tests/` covers parsing, chunking and BM25, tool dispatch and error handling, the agent loop (with a scripted model, so it's deterministic), and every grader, including the hallucinated-ID and abstention checks. The test fixtures are **synthetic** records in the API's shape, not real studies.

## Roadmap

- [ ] Embedding retrieval (hybrid BM25 + dense) and a retrieval-recall eval
- [ ] LLM-as-judge grader for free-text comparison questions, calibrated against human labels
- [ ] Protocol PDF ingestion (full eligibility and schedule of assessments)
- [ ] FastAPI service + Streamlit UI

## Disclaimer

For research and demonstration only. Not medical advice. Data: ClinicalTrials.gov (U.S. National Library of Medicine).

## License

MIT
