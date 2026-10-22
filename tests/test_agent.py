import json
import unittest

from evals.build_dataset import questions_for
from evals.graders import grade
from trial_agent.agent import TrialAgent
from trial_agent.ctgov import FixtureClient, parse_study
from trial_agent.llm import LLMResponse, ScriptedLLM, ToolCall
from trial_agent.retrieval import BM25Index, chunk_trial
from trial_agent.tools import TOOL_SPECS, Toolbox

from tests.fixtures import STUDIES, STUDY_A


class ParseTests(unittest.TestCase):
    def test_parse_fields(self):
        t = parse_study(STUDY_A)
        self.assertEqual(t.nct_id, "NCT00000001")
        self.assertEqual(t.phases, ["PHASE3"])
        self.assertEqual(t.enrollment, 420)
        self.assertEqual(t.sponsor, "Example Pharma Inc")
        self.assertIn("Drug: Example Drug X", t.interventions)
        self.assertEqual(t.primary_outcomes[0].time_frame, "Week 16")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.passages = chunk_trial(parse_study(STUDY_A))

    def test_chunking_splits_inclusion_and_exclusion(self):
        sections = {p.section for p in self.passages}
        self.assertTrue({"inclusion", "exclusion", "primary_outcome"} <= sections)
        excl = [p for p in self.passages if p.section == "exclusion"]
        self.assertEqual(excl[0].citation, "[NCT00000001:exclusion#1]")

    def test_bm25_finds_relevant_exclusion(self):
        hits = BM25Index(self.passages).search("previous IL-17 biologic treatment", k=1)
        self.assertEqual(hits[0][0].section, "exclusion")
        self.assertIn("IL-17", hits[0][0].text)

    def test_section_filter(self):
        hits = BM25Index(self.passages).search("psoriasis", section="inclusion")
        self.assertTrue(hits and all(p.section == "inclusion" for p, _ in hits))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tb = Toolbox(FixtureClient(STUDIES))

    def test_specs_match_handlers(self):
        self.assertEqual({s["name"] for s in TOOL_SPECS}, set(self.tb._handlers))

    def test_search_filters(self):
        out = json.loads(self.tb.call("search_trials", {"condition": "psoriasis", "phase": "PHASE3"}))
        self.assertEqual([r["nct_id"] for r in out["results"]], ["NCT00000001"])

    def test_errors_are_returned_not_raised(self):
        out = json.loads(self.tb.call("get_trial", {"nct_id": "NCT99999999"}))
        self.assertIn("error", out)
        out = json.loads(self.tb.call("not_a_tool", {}))
        self.assertIn("error", out)

    def test_compare(self):
        out = json.loads(self.tb.call("compare_trials", {"nct_ids": ["NCT00000001", "NCT00000002"]}))
        self.assertEqual([r["enrollment"] for r in out["trials"]], [420, 120])


class AgentLoopTests(unittest.TestCase):
    def _agent(self, script):
        llm = ScriptedLLM(script)
        return TrialAgent(llm, Toolbox(FixtureClient(STUDIES))), llm

    def test_tool_then_answer(self):
        agent, llm = self._agent([
            LLMResponse(tool_calls=[ToolCall("t1", "search_trial_text",
                                             {"nct_id": "NCT00000001", "query": "IL-17 biologic"})]),
            LLMResponse(text="Yes. Prior IL-17 biologics are excluded in NCT00000001 [NCT00000001:exclusion#1]."),
        ])
        res = agent.run("Can patients who had an IL-17 biologic join NCT00000001?")
        self.assertEqual(len(res.steps), 1)
        self.assertIn("exclusion#1", res.answer)
        # The tool result must be fed back to the model with the matching tool_use_id.
        last_msg = llm.calls[-1][-1]
        self.assertEqual(last_msg["content"][0]["tool_use_id"], "t1")
        self.assertEqual(res.cited_nct_ids, {"NCT00000001"})
        self.assertTrue(res.cited_nct_ids <= res.retrieved_nct_ids)

    def test_step_limit(self):
        loop = [LLMResponse(tool_calls=[ToolCall(f"t{i}", "get_trial", {"nct_id": "NCT00000001"})]) for i in range(20)]
        agent, _ = self._agent(loop)
        agent.max_steps = 3
        res = agent.run("loop forever")
        self.assertEqual(res.stopped_reason, "max_steps")
        self.assertEqual(len(res.steps), 3)


class GraderTests(unittest.TestCase):
    def setUp(self):
        self.client = FixtureClient(STUDIES)
        self.trial = self.client.get("NCT00000001")

    def _result(self, answer, tool_output_ids=("NCT00000001",)):
        from trial_agent.agent import AgentResult, Step
        return AgentResult("q", answer, steps=[Step("get_trial", {}, json.dumps(list(tool_output_ids)), 0.0)])

    def test_dataset_builds_from_record(self):
        types = {q["type"] for q in questions_for(self.trial)}
        self.assertEqual(types, {"phase", "enrollment", "sponsor", "min_age", "primary_outcome", "abstain"})

    def test_correct_and_grounded(self):
        item = {"type": "enrollment", "expected": 420, "nct_id": "NCT00000001"}
        g = grade(item, self._result("NCT00000001 plans to enroll 420 participants."))
        self.assertTrue(g["correct"] and g["cited"] and g["grounded"])

    def test_hallucinated_id_is_not_grounded(self):
        item = {"type": "phase", "expected": ["PHASE3"], "nct_id": "NCT00000001"}
        g = grade(item, self._result("NCT00000001 is Phase 3, like NCT12345678."))
        self.assertTrue(g["correct"])
        self.assertFalse(g["grounded"])

    def test_abstention(self):
        item = {"type": "abstain", "expected": None, "nct_id": "NCT00000001"}
        self.assertTrue(grade(item, self._result("The p-value is not available in the registry data for NCT00000001."))["correct"])
        self.assertFalse(grade(item, self._result("The p-value was 0.003 for NCT00000001."))["correct"])

    def test_discovery_verified_against_registry(self):
        item = {"type": "discovery", "expected": {"status": "RECRUITING", "phase": "PHASE3"}}
        self.assertTrue(grade(item, self._result("Try NCT00000001."), self.client)["correct"])
        res = self._result("Try NCT00000002.", tool_output_ids=("NCT00000002",))
        self.assertFalse(grade(item, res, self.client)["correct"])


if __name__ == "__main__":
    unittest.main()
