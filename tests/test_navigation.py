import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from decision_tree.__main__ import main
from decision_tree.judges import ENDPOINT, JevJudge, JudgeError, LexicalJudge, _NoRedirect, validate_scores
from decision_tree.models import Candidate, Document, Passage, load_corpus, validate_corpus
from decision_tree.navigate import retrieve

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus.json"


class ScriptedJudge:
    label = "scripted test double"

    def __init__(self):
        self.stages = []
        self.choice_candidates = []

    def score(self, query, candidates, stage):
        self.stages.append((stage, candidates))
        values = {"a": .8, "b": .9, "c": .1,
                  "a1": .99, "a2": .7, "b1": .8, "b2": .2}
        return {c.id: values[c.id] for c in candidates}

    def choose(self, query, candidates):
        self.choice_candidates = candidates
        return {candidates[0].id: .1, candidates[1].id: .9}


def corpus():
    return (Document("a", "A", "summary A", (Passage("a1", "evidence A1"), Passage("a2", "evidence A2"))),
            Document("b", "B", "summary B", (Passage("b1", "evidence B1"), Passage("b2", "evidence B2"))),
            Document("c", "C", "summary C", (Passage("c1", "PRUNED_SECRET_TEXT"),)))


class NavigationTests(unittest.TestCase):
    def test_shared_stage_and_irreversible_pruning(self):
        judge = ScriptedJudge()
        result = retrieve("query", corpus(), judge)
        self.assertEqual([name for name, _ in judge.stages], ["documents", "passages"])
        self.assertEqual([c.id for c in judge.stages[0][1]], ["a", "b", "c"])
        self.assertNotIn("PRUNED_SECRET_TEXT", str(judge.stages))
        self.assertEqual(result["trace"]["pruned_documents"], ["c"])
        self.assertEqual(result["trace"]["retained_documents"], ["b", "a"])
        self.assertEqual(result["trace"]["scored_passages"], ["a1", "a2", "b1", "b2"])

    def test_global_passage_budget_not_per_document(self):
        result = retrieve("q", corpus(), ScriptedJudge(), passage_budget=2)
        self.assertEqual([r["id"] for r in result["results"]], ["a1", "b1"])

    def test_only_first_m_reranked_tail_unchanged(self):
        judge = ScriptedJudge()
        result = retrieve("q", corpus(), judge, passage_budget=4, rerank=2)
        self.assertEqual([c.id for c in judge.choice_candidates], ["a1", "b1"])
        self.assertEqual([r["id"] for r in result["results"]], ["b1", "a1", "a2", "b2"])
        self.assertIsNone(result["results"][2]["choice_probability"])

    def test_single_candidate_does_not_make_choice_call(self):
        judge = ScriptedJudge()
        result = retrieve("q", corpus(), judge, passage_budget=1, rerank=5)
        self.assertEqual(judge.choice_candidates, [])
        self.assertIsNone(result["trace"]["choice_scores"])

    def test_document_weighting_is_explicit(self):
        result = retrieve("q", corpus(), ScriptedJudge(), weight_by_document=True)
        scores = {r["id"]: r for r in result["results"]}
        self.assertAlmostEqual(scores["a1"]["ranking_score"], .99 * .8)
        self.assertAlmostEqual(scores["b1"]["ranking_score"], .8 * .9)

    def test_binary_probabilities_need_not_sum_to_one(self):
        self.assertEqual(validate_scores({"a": .8, "b": .9}, [Candidate("a", ""), Candidate("b", "")]),
                         {"a": .8, "b": .9})

    def test_ties_keep_corpus_order(self):
        judge = ScriptedJudge()
        judge.score = lambda q, cs, s: {c.id: .5 for c in reversed(cs)}
        result = retrieve("q", corpus(), judge, document_budget=1)
        self.assertEqual(result["trace"]["retained_documents"], ["a"])
        self.assertEqual([r["id"] for r in result["results"]], ["a1", "a2"])

    def test_bad_scores_stop_before_next_stage(self):
        for bad in ({"a": .5}, {"a": float("nan"), "b": .5, "c": .2},
                    {"a": True, "b": .5, "c": .2}, {"a": 1.1, "b": .5, "c": .2},
                    {"a": 10 ** 400, "b": .5, "c": .2}):
            with self.subTest(bad=bad):
                judge = ScriptedJudge()
                judge.score = lambda *args: bad
                with self.assertRaises(JudgeError):
                    retrieve("q", corpus(), judge)
                self.assertEqual(judge.stages, [])

    def test_bad_choice_distribution_stops(self):
        judge = ScriptedJudge()
        judge.choose = lambda q, cs: {c.id: .8 for c in cs}
        with self.assertRaises(JudgeError):
            retrieve("q", corpus(), judge, rerank=2)

    def test_invalid_budgets_and_queries(self):
        for kwargs in ({"document_budget": 0}, {"passage_budget": True}, {"rerank": -1},
                       {"rerank": 256}, {"weight_by_document": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                retrieve("q", corpus(), ScriptedJudge(), **kwargs)
        with self.assertRaises(ValueError):
            retrieve(" ", corpus(), ScriptedJudge())

    def test_duplicate_ids_and_empty_corpus_rejected(self):
        for value in ([], [Document("a", "A", "s", ())],
                      [Document("a", "A", "s", (Passage("a", "p"),))]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_corpus(value)

    def test_offline_demo_and_punctuation_query(self):
        documents = load_corpus(CORPUS)
        result = retrieve("How long can I restore a deleted object?", documents, LexicalJudge(), rerank=3)
        self.assertEqual(result["results"][0]["id"], "recover-window")
        self.assertFalse(result["quality_evidence"])
        self.assertEqual(len(retrieve("???", documents, LexicalJudge())["results"]), 3)

    def test_loader_invalid_shapes(self):
        for value in ({}, [1], [{"id": "a"}]):
            with self.subTest(value=value), patch("pathlib.Path.open", return_value=io.BytesIO(json.dumps(value).encode())):
                with self.assertRaises(ValueError):
                    load_corpus("unused.json")

    def test_loader_large_input_rejected(self):
        with patch("pathlib.Path.open", return_value=io.BytesIO(b" " * (4 * 1024 * 1024 + 1))):
            with self.assertRaisesRegex(ValueError, "4 MiB"):
                load_corpus("unused.json")

    def test_rerank_budget_clamped_to_retained_count(self):
        judge = ScriptedJudge()
        result = retrieve("q", corpus(), judge, passage_budget=2, rerank=5)
        self.assertEqual(len(judge.choice_candidates), 2)
        self.assertEqual([r["id"] for r in result["results"]], ["b1", "a1"])

    def test_cli_outputs_json_and_handles_errors(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--rerank", "3"]), 0)
        self.assertEqual(json.loads(out.getvalue())["judge_calls"], 3)
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(main(["--documents", "0"]), 1)
        self.assertIn("positive integer", error.getvalue())


class JevTests(unittest.TestCase):
    def response(self, value):
        return io.BytesIO(json.dumps(value).encode())

    def test_redirects_rejected_before_forwarding_credentials(self):
        request = Request(ENDPOINT, headers={"Authorization": "Bearer test-placeholder"})
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code), self.assertRaises(HTTPError) as failure:
                _NoRedirect().redirect_request(request, None, code, "redirect", {}, "https://other.invalid/")
            self.assertEqual(failure.exception.code, code)

    def test_real_http_shape_shared_context(self):
        judge = JevJudge(api_key="test-placeholder")
        candidates = [Candidate("one", "first"), Candidate("two", "second")]
        answer = {"answers": {"relevant_0": {"type": "noul", "noul": .8},
                               "relevant_1": {"type": "noul", "noul": .7}}}
        with patch("decision_tree.judges.urlopen", return_value=self.response(answer)) as call:
            result = judge.score("query", candidates, "documents")
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, ENDPOINT)
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "jev-1.13.0")
        self.assertEqual(len(payload["state"]["candidates"]), 2)
        self.assertEqual(len(payload["questions"]), 2)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result, {"one": .8, "two": .7})
        self.assertNotIn("test-placeholder", json.dumps(payload))

    def test_choice_shape_and_mapping(self):
        judge = JevJudge(api_key="test-placeholder")
        answer = {"answers": {"best": {"type": "choice", "choice": "p1",
                                         "probabilities": {"p0": .2, "p1": .8}}}}
        with patch("decision_tree.judges.urlopen", return_value=self.response(answer)) as call:
            result = judge.choose("q", [Candidate("odd/id", "a"), Candidate("other", "b")])
        self.assertEqual(result, {"odd/id": .2, "other": .8})
        self.assertEqual(json.loads(call.call_args.args[0].data)["questions"]["best"]["type"], "choice")

    def test_malformed_http_answers_fail_closed(self):
        for response in ([], {}, {"answers": {}}, {"answers": {"relevant_0": {"type": "choice"}}},
                         {"answers": {"relevant_0": {"type": "noul", "noul": float("inf")}}},
                         {"answers": {"relevant_0": {"type": "noul", "noul": "0.9"}}}):
            with self.subTest(response=response), patch("decision_tree.judges.urlopen", return_value=self.response(response)):
                with self.assertRaises(JudgeError):
                    JevJudge(api_key="test-placeholder").score("q", [Candidate("a", "x")], "documents")

    def test_bad_choice_options_and_distribution(self):
        for probabilities in ({"p0": .8}, {"p0": .8, "p1": .8}, {"p0": True, "p1": 0}):
            answer = {"answers": {"best": {"type": "choice", "choice": "p0", "probabilities": probabilities}}}
            with self.subTest(probabilities=probabilities), patch("decision_tree.judges.urlopen", return_value=self.response(answer)):
                with self.assertRaises(JudgeError):
                    JevJudge(api_key="test-placeholder").choose("q", [Candidate("a", "x"), Candidate("b", "y")])

    def test_http_failure_does_not_echo_body_or_key(self):
        for failure in (HTTPError(ENDPOINT, 401, "SECRET_SERVER_TEXT", {}, None), URLError("SECRET_SERVER_TEXT")):
            with self.subTest(failure=failure), patch("decision_tree.judges.urlopen", side_effect=failure):
                with self.assertRaises(JudgeError) as error:
                    JevJudge(api_key="test-placeholder").score("q", [Candidate("a", "x")], "documents")
                self.assertNotIn("SECRET", str(error.exception))
                self.assertNotIn("test-placeholder", str(error.exception))

    def test_large_request_never_opens_connection(self):
        with patch("decision_tree.judges.urlopen") as call, self.assertRaises(JudgeError):
            JevJudge(api_key="test-placeholder").score("q", [Candidate("a", "x" * (257 * 1024))], "documents")
        call.assert_not_called()

    def test_bad_json_and_large_response(self):
        for raw in (b"invalid json", b"x" * (1024 * 1024 + 1)):
            with self.subTest(size=len(raw)), patch("decision_tree.judges.urlopen", return_value=io.BytesIO(raw)):
                with self.assertRaises(JudgeError):
                    JevJudge(api_key="test-placeholder").score("q", [Candidate("a", "x")], "documents")

    def test_missing_key_and_timeout_validation(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            JevJudge()
        for timeout in (True, 0, 301, float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                JevJudge(api_key="test-placeholder", timeout=timeout)


if __name__ == "__main__":
    unittest.main()
