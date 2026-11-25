"""Offline tests for the full graph, evidence contracts, and free-only routing."""

from io import BytesIO
from time import monotonic
import unittest
from unittest.mock import patch

from agents import Reading, ResearchAgents, Review, SearchPlan, select_evidence
from pipeline import ResearchRun, run_research_pipeline
from providers import (
    GROQ_REVIEW_MODEL, Attempt, ModelError, ModelRequest, ModelRouter, Route, _model_call, _reserve, _routes,
)
from tools import Batch, Candidate, ToolError, _page_html, _resolve_public, canonical_url, public_url

TOPIC = "How do LFP batteries compare with sodium-ion for grid storage?"
PRIMARY = "Stationary storage has different cycle life and energy density requirements than transport."
FOLLOWUP = "The pilot compared usable energy and cycling performance under an identical test protocol."


class FixtureRouter:
    def __init__(self, scenario="normal"):
        self.scenario = scenario
        self.calls = []

    def invoke(self, request, record_attempt, emit):
        payload = request.data
        schema = request.schema
        self.calls.append((schema, request.tier, payload))
        prior_calls = sum(schema is call[0] for call in self.calls)
        interrupted = self.scenario == "critic-timeout" and schema is Review and prior_calls == 1
        limited = self.scenario == "writer-limited" and schema is None and prior_calls == 1
        outcome = "timeout" if interrupted else "rate_limit" if limited else "ok"
        record_attempt(Attempt(provider="fixture", model="scripted", outcome=outcome,
                               seconds=0.01, input_tokens=80, output_tokens=25, estimated_input=80))
        if interrupted:
            raise ModelError("timeout")
        if limited:
            raise ModelError("rate_limit")
        if schema is SearchPlan:
            return SearchPlan(facets=["Storage performance"],
                              queries=["grid storage pilot results" if payload["gap"] else "grid storage evidence"])
        if schema is Reading:
            by_source = {}
            for passage in payload["passages"]:
                by_source.setdefault(passage["source_id"], passage["text"])
            return Reading.model_validate({
                "selections": [
                    {"source_id": source, "facet": "Storage performance", "excerpt": excerpt}
                    for source, excerpt in by_source.items()
                ],
                "gaps": [],
            })
        if schema is Review:
            kind = {"coverage": "coverage", "repair": "unsupported", "persistent": "coverage"}.get(self.scenario)
            needs_work = kind and (prior_calls == 1 or self.scenario == "persistent")
            return Review.model_validate({
                "summary": "Evidence compared with the draft.",
                "findings": [{"kind": kind, "claim": "Pilot field results",
                              "detail": "Find or qualify the source for the field comparison.",
                              "source_ids": ["S1"]}] if needs_work else [],
            })
        reference = "[S1] [S2]" if payload["revision"] and self.scenario == "coverage" else "[S1]"
        if self.scenario == "unknown-citation":
            reference = "[S999]"
        report = "## Findings\n\nGrid storage and transport have different constraints " + reference + "."
        emit({"type": "draft_reset"})
        emit({"type": "draft_delta", "text": report})
        return report


class FixtureTools:
    def __init__(self, fail=False, pages_timeout=False):
        self.fail = fail
        self.pages_timeout = pages_timeout
        self.queries = []
        self.read_batches = []

    def check_search_configuration(self):
        pass

    def search(self, queries, max_results, deadline, publish):
        self.queries.extend(queries)
        if self.fail:
            self.fail = False
            return Batch(attempted=1, errors=["Tavily search timed out."])
        for query in queries:
            followup = "pilot" in query
            host = "research.example.org" if followup else "example.org"
            publish(Candidate(
                url=f"https://{host}/storage", title="Grid storage study", host=host,
                snippet=FOLLOWUP if followup else PRIMARY, score=0.9,
            ))
            if not followup:
                publish(Candidate(url=f"https://{host}/storage?utm_source=repeat", title="Duplicate",
                                  host=host, snippet=PRIMARY, score=0.8))
        return Batch(attempted=len(queries))

    def read(self, sources, deadline, publish):
        self.read_batches.append([source["id"] for source in sources])
        if self.pages_timeout:
            return Batch(attempted=len(sources), errors=["The page could not be read before its deadline."])
        for source in sources:
            publish(source["id"], source["snippet"])
        return Batch(attempted=len(sources))


def fixture_run(mode="fast", scenario="normal", search_failure=False, pages_timeout=False):
    router = FixtureRouter(scenario)
    tools = FixtureTools(fail=search_failure, pages_timeout=pages_timeout)
    return ResearchRun(TOPIC, mode, ResearchAgents(router), tools), router, tools


def execute(run, retry=False):
    events = list(run.events(retry=retry))
    return events, events[-1]["result"]


class PipelineTests(unittest.TestCase):
    def test_blank_topic(self):
        with self.assertRaises(ValueError):
            run_research_pipeline("   ")

    def test_fast_runs_four_actual_graph_nodes_and_emits_evidence_before_drafting(self):
        run, router, tools = fixture_run()
        events, result = execute(run)
        self.assertEqual(result["status"], "complete", result["message"])
        self.assertEqual([stage["stage"] for stage in result["trace"]], ["search", "reader", "writer", "critic"])
        self.assertEqual(result["metrics"]["model_calls"], 4)
        self.assertEqual([call[1] for call in router.calls], ["fast"] * 4)
        self.assertEqual(len(tools.queries), 1)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("[S1]", result["report"])
        event_types = [event["type"] for event in events]
        self.assertLess(event_types.index("source"), event_types.index("draft_delta"))
        self.assertLess(event_types.index("evidence"), event_types.index("draft_delta"))

    def test_deep_research_revisits_search_after_evidence_gap(self):
        run, router, tools = fixture_run("deep", "coverage")
        events, result = execute(run)
        self.assertEqual(result["status"], "complete", result["message"])
        self.assertEqual([stage["stage"] for stage in result["trace"]],
                         ["search", "reader", "writer", "critic"] * 2)
        self.assertEqual([event["target"] for event in events if event["type"] == "routing"], ["search"])
        self.assertEqual(tools.read_batches, [["S1"], ["S2"]])
        self.assertEqual(result["metrics"]["model_calls"], 8)
        self.assertEqual([call[1] for call in router.calls], ["fast", "fast", "deep", "deep"] * 2)
        self.assertEqual({item["source_id"] for item in result["evidence"]}, {"S1", "S2"})

    def test_deep_critique_requests_one_targeted_writer_revision(self):
        run, router, tools = fixture_run("deep", "repair")
        events, result = execute(run)
        stages = ["search", "reader", "writer", "critic", "writer", "critic"]
        self.assertEqual([stage["stage"] for stage in result["trace"]], stages)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([event["target"] for event in events if event["type"] == "routing"], ["writer"])
        self.assertEqual(len(tools.queries), 1)
        self.assertTrue(router.calls[-2][2]["prior_draft"])
        self.assertIn("field comparison", router.calls[-2][2]["revision"])

    def test_deep_stops_at_feedback_budget_when_gap_persists(self):
        run, router, tools = fixture_run("deep", "persistent")
        events, result = execute(run)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["followups"], 1)
        self.assertEqual(result["metrics"]["model_calls"], 8)
        self.assertEqual(sum(event["type"] == "routing" for event in events), 1)

    def test_critic_timeout_preserves_draft_and_resumes_at_critic(self):
        run, router, tools = fixture_run("fast", "critic-timeout")
        _, partial = execute(run)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["retry_stage"], "critic")
        self.assertTrue(partial["report"])
        events, result = execute(run, retry=True)
        self.assertEqual(result["status"], "complete", result["message"])
        self.assertEqual(result["report"], partial["report"])
        self.assertEqual(result["id"], partial["id"])
        self.assertEqual([event["stage"] for event in events if event["type"] == "stage_start"], ["critic"])
        self.assertEqual(result["metrics"]["model_calls"], 5)
        self.assertEqual(len(tools.queries), 1)
        with self.assertRaises(ValueError):
            list(run.events(retry=True))

    def test_search_timeout_reuses_plan_on_retry(self):
        run, router, tools = fixture_run(search_failure=True)
        _, failure = execute(run)
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["retry_stage"], "search")
        _, result = execute(run, retry=True)
        self.assertEqual(result["status"], "complete", result["message"])
        self.assertEqual(len(tools.queries), 2)
        self.assertEqual(sum(schema is SearchPlan for schema, _, _ in router.calls), 1)

    def test_writer_rate_limit_retains_reader_evidence(self):
        run, _, tools = fixture_run(scenario="writer-limited")
        _, partial = execute(run)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["retry_stage"], "writer")
        self.assertTrue(partial["evidence"])
        events, result = execute(run, retry=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([event["stage"] for event in events if event["type"] == "stage_start"],
                         ["writer", "critic"])
        self.assertEqual(len(tools.queries), 1)

    def test_scrape_timeout_leaves_verified_snippet_evidence(self):
        run, _, _ = fixture_run(pages_timeout=True)
        _, result = execute(run)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evidence"][0]["kind"], "search snippet")
        self.assertEqual(result["metrics"]["pages_with_text"], 0)
        self.assertIn("The page could not be read before its deadline.", result["warnings"])

    def test_early_quota_retry_does_not_consume_saved_critic_retry(self):
        run, router, _ = fixture_run()
        route = router.invoke

        def limit_review(request, record_attempt, emit):
            if request.role == "critic":
                raise ModelError("rate_limit", retry_after=30)
            return route(request, record_attempt, emit)

        with patch.object(router, "invoke", side_effect=limit_review), patch("pipeline.monotonic", return_value=1_000):
            _, first = execute(run)
            events, blocked = execute(run, retry=True)
            self.assertEqual(first["report"], blocked["report"])
            self.assertEqual(blocked["retry_stage"], "critic")
            self.assertEqual([event["type"] for event in events], ["result"])
            self.assertEqual(run.retry_wait_seconds, 30)
            self.assertTrue(run.can_retry)
        with patch("pipeline.monotonic", return_value=1_031):
            events, result = execute(run, retry=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([event["stage"] for event in events if event["type"] == "stage_start"], ["critic"])
        self.assertEqual(router.calls[-1][2]["evidence_by_source"], {"S1": {"page": [PRIMARY]}})

    def test_invalid_citation_cannot_pass_even_when_critic_returns_no_findings(self):
        _, result = execute(fixture_run(scenario="unknown-citation")[0])
        self.assertEqual(result["status"], "partial")
        self.assertIn("S999", result["findings"][0]["detail"])


class EvidenceAndProviderTests(unittest.TestCase):
    def test_only_exact_excerpts_from_known_sources_are_retained(self):
        result = Reading.model_validate({
            "selections": [
                {"source_id": "S2", "facet": "storage", "excerpt": PRIMARY},
                {"source_id": "S1", "facet": "storage", "excerpt": FOLLOWUP},
                {"source_id": "S1", "facet": "storage", "excerpt": PRIMARY},
            ],
            "gaps": [],
        })
        passages = [{"source_id": "S1", "kind": "page", "text": PRIMARY}]
        evidence = select_evidence(result, passages, 6, 500, 250)
        self.assertEqual([(item["source_id"], item["text"]) for item in evidence], [("S1", PRIMARY)])

    def test_url_policy_blocks_private_dns_answers(self):
        result = [(None, None, None, "", ("203.0.113.1", 443))]
        with patch("tools.socket.getaddrinfo", return_value=result):
            with self.assertRaises(ToolError):
                _resolve_public("example.org", 443)
        with self.assertRaises(ToolError):
            public_url("https://example.org:8443")
        self.assertEqual(canonical_url("https://www.example.org/page?utm_source=x&q=y"),
                         canonical_url("http://example.org/page?q=y"))

    def test_connection_close_response_keeps_socket_available_during_body_read(self):
        class FixtureSocket:
            closed = False

            def makefile(self, mode):
                return BytesIO(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                               b"Connection: close\r\nContent-Length: 11\r\n\r\n<p>test</p>")

            def settimeout(self, seconds):
                if self.closed:
                    raise OSError("The reader tried to update a closed socket.")

            def sendall(self, data):
                pass

            def close(self):
                self.closed = True

        sock = FixtureSocket()
        with (
            patch("tools._resolve_public", return_value="8.8.8.8"),
            patch("tools.socket.create_connection", return_value=sock),
        ):
            body, encoding = _page_html("http://example.org/", monotonic() + 5)
        self.assertEqual((body, encoding), (b"<p>test</p>", "utf-8"))
        self.assertTrue(sock.closed)

    def test_groq_429_falls_back_to_configured_second_route(self):
        routes = [Route("groq", "openai/gpt-oss-20b"), Route("cloudflare", "@cf/openai/gpt-oss-20b")]
        request = ModelRequest("fast", "Write", {"question": TOPIC}, None, float("inf"), 100, 2)
        attempts = []
        with patch("providers._routes", return_value=routes), patch.dict("providers._COOLDOWN", clear=True), \
             patch.dict("providers._USAGE", clear=True), \
             patch("providers._model_call", side_effect=[ModelError("rate_limit", 10), "Draft"]):
            result = ModelRouter().invoke(request, attempts.append, lambda event: None)
        self.assertEqual(result, "Draft")
        self.assertEqual([(item["provider"], item["outcome"]) for item in attempts],
                         [("groq", "rate_limit"), ("cloudflare", "ok")])

    def test_deep_review_keeps_running_when_writer_consumed_its_groq_pool(self):
        env = {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "fixture"}
        request = ModelRequest("deep", "Review citations.", {"draft": "check evidence " * 450},
                               Review, float("inf"), 620, 3, role="critic")
        attempts = []
        with (
            patch.dict("os.environ", env, clear=True),
            patch.dict("providers._USAGE", clear=True),
            patch.dict("providers._COOLDOWN", clear=True),
            patch("providers._model_call", return_value=Review(summary="Reviewed", findings=[])),
        ):
            writer_route = _routes("deep", "writer")[0]
            _reserve(writer_route, 6_000)
            result = ModelRouter().invoke(request, attempts.append, lambda event: None)
        self.assertEqual(writer_route.model, "openai/gpt-oss-120b")
        self.assertEqual(result.summary, "Reviewed")
        self.assertEqual([attempt["model"] for attempt in attempts], [GROQ_REVIEW_MODEL])

    def test_deep_critic_timeout_uses_configured_groq_spare_model(self):
        env = {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "fixture"}
        request = ModelRequest("deep", "Review citations.", {}, Review, float("inf"), 620, 3, role="critic")
        attempts = []
        with (
            patch.dict("os.environ", env, clear=True),
            patch.dict("providers._USAGE", clear=True),
            patch.dict("providers._COOLDOWN", clear=True),
            patch("providers._model_call", side_effect=[ModelError("timeout"), Review(summary="Reviewed", findings=[])]),
        ):
            result = ModelRouter().invoke(request, attempts.append, lambda event: None)
        self.assertEqual(result.summary, "Reviewed")
        self.assertEqual([(item["model"], item["outcome"]) for item in attempts],
                         [(GROQ_REVIEW_MODEL, "timeout"), ("openai/gpt-oss-120b", "ok")])

    def test_qwen_critical_review_requests_a_strict_schema(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        model = Mock()
        model.bind.return_value.invoke.return_value = SimpleNamespace(
            content='{"summary":"Reviewed","findings":[]}', usage_metadata={"input_tokens": 30, "output_tokens": 10},
        )
        usage = {}
        with patch("providers._client", return_value=model):
            result = _model_call(Route("groq", GROQ_REVIEW_MODEL), [], Review, 10, 620,
                                 float("inf"), lambda event: None, usage)
        response_format = model.bind.call_args.kwargs["response_format"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(result.summary, "Reviewed")
        self.assertEqual(usage["input_tokens"], 30)

    def test_openrouter_paid_model_is_not_eligible(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "fixture",
                                       "OPENROUTER_MODEL": "meta-llama/llama-3.1-8b-instruct"}, clear=True):
            self.assertEqual(_routes("fast"), [])


class StreamlitTests(unittest.TestCase):
    def test_suggestion_submission_report_and_session_history(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("app.py", default_timeout=20).run()
        self.assertFalse(app.exception)
        app.button(key="suggest-Energy storage").click().run()
        self.assertIn("sodium-ion", app.text_area(key="relay_topic").value)
        with patch("pipeline.ResearchRun", return_value=fixture_run()[0]):
            next(button for button in app.button if button.label == "Start research").click().run()
        self.assertFalse(app.exception)
        self.assertIn("Research complete", [item.label for item in app.status])
        self.assertTrue(any("Grid storage and transport" in item.value for item in app.markdown))
        self.assertTrue(any("[S1](<https://example.org/storage>)" in item.value for item in app.markdown))
        self.assertEqual(len(app.session_state.relay_history), 1)


if __name__ == "__main__":
    unittest.main()
