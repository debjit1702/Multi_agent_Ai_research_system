"""Checkpointed LangGraph orchestration for ResearchRelay."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import ceil
from threading import BoundedSemaphore
from time import monotonic
from typing import Callable, Iterator, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from agents import AgentContext, Finding, ResearchAgents, citation_issues, display_report, select_evidence
from providers import Attempt, Event, ModelError, Tier
from tools import (
    Batch, Candidate, Evidence, Passage, ResearchTools, Source, ToolError, canonical_url,
    clean_text, estimate_tokens, rank_passages, rank_sources, trim_tokens,
)

MAX_TOPIC = 600
STAGES = ("search", "reader", "writer", "critic")
_RUN_SLOTS = BoundedSemaphore(3)
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Policy:
    queries: int
    results_per_query: int
    sources: int
    pages: int
    reader_tokens: int
    evidence_items: int
    evidence_tokens: int
    attempts: int
    seconds: int


POLICIES = {
    "fast": Policy(2, 4, 8, 3, 1_900, 6, 2_500, 7, 50),
    "deep": Policy(4, 5, 16, 8, 3_500, 12, 4_500, 13, 105),
}


class Metrics(TypedDict):
    model_calls: int
    search_calls: int
    page_fetches: int
    search_cache_hits: int
    page_cache_hits: int
    pages_with_text: int
    first_source_seconds: float | None
    first_evidence_seconds: float | None
    first_draft_seconds: float | None


class TraceEntry(TypedDict):
    stage: str
    seconds: float
    outcome: str
    detail: str
    attempts: list[Attempt]


class ResearchState(TypedDict):
    topic: str
    mode: Tier
    entry: str
    resume_from: str
    status: str
    retry_stage: str
    retry_not_before: float
    message: str
    next_stage: str
    sources: list[Source]
    new_source_ids: list[str]
    used_queries: list[str]
    pending_queries: list[str]
    facets: list[str]
    passages: list[Passage]
    fetched_ids: list[str]
    evidence: list[Evidence]
    gaps: list[str]
    report: str
    review_summary: str
    findings: list[dict]
    followups: int
    gap: str
    revision: str
    warnings: list[str]
    trace: list[TraceEntry]
    metrics: Metrics


@dataclass
class Stage:
    ctx: AgentContext
    metrics: Metrics
    warnings: list[str]
    emit: Event
    changes: dict
    detail: str = ""

    def warn(self, *messages: str) -> None:
        for message in messages:
            if message not in self.warnings:
                self.warnings.append(message)


def validate_topic(topic: str) -> str:
    if not isinstance(topic, str):
        raise ValueError("Enter a research question.")
    topic = clean_text(topic)
    if not topic:
        raise ValueError("Enter a research question.")
    if len(topic) > MAX_TOPIC:
        raise ValueError(f"Keep the research question to {MAX_TOPIC} characters or fewer.")
    return topic


def _bounded_strings(items: list[str], count: int, limit: int) -> list[str]:
    unique = {}
    for item in items[:count * 4]:
        text = trim_tokens(clean_text(item), limit)
        if text:
            unique.setdefault(text.casefold(), text)
    return list(unique.values())[:count]


def _route_after(stage: str) -> Callable[[ResearchState], str]:
    def route(state: ResearchState) -> str:
        return stage if state["status"] == "running" else END
    return route


class ResearchRun:
    """Own one graph thread, its retryable checkpoint, and its session-local search cache."""

    def __init__(self, topic: str, mode: Tier = "fast",
                 agents: ResearchAgents | None = None, tools: ResearchTools | None = None) -> None:
        topic = validate_topic(topic)
        if mode not in POLICIES:
            raise ValueError("Choose Fast Research or Deep Research.")
        load_dotenv()
        self.id = uuid4().hex
        self.policy = POLICIES[mode]
        self._agents = agents if agents is not None else ResearchAgents()
        self._tools = tools if tools is not None else ResearchTools()
        self._deadline = 0.0
        self._elapsed_before_attempt = 0.0
        self._started_at = 0.0
        self._retry_count = 0
        self._invoked = False
        self._config = {"configurable": {"thread_id": self.id}, "recursion_limit": 24}
        self._state: ResearchState = {
            "topic": topic, "mode": mode, "entry": "search", "resume_from": "", "status": "running",
            "retry_stage": "", "retry_not_before": 0.0, "message": "", "next_stage": "",
            "sources": [], "new_source_ids": [], "used_queries": [], "pending_queries": [], "facets": [],
            "passages": [], "fetched_ids": [], "evidence": [], "gaps": [], "report": "",
            "review_summary": "", "findings": [], "followups": 0, "gap": "", "revision": "",
            "warnings": [], "trace": [],
            "metrics": {
                "model_calls": 0, "search_calls": 0, "page_fetches": 0,
                "search_cache_hits": 0, "page_cache_hits": 0, "pages_with_text": 0, "first_source_seconds": None,
                "first_evidence_seconds": None, "first_draft_seconds": None,
            },
        }
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ResearchState)
        handlers = (self._search, self._read, self._write, self._critique)
        for stage, handler in zip(STAGES, handlers):
            builder.add_node(stage, self._node(stage, handler))
        builder.add_conditional_edges(START, lambda state: state["entry"], {stage: stage for stage in STAGES})
        for current, following in zip(STAGES, STAGES[1:]):
            builder.add_conditional_edges(current, _route_after(following), {following: following, END: END})
        builder.add_conditional_edges("critic", lambda state: state["next_stage"] or END,
                                      {"search": "search", "writer": "writer", END: END})
        return builder.compile(checkpointer=InMemorySaver())

    def _elapsed(self) -> float:
        return self._elapsed_before_attempt + monotonic() - self._started_at

    def _node(self, name: str, handler: Callable[[ResearchState, Stage], dict]):
        def node(state: ResearchState) -> dict:
            writer = get_stream_writer()
            metrics = state["metrics"].copy()
            attempts: list[Attempt] = []
            started = monotonic()
            detail = ""

            def emit(event: dict) -> None:
                writer({"stage": name, "elapsed": round(self._elapsed(), 3), **event})

            def record(attempt: Attempt) -> None:
                attempts.append(attempt)
                metrics["model_calls"] += 1
                emit({"type": "model_attempt", **attempt})

            def remaining_attempts() -> int:
                return self.policy.attempts - metrics["model_calls"]

            stage_deadline = self._deadline
            if name == "writer":
                stage_deadline -= 10 if state["mode"] == "deep" else 6
            working = Stage(
                AgentContext(stage_deadline, remaining_attempts, record, emit),
                metrics, list(state["warnings"]), emit, {},
            )
            emit({"type": "stage_start"})
            try:
                if monotonic() >= self._deadline:
                    raise ModelError("budget")
                working.changes.update(handler(state, working))
                outcome = working.changes.get("status", "complete")
            except (ToolError, ModelError) as exc:
                detail = str(exc)
                useful = working.changes.get("sources", state["sources"]) or state["report"]
                outcome = "partial" if useful else "failed"
                working.changes.update(
                    status=outcome, message=detail, next_stage="",
                    retry_stage=name if remaining_attempts() > 0 else "",
                )
                if isinstance(exc, ModelError) and exc.kind == "rate_limit" and exc.retry_after > 0:
                    working.changes["retry_not_before"] = monotonic() + exc.retry_after
                if name == "critic" and state["report"]:
                    working.changes["review_summary"] = "The draft is available. The Critic review was interrupted."
            detail = detail or working.detail or working.changes.get("message", "")
            if outcome == "running":
                outcome = "routed"
            trace = TraceEntry(stage=name, seconds=round(monotonic() - started, 3),
                               outcome=outcome, detail=detail, attempts=attempts)
            emit({"type": "stage_end", "outcome": outcome, "detail": detail})
            return {**working.changes, "resume_from": "", "metrics": metrics, "warnings": working.warnings,
                    "trace": [*state["trace"], trace]}
        return node

    def _track_batch(self, stage: Stage, batch: Batch, kind: str) -> None:
        calls, cache_hits = {
            "search": ("search_calls", "search_cache_hits"),
            "reader": ("page_fetches", "page_cache_hits"),
        }[kind]
        stage.metrics[calls] += batch.attempted
        stage.metrics[cache_hits] += batch.cached
        stage.warn(*batch.errors)

    def _search(self, state: ResearchState, stage: Stage) -> dict:
        self._tools.check_search_configuration()
        sources = list(state["sources"])
        by_url = {canonical_url(source["url"]): index for index, source in enumerate(sources)}
        facets = list(state["facets"])
        used_queries = list(state["used_queries"])
        query_budget = self.policy.queries - stage.metrics["search_calls"] + self._retry_count
        queries = state["pending_queries"][:max(0, query_budget)]
        if not queries:
            reserve = 1 if state["mode"] == "deep" and not state["followups"] else 0
            cap = max(0, query_budget - reserve)
            if not cap:
                raise ToolError("The run has used its search budget.")
            plan = self._agents.search(stage.ctx, state["topic"], cap, state["gap"])
            seen = {query.casefold() for query in used_queries}
            offered = _bounded_strings(plan.queries, cap * 3, 65)
            queries = [query for query in offered if query.casefold() not in seen][:cap]
            if not queries:
                raise ToolError("The Search agent could not produce another distinct query.")
            facets = _bounded_strings([*facets, *plan.facets], 8, 28)
        stage.changes.update(facets=facets, pending_queries=queries)

        def add_source(candidate: Candidate) -> None:
            identity = canonical_url(candidate["url"])
            index = by_url.get(identity)
            if index is not None:
                if candidate["score"] > sources[index]["score"]:
                    original = sources[index]
                    sources[index] = Source(
                        **candidate, id=original["id"],
                    ) if candidate["url"] == original["url"] else {
                        **original, "score": candidate["score"],
                    }
                return
            if len(sources) >= self.policy.sources:
                return
            source = Source(**candidate, id=f"S{len(sources) + 1}")
            by_url[identity] = len(sources)
            sources.append(source)
            if stage.metrics["first_source_seconds"] is None:
                stage.metrics["first_source_seconds"] = round(self._elapsed(), 3)
            stage.emit({"type": "source", "source": source, "count": len(sources)})

        deadline = min(self._deadline, monotonic() + (13 if state["mode"] == "deep" else 9))
        previous_count = len(sources)
        batch = self._tools.search(queries, self.policy.results_per_query, deadline, add_source)
        self._track_batch(stage, batch, "search")
        stage.changes.update(sources=sources, new_source_ids=[source["id"] for source in sources[previous_count:]])
        if not sources or (state["gap"] and len(sources) == previous_count):
            raise ToolError(batch.errors[0] if batch.errors else "No additional relevant sources were found.")
        return {"used_queries": list(dict.fromkeys([*used_queries, *queries])), "pending_queries": []}

    def _read(self, state: ResearchState, stage: Stage) -> dict:
        ranked = rank_sources(state["sources"], state["gap"] or state["topic"])
        if state["gap"]:
            fresh_ids = set(state["new_source_ids"])
            ranked.sort(key=lambda source: source["id"] not in fresh_ids)
        selected_count = 4 if state["mode"] == "fast" else (8 if state["followups"] else 6)
        selected = ranked[:selected_count]
        fetched = set(state["fetched_ids"])
        # A manual retry can rebuild a failed model judgment from the checkpointed passages.
        previous_passages = list(state["passages"]) if state["resume_from"] == "reader" else []
        pages_left = self.policy.pages - len(fetched)
        if state["mode"] == "deep" and not state["followups"]:
            pages_left -= 2
        pages = {}
        to_fetch = [source for source in selected if source["id"] not in fetched][:max(0, pages_left)]
        if state["mode"] == "fast" and to_fetch:
            # A full page anchors the evidence. Spend additional fetches on thin snippets.
            to_fetch = [to_fetch[0], *(item for item in to_fetch[1:] if len(item["snippet"]) < 320)]
        if to_fetch and not previous_passages:
            deadline = min(self._deadline, monotonic() + (13 if state["mode"] == "deep" else 8))

            def receive(source_id: str, text: str) -> None:
                pages[source_id] = text
                stage.metrics["pages_with_text"] += 1
                stage.emit({"type": "page", "source_id": source_id})

            batch = self._tools.read(to_fetch, deadline, receive)
            self._track_batch(stage, batch, "reader")
            fetched.update(source["id"] for source in to_fetch[:batch.attempted + batch.cached])

        passages = previous_passages
        if not passages:
            remaining = self.policy.reader_tokens
            per_source = 430 if state["mode"] == "fast" else 480
            for source in selected:
                current = rank_passages(
                    source, pages.get(source["id"], ""), state["gap"] or state["topic"],
                    min(remaining, per_source),
                )
                passages.extend(current)
                remaining -= sum(estimate_tokens(item["text"]) + 12 for item in current)
            # Retain previous excerpts during a Deep follow-up.
            passage_ids = {item["source_id"] for item in passages}
            for item in state["evidence"]:
                cost = estimate_tokens(item["text"]) + 12
                if item["source_id"] not in passage_ids and cost <= remaining:
                    passages.append(Passage(source_id=item["source_id"], kind=item["kind"], text=item["text"]))
                    remaining -= cost
        if not passages:
            raise ToolError("The sources did not provide readable evidence.")
        stage.changes.update(passages=passages, fetched_ids=sorted(fetched))
        passage_ids = {item["source_id"] for item in passages}
        visible_sources = [source for source in state["sources"] if source["id"] in passage_ids]
        result = self._agents.read(stage.ctx, state["topic"], state["facets"], visible_sources,
                                   passages, self.policy.evidence_items)
        evidence = select_evidence(result, passages, self.policy.evidence_items,
                                   self.policy.evidence_tokens, 580)
        if state["followups"] and state["gap"]:
            # A follow-up adds evidence. Keep useful excerpts from the first pass within the same budget.
            existing = {(item["source_id"], item["text"]) for item in evidence}
            remaining = self.policy.evidence_tokens - sum(estimate_tokens(item["text"]) + 32 for item in evidence)
            for item in state["evidence"]:
                cost = estimate_tokens(item["text"]) + 32
                per_source = sum(estimate_tokens(stored["text"]) + 32 for stored in evidence
                                 if stored["source_id"] == item["source_id"])
                if ((item["source_id"], item["text"]) not in existing and cost <= remaining
                        and per_source + cost <= 580 and len(evidence) < self.policy.evidence_items):
                    evidence.append(item)
                    remaining -= cost
        if not evidence:
            raise ToolError("The Reader could not verify excerpts against the retrieved passages.")
        if stage.metrics["first_evidence_seconds"] is None:
            stage.metrics["first_evidence_seconds"] = round(self._elapsed(), 3)
        stage.emit({"type": "evidence", "source_count": len({item["source_id"] for item in evidence}),
                    "passage_count": len(evidence)})
        return {"evidence": evidence, "gaps": _bounded_strings(result.gaps, 6, 45)}

    def _write(self, state: ResearchState, stage: Stage) -> dict:
        prior = trim_tokens(state["report"], 1_500) if state["revision"] else ""
        report = self._agents.write(stage.ctx, state["mode"], state["topic"], state["evidence"],
                                    state["gaps"], prior, state["revision"])
        if not report.strip():
            raise ModelError("invalid_response")
        if prior and citation_issues(report, state["evidence"]) and not citation_issues(prior, state["evidence"]):
            raise ToolError("The targeted revision introduced citation problems. The earlier draft is available.")
        if stage.metrics["first_draft_seconds"] is None:
            stage.metrics["first_draft_seconds"] = round(self._elapsed(), 3)
        stage.emit({"type": "draft", "report": report})
        return {"report": report, "review_summary": "", "findings": []}

    def _critique(self, state: ResearchState, stage: Stage) -> dict:
        deterministic = citation_issues(state["report"], state["evidence"])
        # Store deterministic findings even if the model review later fails.
        stage.changes["findings"] = [item.model_dump() for item in deterministic]
        result = self._agents.critique(
            stage.ctx, state["mode"], state["topic"], state["report"], state["evidence"], state["gaps"],
        )
        valid_ids = {item["source_id"] for item in state["evidence"]}
        findings = deterministic + [
            Finding(kind=item.kind, claim=trim_tokens(clean_text(item.claim), 50),
                    detail=trim_tokens(clean_text(item.detail), 80),
                    source_ids=[value for value in item.source_ids if value in valid_ids][:5])
            for item in result.findings[:8]
        ]
        coverage = next((item for item in findings if item.kind == "coverage"), None)
        message = "Some claims or coverage gaps need further review." if findings else ""
        update = {
            "review_summary": trim_tokens(clean_text(result.summary), 150),
            "findings": [item.model_dump() for item in findings],
            "status": "partial" if findings else "complete", "message": message,
            "next_stage": "", "retry_stage": "",
        }
        can_follow_up = (
            state["mode"] == "deep" and not state["followups"] and findings
            and self._deadline - monotonic() >= 8
        )
        if can_follow_up and coverage and stage.ctx.attempts() >= 4 and stage.metrics["search_calls"] < self.policy.queries:
            gap = trim_tokens(coverage.claim + " " + coverage.detail, 90)
            stage.detail = "The Critic identified an evidence gap."
            stage.emit({"type": "routing", "target": "search", "reason": stage.detail})
            return {**update, "status": "running", "message": "", "next_stage": "search",
                    "followups": 1, "gap": gap, "revision": gap, "pending_queries": []}
        repair = [item for item in findings if item.kind != "coverage"]
        if can_follow_up and repair and stage.ctx.attempts() >= 2:
            revision = "; ".join(f"{item.claim} {item.detail}".strip() for item in repair[:3])
            stage.detail = "The Critic requested a targeted revision."
            stage.emit({"type": "routing", "target": "writer", "reason": stage.detail})
            return {**update, "status": "running", "message": "", "next_stage": "writer",
                    "followups": 1, "revision": revision}
        return update

    @property
    def can_retry(self) -> bool:
        return self._retry_count == 0 and bool(self._state["retry_stage"])

    @property
    def retry_wait_seconds(self) -> int:
        return max(0, ceil(self._state.get("retry_not_before", 0.0) - monotonic()))

    def _result(self) -> dict:
        result = dict(self._state)
        result["id"] = self.id
        result["elapsed"] = round(self._elapsed_before_attempt, 3)
        result["can_retry"] = self.can_retry
        result["search_results"] = "\n".join(
            f'{source["title"]} — {source["url"]}' for source in result["sources"]
        )
        result["scraped_content"] = "\n".join(item["text"] for item in result["evidence"])
        result["feedback"] = result["review_summary"]
        return result

    def events(self, retry: bool = False) -> Iterator[dict]:
        if retry and not self.can_retry:
            raise ValueError("This run has no retryable checkpoint.")
        if retry and self.retry_wait_seconds:
            yield {"type": "result", "result": self._result()}
            return
        if not retry and self._invoked:
            raise ValueError("This run has already started.")
        if not _RUN_SLOTS.acquire(blocking=False):
            self._state.update(
                status="partial" if self._state["sources"] else "failed",
                message="ResearchRelay is busy. Retry this stage in a moment.",
                retry_stage=self._state["retry_stage"] or self._state["entry"],
            )
            yield {"type": "result", "result": self._result()}
            return
        self._started_at = monotonic()
        self._deadline = self._started_at + self.policy.seconds
        start_state: dict = dict(self._state)
        if retry:
            self._retry_count += 1
            start_state.update(
                entry=self._state["retry_stage"], resume_from=self._state["retry_stage"],
                status="running", retry_stage="", retry_not_before=0.0, message="", next_stage="",
            )
        try:
            self._invoked = True
            for kind, payload in self._graph.stream(start_state, self._config, stream_mode=["custom", "values"]):
                if kind == "custom":
                    yield payload
                else:
                    self._state = payload
        except Exception as exc:
            _LOG.error("Graph execution failed (%s)", type(exc).__name__)
            self._state.update(status="partial" if self._state["sources"] or self._state["report"] else "failed",
                               message="The research workflow was interrupted. Start a new run.", retry_stage="")
        finally:
            self._elapsed_before_attempt = self._elapsed()
            _RUN_SLOTS.release()
        yield {"type": "result", "result": self._result()}


def run_research_pipeline(topic: str, mode: Tier = "fast") -> dict:
    run = ResearchRun(topic, mode)
    for event in run.events():
        if event["type"] == "result":
            return event["result"]
    raise RuntimeError("The research graph did not return a result.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the ResearchRelay graph.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--mode", choices=POLICIES, default="fast")
    args = parser.parse_args()
    try:
        result = run_research_pipeline(args.question or input("Research question: "), args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    print(display_report(result["report"]) or result["message"] or "The run did not produce a report.")
    if result["report"] and result["message"]:
        print("\nRun status:", result["message"])
    if result["search_results"]:
        print("\nSources\n" + result["search_results"])
