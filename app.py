"""Streamlit interface for ResearchRelay."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
from time import monotonic

import streamlit as st

st.set_page_config(page_title="ResearchRelay", page_icon=":material/travel_explore:",
                   layout="centered", initial_sidebar_state="auto")

st.html("""
<style>
.relay-brand { display:flex; align-items:center; gap:.65rem; margin:0 0 2.5rem; }
.relay-mark { display:grid; place-items:center; width:2.25rem; height:2.25rem;
    border-radius:.7rem; background:#103b31; color:#e4f5db; font-size:1.4rem; }
.relay-name { font-weight:750; letter-spacing:-.035em; font-size:1.2rem; }
.relay-hero h1 { margin:0; padding:0; font-size:clamp(2.25rem, 6vw, 4.1rem);
    line-height:1.07; letter-spacing:-.07em; font-weight:760; }
.relay-hero p { margin:.8rem 0 1.6rem; color:#4a5b54; font-size:1.05rem; }
.st-key-relay_composer { margin-bottom:.65rem; }
.st-key-relay_report h1, .st-key-relay_report h2 { letter-spacing:-.045em; }
.st-key-relay_report p, .st-key-relay_report li { line-height:1.7; }
@media (max-width: 600px) {
    .relay-brand { margin-bottom:1.35rem; }
    .relay-hero p { font-size:.97rem; }
}
</style>
<div class="relay-brand">
    <span class="relay-mark" aria-hidden="true">↗</span>
    <span class="relay-name">ResearchRelay</span>
</div>
<div class="relay-hero">
    <h1>What would you like to research?</h1>
    <p>Start with a question and get a research report with sources you can inspect.</p>
</div>
""")

MODE_LABELS = {"fast": "Fast Research", "deep": "Deep Research"}
STAGE_LABELS = {
    "search": "Discovering sources",
    "reader": "Selecting evidence",
    "writer": "Drafting the report",
    "critic": "Reviewing claims and citations",
}
STAGE_NAMES = {"search": "Search", "reader": "Reader", "writer": "Writer", "critic": "Critic"}
SUGGESTIONS = {
    "Energy storage": "Where do sodium-ion batteries make sense compared with LFP for grid storage?",
    "Digital privacy": "How do the EU and India approach data portability, and where do they differ?",
    "Urban climate": "Which interventions have the strongest evidence for reducing urban heat?",
}
HISTORY_LIMIT = 12
_LOG = logging.getLogger(__name__)


def _initial_history() -> list[dict]:
    history = st.session_state.get("history", [])
    entries = []
    for index, entry in enumerate(reversed(history[-HISTORY_LIMIT:])):
        state = entry.get("state")
        if isinstance(state, dict):
            entries.append({
                **state, "id": f"legacy-{index}", "topic": entry.get("topic", "Research report"),
                "mode": "fast", "status": "complete", "created": entry.get("created"),
            })
    return entries


if "relay_history" not in st.session_state:
    st.session_state.relay_history = _initial_history()
if "relay_selected" not in st.session_state:
    entries = st.session_state.relay_history
    st.session_state.relay_selected = entries[0]["id"] if entries else ""


def _remember(result: dict, created: datetime) -> None:
    prior = next((item for item in st.session_state.relay_history if item["id"] == result["id"]), None)
    # The active graph owns the full checkpoint. History only needs report-facing fields.
    fields = ("id", "topic", "mode", "status", "message", "report", "review_summary",
              "feedback", "findings", "sources", "evidence", "warnings", "trace", "metrics",
              "elapsed", "can_retry", "retry_stage")
    summary = {key: result[key] for key in fields if key in result}
    summary["created"] = prior["created"] if prior else created
    remaining = [item for item in st.session_state.relay_history if item["id"] != result["id"]]
    st.session_state.relay_history = [summary, *remaining][:HISTORY_LIMIT]
    st.session_state.relay_selected = result["id"]


with st.sidebar:
    st.subheader("Research history")
    if not st.session_state.relay_history:
        st.caption("Your reports from this session will appear here.")
    for entry in st.session_state.relay_history:
        title = re.sub(r"[\[\]<>]", "", entry["topic"])
        if st.button(title[:75] or "Research question", key=f'open-{entry["id"]}', width="stretch",
                     help=MODE_LABELS.get(entry["mode"], "Research")):
            st.session_state.relay_selected = entry["id"]


def _choose_example(label: str) -> None:
    st.session_state.relay_topic = SUGGESTIONS[label]


with st.container(key="relay_composer", border=True):
    with st.form("research_request", border=False):
        topic = st.text_area("Research question", key="relay_topic", max_chars=600, height=105,
                             placeholder="Compare sodium-ion and LFP for grid storage")
        mode = st.segmented_control("Research mode", ["fast", "deep"], default="fast",
                                    format_func=MODE_LABELS.get)
        st.caption("Fast gives a focused, cited answer. Deep searches more broadly and can revisit gaps.")
        submitted = st.form_submit_button("Start research", type="primary", icon=":material/arrow_forward:")

st.caption("Try a question")
for label, column in zip(SUGGESTIONS, st.columns(len(SUGGESTIONS))):
    column.button(label, key=f"suggest-{label}", on_click=_choose_example,
                  args=(label,), width="stretch")


def _execute(run, retry: bool = False) -> dict:
    from agents import display_report

    created = datetime.now(timezone.utc)
    activity = st.status("Starting research…", state="running", expanded=True)
    activity_line = activity.empty()
    preview = activity.empty()
    draft_area = st.empty()
    draft_parts = []
    last_refresh = 0.0
    discovered = 0
    accepted = 0

    def show_activity(elapsed: float) -> None:
        activity_line.caption(
            f"{elapsed:.1f}s elapsed · {discovered} discovered · {accepted} with selected evidence"
        )

    for event in run.events(retry=retry):
        kind = event["type"]
        if kind == "result":
            result = event["result"]
            status = result["status"]
            title = {"complete": "Research complete", "partial": "Partial result", "failed": "Run interrupted"}
            activity.update(label=title.get(status, "Run ended"),
                            state="complete" if status == "complete" else "error", expanded=False)
            draft_area.empty()
            preview.empty()
            _remember(result, created)
            if run.can_retry:
                st.session_state.relay_retry_run = run
            else:
                st.session_state.pop("relay_retry_run", None)
            return result
        elapsed = event["elapsed"]
        if kind == "stage_start":
            activity.update(label=STAGE_LABELS[event["stage"]])
            show_activity(elapsed)
        elif kind == "source":
            discovered = event["count"]
            preview.text(event["source"]["title"])
            show_activity(elapsed)
        elif kind == "evidence":
            accepted = event["source_count"]
            show_activity(elapsed)
        elif kind == "routing":
            preview.caption(event["reason"])
        elif kind == "model_attempt" and event["outcome"] != "ok":
            preview.caption(f"{event['provider'].title()} · {event['outcome'].replace('_', ' ')}")
        elif kind == "draft_reset":
            draft_parts.clear()
            draft_area.empty()
        elif kind == "draft_delta":
            draft_parts.append(event["text"])
            if monotonic() - last_refresh >= 0.15:
                draft_area.markdown(display_report("".join(draft_parts)) + " ▍")
                show_activity(elapsed)
                last_refresh = monotonic()
        elif kind == "draft":
            draft_area.markdown(display_report(event["report"]))
            show_activity(elapsed)
    raise RuntimeError("The workflow ended without a result.")


if submitted:
    if not topic.strip():
        st.error("Enter a research question to get started.")
    elif mode is None:
        st.error("Select a research mode.")
    else:
        try:
            from pipeline import ResearchRun

            run = ResearchRun(topic, mode)
            st.session_state.pop("relay_retry_run", None)
            _execute(run)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            _LOG.error("Research launch failed (%s)", type(exc).__name__)
            st.error("Research could not start. Check the server setup and try again.")


def _render_sources(sources: list[dict], evidence: list[dict]) -> None:
    if not sources:
        return
    by_source: dict[str, list[dict]] = {}
    for excerpt in evidence:
        by_source.setdefault(excerpt["source_id"], []).append(excerpt)
    used, other = [], []
    for source in sources:
        (used if source["id"] in by_source else other).append(source)

    st.subheader("Sources")
    for source in used + other:
        source_id = source["id"]
        excerpts = by_source.get(source_id, [])
        label = f'{source_id} · {source["host"]}'
        with st.expander(label):
            st.text(source["title"])
            st.link_button("Open source", source["url"], icon=":material/open_in_new:")
            for excerpt in excerpts:
                st.caption("Full page excerpt" if excerpt["kind"] == "page" else "Search snippet")
                st.text(excerpt["text"])
            if not excerpts and source.get("snippet"):
                st.caption("Discovered source")
                st.text(source["snippet"])


def _render_trace(result: dict) -> None:
    trace = result.get("trace") or []
    if not trace:
        return
    metrics = result["metrics"]
    with st.expander("Run details"):
        st.caption(
            f'{metrics["model_calls"]} model attempts · {metrics["search_calls"]} search attempts · '
            f'{metrics["page_fetches"]} fetch attempts · '
            f'{metrics["search_cache_hits"] + metrics["page_cache_hits"]} cache hits'
        )
        st.caption(f'{metrics.get("pages_with_text", 0)} readable pages, including cached pages')
        for key, label in (("first_source_seconds", "First source"),
                           ("first_evidence_seconds", "Selected evidence"),
                           ("first_draft_seconds", "First complete draft")):
            if metrics[key] is not None:
                st.caption(f"{label} after {metrics[key]:.2f}s")
        for item in trace:
            stage = STAGE_NAMES.get(item["stage"], item["stage"])
            st.markdown(f'**{stage}** · {item["seconds"]:.2f}s · {item["outcome"]}')
            if item["detail"]:
                st.caption(item["detail"])
            for attempt in item["attempts"]:
                usage = f'~{attempt["estimated_input"]} estimated input tokens'
                if attempt["input_tokens"] is not None:
                    usage = f'{attempt["input_tokens"]} input / {attempt["output_tokens"]} output tokens'
                st.caption(f'{attempt["provider"]} · {attempt["model"]} · {attempt["outcome"]} · '
                           f'{attempt["seconds"]:.2f}s · {usage}')
        for warning in result.get("warnings", []):
            st.caption(warning)


def _render_result(result: dict) -> None:
    from agents import display_report

    st.divider()
    created = result.get("created")
    details = [MODE_LABELS[result["mode"]]]
    if isinstance(created, datetime):
        details.append(created.strftime("%b %d, %Y"))
    if result.get("elapsed") is not None:
        details.append(f'{result["elapsed"]:.1f}s total')
    st.caption(" · ".join(details))
    st.header(display_report(result["topic"]))
    if result.get("message"):
        notice = st.error if result["status"] == "failed" else st.warning
        notice(result["message"])
    report = result.get("report") or ""
    sources = result.get("sources") or []
    evidence = result.get("evidence") or []
    if report:
        evidence_ids = {item["source_id"] for item in evidence}
        source_links = {source["id"]: source["url"] for source in sources if source["id"] in evidence_ids}

        def citation_link(match: re.Match) -> str:
            source_id = match[1].upper()
            source_url = source_links.get(source_id)
            return f"[{source_id}](<{source_url}>)" if source_url else match[0]

        export = re.sub(r"\[(S\d+)\]", citation_link, display_report(report), flags=re.IGNORECASE)
        with st.container(key="relay_report"):
            st.markdown(export)
        cited_ids = {source_id.upper() for source_id in re.findall(r"\[(S\d+)\]", report, re.IGNORECASE)}
        citations = [
            f'- {source["id"]} · {display_report(source["title"])} · <{source["url"]}>'
            for source in sources if source["id"] in cited_ids and source["id"] in evidence_ids
        ]
        if citations:
            export += "\n\n## Sources\n\n" + "\n".join(citations)
        download, copy = st.columns(2)
        download.download_button("Download Markdown", export, file_name="researchrelay-report.md",
                                 mime="text/markdown", on_click="ignore", icon=":material/download:")
        with copy.popover("Copy Markdown"):
            st.code(export, language=None, wrap_lines=True)
    feedback = result.get("review_summary") or result.get("feedback")
    findings = result.get("findings", [])
    if feedback or findings:
        with st.expander("Critic review", expanded=result["status"] == "partial"):
            if feedback:
                st.markdown(display_report(feedback))
            for issue in findings:
                label = issue["kind"].replace("_", " ").capitalize()
                st.markdown(f'**{label}** · {display_report(issue["claim"] or issue["detail"])}')
                if issue["claim"] and issue["detail"]:
                    st.caption(display_report(issue["detail"]))
    _render_sources(sources, evidence)
    if not sources and result.get("search_results"):
        with st.expander("Search results"):
            st.text(result["search_results"])
    _render_trace(result)


selected = next((entry for entry in st.session_state.relay_history
                 if entry["id"] == st.session_state.relay_selected), None)
if selected is not None:
    active = st.session_state.get("relay_retry_run")
    if active is not None and active.id == selected["id"] and active.can_retry:
        stage = STAGE_NAMES.get(selected.get("retry_stage"), "this stage")
        wait = getattr(active, "retry_wait_seconds", 0)
        if wait:
            st.caption(f"The saved {stage} stage can be retried in about {wait} seconds.")
        if st.button(f"Retry {stage}", icon=":material/refresh:"):
            if wait:
                st.info("The retry is still available once the allowance resets.")
            else:
                try:
                    _execute(active, retry=True)
                    selected = st.session_state.relay_history[0]
                except Exception as exc:
                    _LOG.error("Research retry failed (%s)", type(exc).__name__)
                    st.error("The saved stage could not be restarted.")
    _render_result(selected)
