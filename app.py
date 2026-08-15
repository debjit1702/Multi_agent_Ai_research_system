"""
Streamlit UI for the multi-agent research pipeline.

Run with:
    streamlit run app.py
"""

import streamlit as st


def _load_pipeline_runner():
    """Import the pipeline lazily so UI always loads first."""
    try:
        from pipeline import run_research_pipeline as _runner
        return _runner, None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


def _render_state(topic: str, state: dict) -> None:
    st.subheader(f"Results for: {topic}")
    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["Report", "Critic Feedback", "Search Results", "Scraped Content"]
    )

    with tab_report:
        report = state.get("report", "")
        report_text = report if isinstance(report, str) else str(report)
        st.markdown(report_text)
        st.download_button(
            "Download report (.md)",
            data=report_text,
            file_name=f"{topic[:40].strip().replace(' ', '_') or 'report'}.md",
            mime="text/markdown",
        )

    with tab_feedback:
        feedback = state.get("feedback", "")
        st.markdown(feedback if isinstance(feedback, str) else str(feedback))

    with tab_search:
        st.text(state.get("search_results", "No search results captured."))

    with tab_scraped:
        st.text(state.get("scraped_content", "No scraped content captured."))


st.set_page_config(page_title="Multi-Agent Research Pipeline", page_icon="🔎", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "selected_run" not in st.session_state:
    st.session_state.selected_run = None

with st.sidebar:
    st.title("Research Pipeline")
    st.caption("Search agent -> Reader agent -> Writer -> Critic")

    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.selected_run = None
        st.rerun()

    history_labels = [run["topic"] for run in st.session_state.history]
    if history_labels:
        selected_label = st.selectbox(
            "Run history",
            options=list(reversed(history_labels)),
            index=0,
        )
        st.session_state.selected_run = history_labels.index(selected_label)
    else:
        st.caption("No runs yet.")

st.title("Multi-Agent Research Pipeline")
st.write(
    "Enter a topic and the pipeline will search the web, scrape the most "
    "relevant source, draft a report, and have a critic review it."
)

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Latest developments in solid-state batteries",
    )
    submitted = st.form_submit_button("Run pipeline", type="primary")

if submitted:
    if not topic or not topic.strip():
        st.error("Please enter a research topic before running the pipeline.")
    else:
        run_research_pipeline, pipeline_load_error = _load_pipeline_runner()
        if run_research_pipeline is None:
            st.error(
                "Pipeline backend could not be loaded. "
                f"Details: {pipeline_load_error}"
            )
            st.stop()

        try:
            with st.spinner("Running research pipeline..."):
                result_state = run_research_pipeline(topic.strip())

            st.session_state.history.append(
                {"topic": topic.strip(), "state": result_state}
            )
            st.session_state.selected_run = len(st.session_state.history) - 1
            st.success("Pipeline complete.")
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
        except Exception as exc:
            st.error(f"Research pipeline failed: {exc}")

selected_idx = st.session_state.get("selected_run")
if (
    selected_idx is not None
    and 0 <= selected_idx < len(st.session_state.history)
):
    run = st.session_state.history[selected_idx]
    st.markdown("---")
    _render_state(run["topic"], run["state"])
