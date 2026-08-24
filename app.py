"""Streamlit UI for the multi-agent research pipeline.

Run from the project root with:
    python -m streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from pipeline import run_research_pipeline


st.set_page_config(
    page_title="Research Pipeline",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#e9edf5; --muted:#8d98ab; --line:#293244; --panel:#131b29; --panel2:#192336; --cyan:#52d6d0; --orange:#ffad6b; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(900px 500px at 10% -10%, #163b4c 0%, transparent 62%), radial-gradient(850px 460px at 100% 0%, #4a2b27 0%, transparent 60%), #0b111b; color:var(--ink); }
    section[data-testid="stSidebar"] { background:#101824; border-right:1px solid var(--line); }
    .hero { padding:2.5rem 2.7rem 2.25rem; margin-bottom:1.5rem; border:1px solid var(--line); border-radius:10px; background:linear-gradient(110deg, rgba(24,47,63,.96), rgba(29,29,43,.92)); }
    .eyebrow { color:var(--cyan); font-size:.75rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }
    h1, h2, h3, .brand { font-family:'Space Grotesk', sans-serif !important; }
    .hero h1 { color:var(--ink); font-size:clamp(2rem, 4vw, 3.6rem); line-height:1.05; margin:.45rem 0 .8rem; max-width:720px; }
    .hero p { color:var(--muted); max-width:690px; font-size:1rem; line-height:1.6; margin:0; }
    .track { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.5rem; }
    .chip { border:1px solid var(--line); background:rgba(11,17,27,.45); color:#c5cfdd; padding:.5rem .75rem; border-radius:6px; font-size:.78rem; }
    .card { border:1px solid var(--line); background:rgba(19,27,41,.9); border-radius:8px; padding:1.25rem 1.4rem; }
    .stTextInput input { background:var(--panel2)!important; color:var(--ink)!important; border:1px solid var(--line)!important; border-radius:6px!important; }
    .stButton > button, .stFormSubmitButton > button { border-radius:6px!important; border:1px solid #58b9b5!important; background:#2e827f!important; color:white!important; font-weight:700!important; transition:transform .15s, background .15s!important; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { transform:translateY(-1px); background:#3a9b96!important; }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color:var(--muted); padding:.7rem 1rem; }
    .stTabs [aria-selected="true"] { color:var(--cyan)!important; }
    .metric { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; }
    .scroll { max-height:420px; overflow:auto; white-space:pre-wrap; color:#cbd5e1; line-height:1.65; }
    </style>
    """,
    unsafe_allow_html=True,
)


def as_text(value: object, empty: str = "No content captured.") -> str:
    text = value if isinstance(value, str) else str(value or "")
    return text.strip() or empty


def render_results(topic: str, state: dict) -> None:
    report = as_text(state.get("report"))
    feedback = as_text(state.get("feedback"))
    search = as_text(state.get("search_results"))
    scraped = as_text(state.get("scraped_content"))
    st.markdown(f'<div class="metric">Reviewed research / {topic}</div>', unsafe_allow_html=True)
    st.markdown("### Research output")

    report_tab, critic_tab, search_tab, reader_tab = st.tabs(
        ["Final report", "Critic feedback", "Search results", "Scraped content"]
    )
    with report_tab:
        st.markdown(report)
        st.download_button(
            "Download report (.md)", report,
            file_name=f"{topic[:40].strip().replace(' ', '_') or 'report'}.md",
            mime="text/markdown", key="download-report",
        )
    with critic_tab:
        st.markdown(feedback)
    with search_tab:
        st.markdown(f'<div class="card scroll">{search}</div>', unsafe_allow_html=True)
    with reader_tab:
        st.markdown(f'<div class="card scroll">{scraped}</div>', unsafe_allow_html=True)


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown('<div class="brand">🔎 Research Pipeline</div>', unsafe_allow_html=True)
    st.caption("Search → Read → Write → Critique")
    st.divider()
    st.markdown("**Run history**")
    if not st.session_state.history:
        st.caption("No runs yet.")
    for index, item in enumerate(reversed(st.session_state.history)):
        original_index = len(st.session_state.history) - index - 1
        label = item["topic"][:34] + ("…" if len(item["topic"]) > 34 else "")
        if st.button(f"📄  {label}", key=f"history-{original_index}", use_container_width=True):
            st.session_state.selected_run = original_index
            st.rerun()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.pop("selected_run", None)
        st.rerun()

st.markdown(
    '<div class="hero"><div class="eyebrow">Multi-agent research system</div>'
    '<h1>Turn a question into a reviewed report.</h1>'
    '<p>Search reliable sources, read the strongest result, draft a structured report, and let a critic challenge the outcome.</p>'
    '<div class="track"><span class="chip">01 · Search agent</span><span class="chip">02 · Reader agent</span>'
    '<span class="chip">03 · Writer chain</span><span class="chip">04 · Critic review</span></div></div>',
    unsafe_allow_html=True,
)

with st.form("research-form"):
    left, right = st.columns([4, 1])
    with left:
        topic = st.text_input("Research topic", placeholder="e.g. Latest developments in solid-state batteries")
    with right:
        st.write("")
        submitted = st.form_submit_button("Run pipeline", use_container_width=True)

if submitted:
    if not topic.strip():
        st.error("Please enter a research topic before running the pipeline.")
    else:
        try:
            with st.status("Running the research pipeline", expanded=True) as status:
                st.write("🔍 Step 1/4 — Search agent gathering reliable sources")
                st.write("📖 Step 2/4 — Reader agent extracting the strongest source")
                st.write("✍️ Step 3/4 — Writer drafting the report")
                st.write("🧐 Step 4/4 — Critic reviewing the draft")
                result = run_research_pipeline(topic.strip())
                status.update(label="Pipeline complete", state="complete", expanded=False)
            st.session_state.history.append({"topic": topic.strip(), "state": result, "created": datetime.now()})
            st.session_state.selected_run = len(st.session_state.history) - 1
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
        except Exception as exc:
            st.error(f"Research pipeline failed: {exc}")

selected = st.session_state.get("selected_run")
if selected is not None and 0 <= selected < len(st.session_state.history):
    selected_run = st.session_state.history[selected]
    render_results(selected_run["topic"], selected_run["state"])
else:
    st.markdown('<div class="card"><span class="metric">Ready when you are</span><p>Run a topic above to see the report, source trail, and critic feedback.</p></div>', unsafe_allow_html=True)