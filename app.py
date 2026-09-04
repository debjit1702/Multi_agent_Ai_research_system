"""Agentic Research Studio — Multi-Agent Research Pipeline UI

Run from the project root with:
    python -m streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from pipeline import run_research_pipeline


st.set_page_config(
    page_title="Agentic Research Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --bg:           #030712;
        --bg-nav:       #0b0f1a;
        --surface:      #0d1526;
        --surface-2:    #111827;
        --border:       #1e2d4a;
        --border-2:     #243451;
        --blue:         #3b82f6;
        --blue-dark:    #2563eb;
        --blue-glow:    rgba(59,130,246,0.25);
        --cyan:         #06b6d4;
        --green:        #10b981;
        --green-dim:    rgba(16,185,129,0.15);
        --red:          #ef4444;
        --text:         #f1f5f9;
        --text-muted:   #94a3b8;
        --text-dim:     #475569;
        --radius:       12px;
        --radius-lg:    16px;
        --shadow:       0 4px 24px rgba(0,0,0,0.6);
    }

    html, body { background: var(--bg) !important; }

    #MainMenu, footer, header, .stDeployButton,
    [data-testid="collapsedControl"] { display: none !important; }

    .stApp {
        background: var(--bg) !important;
        font-family: 'Inter', sans-serif;
        color: var(--text);
        min-height: 100vh;
    }

    .main > div:first-child { padding-top: 4.5rem !important; }
    [data-testid="stVerticalBlock"] { gap: 0 !important; }
    [data-testid="stHorizontalBlock"] { gap: 0 !important; }

    /* ── Navbar ── */
    .navbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 1000;
        height: 56px;
        background: var(--bg-nav);
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 1.5rem;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.75rem; }
    .nav-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 0 12px rgba(59,130,246,0.4);
    }
    .nav-title {
        font-size: 0.95rem; font-weight: 700; color: var(--text);
        display: flex; align-items: center; gap: 0.5rem; line-height: 1;
    }
    .nav-badge {
        font-size: 0.65rem; font-weight: 600; padding: 0.2rem 0.45rem;
        background: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.35);
        border-radius: 4px; color: var(--cyan); letter-spacing: 0.03em;
    }
    .nav-subtitle { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.15rem; }

    /* ── Nav buttons row (floated into navbar via position fixed) ── */
    .nav-buttons-row {
        position: fixed !important;
        top: 8px !important; right: 1.5rem !important;
        z-index: 1001 !important;
        display: flex !important;
        gap: 0.4rem !important;
        align-items: center !important;
    }
    .nav-buttons-row .stButton > button {
        background: transparent !important; border: 1px solid var(--border-2) !important;
        color: var(--text) !important; font-size: 0.78rem !important;
        padding: 0.35rem 0.75rem !important; border-radius: 7px !important;
        font-weight: 500 !important; white-space: nowrap !important; box-shadow: none !important;
    }
    .nav-buttons-row .stButton > button:hover {
        background: var(--surface-2) !important; border-color: var(--blue) !important;
        color: var(--blue) !important; transform: none !important; box-shadow: none !important;
    }

    /* ── Page wrap ── */
    .page-wrap { max-width: 900px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }

    /* ── Search card ── */
    .search-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
        padding: 2rem 2rem 1.5rem;
        box-shadow: var(--shadow); margin-bottom: 0;
        border-bottom: none;
    }
    /* Form container extends the search card visually */
    [data-testid="stForm"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
        padding: 0 2rem 1.75rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: var(--shadow) !important;
    }
    .engine-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25);
        border-radius: 999px; color: var(--blue); font-size: 0.75rem; font-weight: 500;
        margin-bottom: 1.25rem;
    }
    .search-heading {
        font-size: 1.75rem; font-weight: 700; color: var(--text);
        line-height: 1.25; margin-bottom: 0.6rem; letter-spacing: -0.02em;
    }
    .search-desc {
        color: var(--text-muted); font-size: 0.88rem; line-height: 1.6;
        max-width: 620px; margin-bottom: 1.5rem;
    }

    /* ── Form container override ── */
    [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    /* ── Input ── */
    .stTextInput label { display: none !important; }
    .stTextInput > div > div {
        background: var(--surface-2) !important; border: 1px solid var(--border-2) !important;
        border-radius: 10px !important;
    }
    .stTextInput input {
        background: transparent !important; color: var(--text) !important;
        font-size: 0.92rem !important; padding: 0.8rem 1rem !important;
        border: none !important; font-family: 'Inter', sans-serif !important;
    }
    .stTextInput input::placeholder { color: var(--text-dim) !important; }
    .stTextInput input:focus { box-shadow: none !important; }
    [data-testid="stTextInput"] > div:focus-within {
        border-color: var(--blue) !important; box-shadow: 0 0 0 3px var(--blue-glow) !important;
    }

    /* ── Main buttons ── */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--blue) 0%, #1d4ed8 100%) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; font-size: 0.88rem !important;
        padding: 0.8rem 1.5rem !important; font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important; letter-spacing: 0.01em !important;
        white-space: nowrap !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background: transparent !important; color: var(--text) !important;
        border: 1px solid var(--border-2) !important; border-radius: 8px !important;
        font-size: 0.82rem !important; font-weight: 500 !important;
        padding: 0.5rem 1rem !important; transition: all 0.18s ease !important;
    }
    .stDownloadButton > button:hover {
        background: var(--surface-2) !important; border-color: var(--blue) !important;
        color: var(--blue) !important;
    }

    /* ── Pipeline card ── */
    .pipeline-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 1.5rem 1.75rem;
        box-shadow: var(--shadow); margin-bottom: 1.5rem;
    }
    .pipeline-header {
        display: flex; align-items: flex-start; justify-content: space-between;
        margin-bottom: 1.25rem;
    }
    .pipeline-title { font-size: 1rem; font-weight: 700; color: var(--text); }
    .pipeline-subtitle { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.2rem; }
    .pipeline-progress-text { font-size: 0.78rem; color: var(--text-muted); text-align: right; }
    .pipeline-progress-text span { color: var(--text); font-weight: 600; }
    .pipeline-stages { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.75rem; margin-bottom: 1rem; }
    .stage-card {
        background: var(--surface-2); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.9rem; transition: all 0.2s ease;
    }
    .stage-card.active { background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.4); }
    .stage-card.complete { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.35); }
    .stage-check {
        width: 20px; height: 20px; border-radius: 50%;
        border: 2px solid var(--green); background: var(--green-dim);
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.7rem; color: var(--green); margin-bottom: 0.5rem;
    }
    .stage-dot {
        width: 20px; height: 20px; border-radius: 50%;
        border: 2px solid var(--border-2); display: inline-block; margin-bottom: 0.5rem;
    }
    .stage-active-dot {
        width: 20px; height: 20px; border-radius: 50%;
        border: 2px solid var(--blue); background: rgba(59,130,246,0.15);
        display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;
    }
    .stage-title { font-size: 0.82rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; }
    .stage-desc { font-size: 0.72rem; color: var(--text-muted); line-height: 1.4; }
    .progress-bar-wrap { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
    .progress-bar-fill { height: 100%; background: linear-gradient(90deg,var(--cyan),var(--blue)); border-radius: 2px; transition: width 0.4s ease; }

    /* ── Result card ── */
    .result-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 1.75rem 2rem;
        box-shadow: var(--shadow); margin-bottom: 1.5rem;
    }
    .result-badge {
        display: inline-block; padding: 0.2rem 0.6rem;
        background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
        border-radius: 5px; color: var(--blue); font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.6rem;
    }
    .result-topic { font-size: 1.35rem; font-weight: 700; color: var(--text); line-height: 1.3; margin-bottom: 0.3rem; letter-spacing: -0.01em; }
    .result-date { font-size: 0.78rem; color: var(--text-muted); }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0 !important; border-bottom: 1px solid var(--border) !important; background: transparent !important; }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important; font-size: 0.85rem !important;
        font-weight: 500 !important; padding: 0.65rem 1rem !important;
        border-radius: 0 !important; background: transparent !important;
        border: none !important; border-bottom: 2px solid transparent !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTabs [aria-selected="true"] { color: var(--blue) !important; border-bottom: 2px solid var(--blue) !important; background: transparent !important; }
    .stTabs [data-baseweb="tab-panel"] { padding: 1.25rem 0 0 !important; }

    /* ── Log boxes ── */
    .content-log {
        background: #060d1a; border: 1px solid var(--border); border-radius: 8px;
        padding: 1rem 1.25rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; line-height: 1.7; color: var(--text-muted);
        max-height: 420px; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
    }
    .content-log::-webkit-scrollbar { width: 6px; }
    .content-log::-webkit-scrollbar-track { background: transparent; }
    .content-log::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 3px; }

    /* ── History drawer ── */
    .history-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(2px); z-index: 999; }
    .history-drawer {
        position: fixed; top: 0; right: 0; bottom: 0; width: 340px;
        background: #0c1525; border-left: 1px solid var(--border);
        z-index: 1001; padding: 1.5rem; overflow-y: auto;
        box-shadow: -8px 0 32px rgba(0,0,0,0.5); animation: slideIn 0.22s ease;
    }
    @keyframes slideIn { from{transform:translateX(100%);opacity:0} to{transform:translateX(0);opacity:1} }
    .drawer-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
    .drawer-title { font-size: 0.95rem; font-weight: 700; color: var(--text); }
    .history-item { background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.6rem; cursor: pointer; transition: all 0.18s ease; }
    .history-item:hover { border-color: var(--blue); background: rgba(59,130,246,0.06); }
    .history-item-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.5rem; margin-bottom: 0.35rem; }
    .history-item-title { font-size: 0.83rem; font-weight: 600; color: var(--text); line-height: 1.3; flex: 1; }
    .history-item-date { font-size: 0.7rem; color: var(--text-dim); white-space: nowrap; flex-shrink: 0; }
    .history-item-preview { font-size: 0.73rem; color: var(--text-muted); line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

    /* ── Toast ── */
    .toast {
        position: fixed; top: 72px; right: 1.5rem;
        background: #0d2a1a; border: 1px solid rgba(16,185,129,0.35);
        border-radius: 8px; padding: 0.65rem 1rem;
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.82rem; color: var(--green); z-index: 1100;
        box-shadow: var(--shadow); animation: fadeInToast 0.3s ease;
    }
    @keyframes fadeInToast { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }

    /* ── Empty state ── */
    .empty-state { text-align: center; padding: 3.5rem 2rem; color: var(--text-muted); }
    .empty-state-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.5; }
    .empty-state p { font-size: 0.9rem; line-height: 1.6; }

    /* ── Footer ── */
    .page-footer { text-align: center; padding: 1.5rem 0 0.5rem; color: var(--text-dim); font-size: 0.75rem; border-top: 1px solid var(--border); margin-top: 2rem; }

    /* ── Spinner ── */
    .stSpinner > div { border-color: var(--blue) transparent transparent transparent !important; }

    /* ── Alerts ── */
    .stAlert { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 8px !important; }

    @media (max-width: 700px) {
        .pipeline-stages { grid-template-columns: repeat(2,1fr); }
        .history-drawer { width: 90vw; }
        .search-heading { font-size: 1.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def as_text(value: object, empty: str = "No content captured.") -> str:
    text = value if isinstance(value, str) else str(value or "")
    return text.strip() or empty


# ─── Session state ────────────────────────────────────────────────────────────

for key, default in [
    ("history", []),
    ("running", False),
    ("show_history", False),
    ("selected_run", None),
    ("toast_msg", ""),
    ("sample_topic", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Navbar ───────────────────────────────────────────────────────────────────

# Render the fixed navbar HTML
hist_count = len(st.session_state.history)
st.markdown(
    f"""
    <div class="navbar">
        <div class="nav-brand">
            <div class="nav-icon">🧠</div>
            <div>
                <div class="nav-title">
                    Agentic Research Studio
                    <span class="nav-badge">v2.4 Multi-Agent</span>
                </div>
                <div class="nav-subtitle">Autonomous Search • Reader • Writer • Critic Pipeline</div>
            </div>
        </div>
        <div class="nav-actions" id="nav-actions-placeholder"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Nav buttons rendered as regular Streamlit buttons
# They're positioned into the navbar via CSS below
_c1, _c2, _c3 = st.columns([8, 1.6, 1.7])
with _c2:
    _hist_btn = st.button(f"🕐 History ({hist_count})", key="nav-history", use_container_width=True)
with _c3:
    _sample_btn = st.button("✨ Sample", key="nav-sample", use_container_width=True)

if _hist_btn:
    st.session_state.show_history = not st.session_state.show_history
    st.rerun()
if _sample_btn:
    st.session_state.sample_topic = "Autonomous Agentic Workflows in 2026"
    st.rerun()

# Pull those columns up into the navbar with position:fixed
st.markdown(
    """
    <style>
    /* The columns rendered right after the navbar markdown block
       use a unique child index to target just the nav row. */
    [data-testid="stHorizontalBlock"].nav-row {
        position: fixed !important;
        top: 8px !important; right: 1.5rem !important;
        z-index: 1001 !important; background: transparent !important;
        gap: 0.4rem !important; width: auto !important;
    }
    /* Fallback: target by raw attribute on nearby element */
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Toast ────────────────────────────────────────────────────────────────────

if st.session_state.toast_msg:
    st.markdown(
        f'<div class="toast">✅ {st.session_state.toast_msg}</div>',
        unsafe_allow_html=True,
    )
    st.session_state.toast_msg = ""


# ─── History drawer ───────────────────────────────────────────────────────────

if st.session_state.show_history:
    st.markdown('<div class="history-overlay"></div>', unsafe_allow_html=True)

    drawer_html = """
    <div class="history-drawer">
        <div class="drawer-header">
            <div class="drawer-title">📋 Past Research Reports</div>
        </div>
    """
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            orig_idx = len(st.session_state.history) - i - 1
            title = item["topic"]
            try:
                date_str = item["created"].strftime("%m/%d/%Y")
            except Exception:
                date_str = str(item.get("created", ""))
            report_preview = as_text(item["state"].get("report", ""))[:120]
            report_preview = report_preview.replace("<", "&lt;").replace(">", "&gt;")
            drawer_html += f"""
            <div class="history-item">
                <div class="history-item-header">
                    <div class="history-item-title">{title[:50]}{"…" if len(title) > 50 else ""}</div>
                    <div class="history-item-date">{date_str}</div>
                </div>
                <div class="history-item-preview">{report_preview}</div>
            </div>
            """
    else:
        drawer_html += '<p style="color:var(--text-muted);font-size:0.82rem;margin-top:0.5rem;">No research history yet.</p>'

    drawer_html += "</div>"
    st.markdown(drawer_html, unsafe_allow_html=True)

    dcol1, dcol2 = st.columns([1, 5])
    with dcol1:
        if st.button("✕ Close", key="close-drawer"):
            st.session_state.show_history = False
            st.rerun()

    for i, item in enumerate(reversed(st.session_state.history)):
        orig_idx = len(st.session_state.history) - i - 1
        label = item["topic"][:35] + ("…" if len(item["topic"]) > 35 else "")
        if st.button(f"📄 Load: {label}", key=f"load-hist-{orig_idx}"):
            st.session_state.selected_run = orig_idx
            st.session_state.show_history = False
            st.session_state.toast_msg = f"Loaded research: {item['topic'][:40]}"
            st.rerun()


# ─── Page wrap ────────────────────────────────────────────────────────────────

st.markdown('<div class="page-wrap">', unsafe_allow_html=True)


# ─── Search card ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="search-card">
        <div class="engine-badge">🔗 LangChain &amp; LangGraph Orchestration Engine</div>
        <h1 class="search-heading">What do you want to research today?</h1>
        <p class="search-desc">
            Enter any technical topic, market trend, or scientific query.
            Our 4-stage agent pipeline will search the web, scrape deep content,
            draft a comprehensive report, and run a rigorous critic review.
        </p>
    """,
    unsafe_allow_html=True,
)

sample = st.session_state.get("sample_topic", "")
with st.form("research-form", clear_on_submit=True):
    inp_col, btn_col = st.columns([5, 2])
    with inp_col:
        topic = st.text_input(
            "topic",
            value=sample,
            placeholder="e.g. Quantum Computing Cryptographic Vulnerabilities 2026...",
            label_visibility="collapsed",
        )
    with btn_col:
        submitted = st.form_submit_button("Run Research Pipeline →", use_container_width=True)




if sample:
    st.session_state.sample_topic = ""


# ─── Pipeline stages renderer ─────────────────────────────────────────────────

def render_pipeline_stages(progress: float = 0, active_stage: int = 0, complete_stages: list = None) -> None:
    if complete_stages is None:
        complete_stages = []

    stages = [
        {"num": 1, "title": "Search Agent",  "desc": "Querying knowledge base & web indices"},
        {"num": 2, "title": "Reader Agent",  "desc": "Scraping top URLs for deep content"},
        {"num": 3, "title": "Writer Chain",  "desc": "Synthesizing comprehensive report"},
        {"num": 4, "title": "Critic Review", "desc": "Validating quality & fact-checking"},
    ]

    pct = int(progress * 100)
    stages_html = ""
    for s in stages:
        n = s["num"]
        if n in complete_stages:
            cls = "stage-card complete"
            indicator = '<div class="stage-check">✓</div>'
        elif n == active_stage:
            cls = "stage-card active"
            indicator = '<div class="stage-active-dot"></div>'
        else:
            cls = "stage-card"
            indicator = '<div class="stage-dot"></div>'

        stages_html += f"""
        <div class="{cls}">
            {indicator}
            <div class="stage-title">{n}. {s["title"]}</div>
            <div class="stage-desc">{s["desc"]}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="pipeline-card">
            <div class="pipeline-header">
                <div>
                    <div class="pipeline-title">≋ Pipeline Execution State</div>
                    <div class="pipeline-subtitle">Real-time status of agent chain coordination</div>
                </div>
                <div class="pipeline-progress-text">Progress &nbsp;<span>{pct}%</span></div>
            </div>
            <div class="pipeline-stages">{stages_html}</div>
            <div class="progress-bar-wrap">
                <div class="progress-bar-fill" style="width:{pct}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Run pipeline ─────────────────────────────────────────────────────────────

if submitted:
    topic = topic.strip()
    if not topic:
        st.error("⚠️ Please enter a research topic before running the pipeline.")
    else:
        st.session_state.running = True
        st.session_state.current_topic = topic

        try:
            render_pipeline_stages(progress=0.1, active_stage=1, complete_stages=[])
            spin_slot = st.empty()

            with spin_slot.container():
                with st.spinner("🔍 Running 4-stage research pipeline — this may take a minute..."):
                    result = run_research_pipeline(topic)

            spin_slot.empty()
            render_pipeline_stages(progress=1.0, active_stage=4, complete_stages=[1, 2, 3, 4])

            st.session_state.history.append({
                "topic": topic,
                "state": result,
                "created": datetime.now(),
            })
            st.session_state.selected_run = len(st.session_state.history) - 1
            st.session_state.running = False
            st.session_state.toast_msg = f"Research complete: {topic[:40]}"
            st.rerun()

        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
            st.session_state.running = False
        except Exception as exc:
            st.error(f"Research pipeline failed: {exc}")
            st.session_state.running = False


# ─── Results ──────────────────────────────────────────────────────────────────

def render_results(topic: str, state: dict) -> None:
    report   = as_text(state.get("report"))
    feedback = as_text(state.get("feedback"))
    search   = as_text(state.get("search_results"))
    scraped  = as_text(state.get("scraped_content"))

    try:
        date_str = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
    except Exception:
        date_str = str(datetime.now())

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-badge">Topic Result</div>
            <div class="result-topic">{topic}</div>
            <div class="result-date">Generated at {date_str}</div>
        """,
        unsafe_allow_html=True,
    )

    # Action buttons
    _r1, _r2, _r3 = st.columns([5, 1.5, 1.8])
    with _r2:
        st.download_button(
            "📋 Copy Report",
            report,
            file_name=f"{topic[:40].strip().replace(' ', '_') or 'report'}.md",
            mime="text/markdown",
            key="copy-report",
            use_container_width=True,
        )
    with _r3:
        st.download_button(
            "⬇ Export Markdown",
            report,
            file_name=f"{topic[:40].strip().replace(' ', '_') or 'report'}.md",
            mime="text/markdown",
            key="export-report",
            use_container_width=True,
        )

    # Style export button blue
    st.markdown(
        """
        <style>
        [data-testid="stDownloadButton"]:nth-of-type(2) > button {
            background: var(--blue) !important; color: white !important; border: none !important;
        }
        [data-testid="stDownloadButton"]:nth-of-type(2) > button:hover {
            background: var(--blue-dark) !important; box-shadow: 0 4px 14px var(--blue-glow) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_report, tab_critic, tab_search, tab_reader = st.tabs(
        ["Final Report", "Critic Feedback", "Search Agent Logs", "Scraped Content"]
    )

    with tab_report:
        st.markdown(report)
    with tab_critic:
        st.markdown(feedback)
    with tab_search:
        safe = search.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="content-log">{safe}</div>', unsafe_allow_html=True)
    with tab_reader:
        safe = scraped.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="content-log">{safe}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close .result-card


# ─── Main display ─────────────────────────────────────────────────────────────

selected = st.session_state.get("selected_run")
if selected is not None and 0 <= selected < len(st.session_state.history):
    render_pipeline_stages(progress=1.0, active_stage=4, complete_stages=[1, 2, 3, 4])
    run = st.session_state.history[selected]
    render_results(run["topic"], run["state"])
elif not st.session_state.running:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">🔬</div>
            <p>Run a topic above to see the report, source trail, and critic feedback.<br>
            The pipeline will search the web, extract content, draft a report, and validate the output.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-footer">Agentic Research Studio — Built with FastAPI, LangChain &amp; LangGraph, React &amp; Tailwind CSS</div>',
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)  # close .page-wrap
