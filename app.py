"""
Streamlit UI for the multi-agent research pipeline defined in pipeline.py

Run with:
    streamlit run app.py
"""

import streamlit as st

from pipeline import run_research_pipeline


st.set_page_config(
    page_title="Research Pipeline",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --bg: #0b0d14;
            --surface: #12151f;
            --surface-2: #171b28;
            --border: #262b3a;
            --accent: #7c5cff;
            --accent-2: #22d3ee;
            --text: #e7e9f0;
            --text-dim: #9096ab;
            --success: #34d399;
            --danger: #f87171;
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(124,92,255,0.18), transparent 60%),
                radial-gradient(1000px 500px at 100% 0%, rgba(34,211,238,0.12), transparent 55%),
                var(--bg);
        }

        section[data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--border);
        }

        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(124,92,255,0.16), rgba(34,211,238,0.10));
            border: 1px solid var(--border);
            margin-bottom: 1.6rem;
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: "";
            position: absolute;
            top: -60%; right: -10%;
            width: 320px; height: 320px;
            background: radial-gradient(circle, rgba(124,92,255,0.35), transparent 70%);
            filter: blur(10px);
        }
        .hero-eyebrow {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent-2);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 0.4rem 0;
            line-height: 1.15;
        }
        .hero-sub {
            color: var(--text-dim);
            font-size: 0.98rem;
            max-width: 640px;
            margin: 0;
        }

        .pipeline-track {
            display: flex;
            gap: 0.6rem;
            margin-top: 1.4rem;
            flex-wrap: wrap;
        }
        .step-chip {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            font-size: 0.82rem;
            color: var(--text-dim);
            font-weight: 500;
        }
        .step-chip .num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px; height: 20px;
            border-radius: 50%;
            background: var(--border);
            color: var(--text);
            font-size: 0.7rem;
            font-weight: 700;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 1.0rem;
            color: var(--text);
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .stTextInput input {
            background: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.95rem !important;
        }
        .stTextInput input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(124,92,255,0.25) !important;
        }
        .stTextInput label {
            color: var(--text-dim) !important;
            font-weight: 500 !important;
        }

        .stButton > button, .stFormSubmitButton > button {
            background: linear-gradient(135deg, var(--accent), #5f3dff) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.4rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 6px 18px rgba(124,92,255,0.35) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(124,92,255,0.45) !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            background: var(--surface-2) !important;
            box-shadow: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
            border: 1px solid var(--border) !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            width: 100%;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--accent) !important;
            transform: none;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: var(--surface-2);
            border-radius: 10px 10px 0 0;
            border: 1px solid var(--border);
            border-bottom: none;
            color: var(--text-dim);
            font-weight: 500;
            padding: 0.6rem 1.1rem;
        }
        .stTabs [aria-selected="true"] {
            color: var(--text) !important;
            background: var(--surface) !important;
            border-color: var(--accent) !important;
        }

        div[data-testid="stStatusWidget"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: rgba(52,211,153,0.14);
            color: var(--success);
            border: 1px solid rgba(52,211,153,0.35);
        }

        hr {
            border-color: var(--border) !important;
        }

        .scroll-box {
            max-height: 380px;
            overflow-y: auto;
            padding-right: 0.4rem;
        }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"topic": str, "state": dict}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.2rem;'>"
        "<span style='font-size:1.4rem;'>🔎</span>"
        "<span style='font-family:Space Grotesk,sans-serif;font-weight:700;font-size:1.15rem;color:#e7e9f0;'>Research Pipeline</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#9096ab;font-size:0.82rem;margin-bottom:1.2rem;'>Search → Read → Write → Critique</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-weight:600;font-size:0.85rem;color:#e7e9f0;margin-bottom:0.5rem;'>📚 Run history</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.markdown(
            "<div style='color:#9096ab;font-size:0.8rem;'>No runs yet — start one on the right.</div>",
            unsafe_allow_html=True,
        )
    else:
        for i, run in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            label = run["topic"][:38] + ("…" if len(run["topic"]) > 38 else "")
            if st.button(f"📄  {label}", key=f"hist_{idx}"):
                st.session_state.selected_run = idx

    st.markdown("---")
    if st.button("🗑️  Clear history"):
        st.session_state.history = []
        st.session_state.pop("selected_run", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Multi-agent research system</div>
        <h1 class="hero-title">Turn any topic into a reviewed report</h1>
        <p class="hero-sub">
            Four agents work in sequence — a search agent finds sources, a reader agent
            scrapes the best one, a writer drafts the report, and a critic reviews it
            before you see the result.
        </p>
        <div class="pipeline-track">
            <div class="step-chip"><span class="num">1</span> Search agent</div>
            <div class="step-chip"><span class="num">2</span> Reader agent</div>
            <div class="step-chip"><span class="num">3</span> Writer</div>
            <div class="step-chip"><span class="num">4</span> Critic</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
col_form, col_spacer = st.columns([2.2, 1])
with col_form:
    with st.form("research_form"):
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. Latest developments in solid-state batteries",
            label_visibility="visible",
        )
        submitted = st.form_submit_button("🚀  Run pipeline", type="primary")

if submitted:
    if not topic or not topic.strip():
        st.error("Please enter a research topic before running the pipeline.")
    else:
        result_state = {}
        status_box = st.status("Initializing pipeline...", expanded=True)

        try:
            status_box.write("🔍 **Step 1/4** — Search agent gathering information...")
            status_box.write("📖 **Step 2/4** — Reader agent scraping the top source...")
            status_box.write("✍️ **Step 3/4** — Writer drafting the report...")
            status_box.write("🧐 **Step 4/4** — Critic reviewing the report...")

            # NOTE: run_research_pipeline executes all 4 steps internally and
            # only returns once everything is done, so the status list above
            # is shown up front rather than streamed live. For true live
            # progress, split pipeline.py into per-step functions.
            result_state = run_research_pipeline(topic.strip())

            status_box.update(label="Pipeline complete", state="complete", expanded=False)

            st.session_state.history.append({"topic": topic.strip(), "state": result_state})
            st.session_state.selected_run = len(st.session_state.history) - 1

        except ValueError as exc:
            status_box.update(label="Invalid input", state="error")
            st.error(f"Invalid input: {exc}")
        except Exception as exc:
            status_box.update(label="Pipeline failed", state="error")
            st.error(f"Research pipeline failed: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_state(topic: str, state: dict) -> None:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin:1.6rem 0 0.8rem 0;">
            <div>
                <div style="color:#9096ab;font-size:0.8rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;">Results for</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;color:#e7e9f0;">{topic}</div>
            </div>
            <span class="badge">✓ Reviewed by critic</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📝  Report", "🧐  Critic Feedback", "🔍  Search Results", "📄  Scraped Content"]
    )

    with tab_report:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        report = state.get("report", "")
        report_text = report if isinstance(report, str) else str(report)
        st.markdown(report_text)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download report (.md)",
            data=report_text,
            file_name=f"{topic[:40].strip().replace(' ', '_') or 'report'}.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        feedback = state.get("feedback", "")
        st.markdown(feedback if isinstance(feedback, str) else str(feedback))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_search:
        st.markdown('<div class="card"><div class="scroll-box">', unsafe_allow_html=True)
        st.text(state.get("search_results", "No search results captured."))
        st.markdown("</div></div>", unsafe_allow_html=True)

    with tab_scraped:
        st.markdown('<div class="card"><div class="scroll-box">', unsafe_allow_html=True)
        st.text(state.get("scraped_content", "No scraped content captured."))
        st.markdown("</div></div>", unsafe_allow_html=True)


selected_idx = st.session_state.get("selected_run")
if selected_idx is not None and 0 <= selected_idx < len(st.session_state.history):
    run = st.session_state.history[selected_idx]
    render_state(run["topic"], run["state"])
else:
    st.markdown(
        """
        <div class="card" style="text-align:center;padding:2.4rem;">
            <div style="font-size:2rem;margin-bottom:0.4rem;">🧭</div>
            <div style="color:#9096ab;font-size:0.9rem;">
                Run a topic above to see the report, critic feedback, and raw research here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )