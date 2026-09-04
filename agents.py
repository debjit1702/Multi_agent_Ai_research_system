import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from dotenv import load_dotenv

from tools import web_search, scrape_url

load_dotenv()


def _safe_print(msg: str) -> None:
    """Print safely on Windows (avoids cp1252 UnicodeEncodeError)."""
    try:
        print(msg)
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── Provider detection ────────────────────────────────────────────────────────
# Priority: LLM_PROVIDER env var → auto-detect Ollama → fall back to Groq

def _build_llm_ollama():
    """Build a local Ollama LLM. Completely free, no API key, no limits."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise RuntimeError(
            "langchain-ollama is not installed.\n"
            "Run: pip install langchain-ollama"
        )

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Verify Ollama is running before returning
    import urllib.request
    try:
        urllib.request.urlopen(base_url, timeout=3)
    except Exception:
        raise RuntimeError(
            f"Ollama is not running at {base_url}.\n\n"
            "To fix:\n"
            "  1. Download Ollama from https://ollama.com/download\n"
            "  2. Install and launch it\n"
            f"  3. Pull your model:  ollama pull {model}\n"
            "  4. Re-run the pipeline"
        )

    _safe_print(f"[LLM] Ollama model: {model} @ {base_url}")
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def _build_llm_groq():
    """Build a Groq-hosted LLM (free tier, rate-limited)."""
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise RuntimeError(
            "langchain-groq is not installed.\n"
            "Run: pip install langchain-groq"
        )

    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env file.\n"
            "Get a free key at https://console.groq.com/keys"
        )

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    _safe_print(f"[LLM] Groq model: {model}")
    return ChatGroq(model=model, api_key=groq_key, temperature=0)


def _build_llm():
    """
    Select and build the LLM based on LLM_PROVIDER env var.

    Set in your .env file:
        LLM_PROVIDER=ollama    ← local, free, no limits  (default)
        LLM_PROVIDER=groq      ← hosted, free tier, rate-limited

    Ollama model (set in .env):
        OLLAMA_MODEL=qwen2.5:0.5b   ← ultra-fast on low RAM (default)
        OLLAMA_MODEL=llama3.2:1b    ← good quality, needs ~1.3GB RAM
        OLLAMA_MODEL=llama3.2:3b    ← better quality, needs ~2GB RAM
        OLLAMA_MODEL=mistral:7b     ← best quality, needs ~5GB RAM

    Groq model (set in .env):
        GROQ_MODEL=llama-3.1-8b-instant
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

    if provider == "groq":
        return _build_llm_groq()
    elif provider == "ollama":
        return _build_llm_ollama()
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Valid options: 'ollama' or 'groq'"
        )


# ── Lazy LLM singleton ────────────────────────────────────────────────────────
# Built on first use — avoids crashing the UI at import time.

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _build_llm()
    return _llm_instance


# ── Agents ────────────────────────────────────────────────────────────────────

def build_search_agent():
    return create_agent(model=get_llm(), tools=[web_search])


def build_reader_agent():
    return create_agent(model=get_llm(), tools=[scrape_url])


# ── Writer chain ──────────────────────────────────────────────────────────────

def _make_writer_chain():
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
        ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
    ])
    return writer_prompt | get_llm() | StrOutputParser()


# ── Critic chain ──────────────────────────────────────────────────────────────

def _make_critic_chain():
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sharp and constructive research critic. Be honest and specific."),
        ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
    ])
    return critic_prompt | get_llm() | StrOutputParser()


# ── Lazy chain wrappers ───────────────────────────────────────────────────────

class _LazyChain:
    """Builds the chain on first invoke() — avoids LLM init at import time."""

    def __init__(self, factory):
        self._factory = factory
        self._chain = None

    def _get(self):
        if self._chain is None:
            self._chain = self._factory()
        return self._chain

    def invoke(self, input_dict):
        return self._get().invoke(input_dict)


writer_chain = _LazyChain(_make_writer_chain)
critic_chain = _LazyChain(_make_critic_chain)
