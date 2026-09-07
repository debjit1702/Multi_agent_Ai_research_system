"""
Agent definitions for the Agentic Research Studio pipeline.

LLM Provider priority (set LLM_PROVIDER in .env):
  openrouter → OpenRouter API    (free models available)       ← default
  gemini     → Google Gemini API (free tier, 1M tokens/day)
  groq       → Groq Cloud API    (free tier, rate-limited)
  ollama     → Local Ollama      (free, no limits, needs local install)
"""

import os

# ── Streamlit Cloud secrets compatibility ─────────────────────────────────────
# On Streamlit Cloud, secrets are injected via st.secrets (not .env).
# This block syncs them into os.environ so the rest of the code works unchanged.
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # Not running in Streamlit, or no secrets defined — fall back to .env

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from tools import web_search, scrape_url

load_dotenv()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    """Print safely on Windows (avoids cp1252 UnicodeEncodeError)."""
    try:
        print(msg)
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ─── Provider builders ────────────────────────────────────────────────────────

def _build_gemini():
    """
    Google Gemini API — recommended for deployment.
    Free tier: 15 req/min, 1 million tokens/day, no credit card.
    Get your free key at: https://aistudio.google.com/app/apikey
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise RuntimeError(
            "langchain-google-genai not installed.\n"
            "Run: pip install langchain-google-genai"
        )

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY in .env file.\n\n"
            "Steps to get your FREE key:\n"
            "  1. Go to https://aistudio.google.com/app/apikey\n"
            "  2. Sign in with Google (free)\n"
            "  3. Click 'Create API Key'\n"
            "  4. Add GOOGLE_API_KEY=<your-key> to your .env file"
        )

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    _safe_print(f"[LLM] Google Gemini: {model}")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
    )


def _build_openrouter():
    """
    OpenRouter — unified gateway to 100+ LLMs, many with free tiers.
    Get key at: https://openrouter.ai/keys
    Free models: deepseek/deepseek-chat:free, meta-llama/llama-3.1-8b-instruct:free
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise RuntimeError(
            "langchain-openai not installed.\n"
            "Run: pip install langchain-openai"
        )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY in .env file.\n"
            "Get a free key at https://openrouter.ai/keys"
        )

    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat:free")
    _safe_print(f"[LLM] OpenRouter: {model}")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/multi-agent-research",
            "X-Title": "Multi Agent Research System",
        },
    )


def _build_groq():
    """
    Groq Cloud — very fast inference, free tier (rate-limited).
    Get key at: https://console.groq.com/keys
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise RuntimeError(
            "langchain-groq not installed.\n"
            "Run: pip install langchain-groq"
        )

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY in .env file.\n"
            "Get a free key at https://console.groq.com/keys"
        )

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    _safe_print(f"[LLM] Groq: {model}")
    return ChatGroq(model=model, api_key=api_key, temperature=0)


def _build_ollama():
    """
    Local Ollama — 100% offline, unlimited, but not suitable for deployment.
    Install: https://ollama.com/download
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise RuntimeError(
            "langchain-ollama not installed.\n"
            "Run: pip install langchain-ollama"
        )

    import urllib.request
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    try:
        urllib.request.urlopen(base_url, timeout=3)
    except Exception:
        raise RuntimeError(
            f"Ollama is not running at {base_url}.\n"
            f"Start Ollama, then run:  ollama pull {model}"
        )

    _safe_print(f"[LLM] Ollama: {model}")
    return ChatOllama(model=model, base_url=base_url, temperature=0)


# ─── LLM factory ──────────────────────────────────────────────────────────────

_PROVIDERS = {
    "openrouter": _build_openrouter,
    "gemini":     _build_gemini,
    "groq":       _build_groq,
    "ollama":     _build_ollama,
}


def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower().strip()
    builder = _PROVIDERS.get(provider)
    if builder is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER='{provider}'.\n"
            f"Valid options: {list(_PROVIDERS.keys())}"
        )
    return builder()


# ─── Lazy LLM singleton ───────────────────────────────────────────────────────
# Built on first use — the Streamlit UI loads even if the API key is missing.

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _build_llm()
    return _llm_instance


# ─── Agent builders ───────────────────────────────────────────────────────────

def build_search_agent():
    return create_agent(model=get_llm(), tools=[web_search])


def build_reader_agent():
    return create_agent(model=get_llm(), tools=[scrape_url])


# ─── Writer chain ─────────────────────────────────────────────────────────────

def _make_writer_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
        ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report with:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
    ])
    return prompt | get_llm() | StrOutputParser()


# ─── Critic chain ─────────────────────────────────────────────────────────────

def _make_critic_chain():
    prompt = ChatPromptTemplate.from_messages([
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
    return prompt | get_llm() | StrOutputParser()


# ─── Lazy chain wrappers ──────────────────────────────────────────────────────

class _LazyChain:
    """Builds the LangChain chain on first invoke() — defers LLM initialization."""

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
critic_chain  = _LazyChain(_make_critic_chain)
