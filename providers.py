"""Free-tier model routing and bounded LangChain calls."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import re
from threading import Lock
from time import monotonic, sleep
from typing import Callable, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from tools import estimate_tokens

Tier = Literal["fast", "deep"]
Role = Literal["search", "reader", "writer", "critic"]
Event = Callable[[dict], None]
_PROMPT = ChatPromptTemplate.from_messages([("system", "{instructions}"), ("human", "{data}")])
_LOCK = Lock()
_COOLDOWN: dict[str, float] = {}


@dataclass(frozen=True)
class Route:
    provider: str
    model: str


class Attempt(TypedDict):
    provider: str
    model: str
    outcome: str
    seconds: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_input: int


@dataclass
class Reservation:
    at: float
    tokens: int


_USAGE: dict[str, deque[Reservation]] = defaultdict(deque)

MODELS = {
    "groq": ("openai/gpt-oss-20b", "openai/gpt-oss-120b"),
    "cloudflare": ("@cf/openai/gpt-oss-20b", "@cf/openai/gpt-oss-120b"),
    "openrouter": ("google/gemma-4-26b-a4b-it:free", "google/gemma-4-26b-a4b-it:free"),
    "gemini": ("gemini-3.1-flash-lite", "gemini-3.8-flash"),
    "ollama": ("qwen2.5:0.5b", "qwen2.5:0.5b"),
}
GROQ_REVIEW_MODEL = "qwen/qwen3.8-27b"
_REASON = {
    "budget": "The run reached its model request or time budget.",
    "configuration": "No usable free model route is configured for this stage.",
    "invalid_response": "The model response was empty, truncated, or invalid for this stage.",
    "rate_limit": "The configured free model routes are temporarily at capacity.",
    "timeout": "The model request timed out.",
    "unavailable": "The configured model routes were temporarily unavailable.",
}


class ModelError(RuntimeError):
    def __init__(self, kind: str, retry_after: float = 0, *, detail: str | None = None):
        super().__init__(detail or _REASON[kind])
        self.kind = kind
        self.retry_after = retry_after


def _configured(provider: str) -> bool:
    keys = {
        "groq": ("GROQ_API_KEY",),
        "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "openrouter": ("OPENROUTER_API_KEY",),
        "cloudflare": ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"),
    }
    if provider == "ollama":
        return True
    if provider == "gemini":
        return any(os.getenv(key) for key in keys[provider])
    return all(os.getenv(key) for key in keys[provider])


def _routes(tier: Tier, role: Role | None = None) -> list[Route]:
    primary = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not primary:
        primary = next((name for name in ("groq", "cloudflare", "openrouter") if _configured(name)), "groq")
    fallbacks = os.getenv("LLM_FALLBACK_PROVIDERS", "cloudflare").lower().split(",")
    routes = []
    spare_routes = []
    for provider in dict.fromkeys([primary, *(name.strip() for name in fallbacks)]):
        if provider not in MODELS or not _configured(provider):
            continue
        model = MODELS[provider][tier == "deep"]
        if provider == "groq":
            if tier == "deep" and role == "critic":
                spare_routes.append(Route(provider, model))
                model = GROQ_REVIEW_MODEL
            else:
                spare_routes.append(Route(provider, GROQ_REVIEW_MODEL))
        elif provider == "openrouter":
            model = os.getenv("OPENROUTER_MODEL", model).strip()
            if not re.fullmatch(r"[a-z0-9._-]+/[a-z0-9._-]+:free", model):
                continue
        elif provider == "gemini":
            model = os.getenv("GEMINI_MODEL", model).strip()
            if model not in MODELS["gemini"]:
                continue
        elif provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", model).strip()
        routes.append(Route(provider, model))
    return routes + spare_routes


def _capacity_wait(history: list[Reservation], now: float, window: int,
                   requests: int, tokens: int, token_limit: int) -> float:
    used_requests = len(history)
    used_tokens = sum(item.tokens for item in history)
    wait = 0.0
    for item in history:
        if used_requests < requests and used_tokens + tokens <= token_limit:
            break
        used_requests -= 1
        used_tokens -= item.tokens
        wait = max(0.0, item.at + window - now)
    return wait


def _reserve(route: Route, tokens: int) -> Reservation:
    # The server-side plan remains the source of truth. These are process-local ceilings.
    now = monotonic()
    provider = route.provider
    key = route.model if provider == "groq" else provider
    with _LOCK:
        cooldown = _COOLDOWN.get(key, 0) - now
        if cooldown > 0:
            raise ModelError("rate_limit", cooldown)
        reservation = Reservation(now, tokens)
        if provider not in {"groq", "openrouter"}:
            return reservation
        history = _USAGE[key]
        while history and history[0].at <= now - 86_400:
            history.popleft()
        minute = [item for item in history if item.at > now - 60]
        rpm, rpd = (30, 1_000) if provider == "groq" else (20, 50)
        if provider == "groq" and tokens > 8_000:
            raise ModelError("budget", detail="This stage exceeds a Groq model's free per-minute token allowance.")
        charged_tokens = tokens if provider == "groq" else 0
        wait = max(
            _capacity_wait(minute, now, 60, rpm, charged_tokens, 8_000),
            _capacity_wait(list(history), now, 86_400, rpd, charged_tokens, 200_000),
        )
        if wait > 0:
            raise ModelError("rate_limit", wait, detail="The app's local request or token allowance is temporarily full.")
        history.append(reservation)
        return reservation


def _retry_after(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after", "")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return 0.0


def _classify(exc: Exception) -> ModelError:
    if isinstance(exc, ModelError):
        return exc
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    name = type(exc).__name__.lower()
    if getattr(exc, "code", None) in {"json_validate_failed", "invalid_json", "schema_validation_failed"}:
        return ModelError("invalid_response")
    if status == 429 or "ratelimit" in name:
        return ModelError("rate_limit", _retry_after(exc))
    if status in {408, 504} or "timeout" in name:
        return ModelError("timeout")
    if status in {400, 401, 402, 403, 404, 422} or isinstance(exc, (ImportError, TypeError)):
        return ModelError("configuration")
    if isinstance(exc, ValueError) or "parser" in name:
        return ModelError("invalid_response")
    return ModelError("unavailable")


def _client(route: Route, timeout: float, output: int):
    provider = route.provider
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=route.model, timeout=timeout, max_retries=0, max_tokens=output,
                        reasoning_effort="low", temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=route.model, api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            vertexai=False, timeout=timeout, max_retries=0, max_output_tokens=output,
            thinking_level="low" if route.model == MODELS["gemini"][1] else "minimal",
            temperature=0,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=route.model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            client_kwargs={"timeout": timeout}, num_predict=output, temperature=0,
        )

    from langchain_openai import ChatOpenAI
    if provider == "cloudflare":
        account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not re.fullmatch(r"[a-fA-F0-9]{32}", account):
            raise ModelError("configuration")
        url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
        key = os.getenv("CLOUDFLARE_API_TOKEN")
    else:
        url = "https://openrouter.ai/api/v1"
        key = os.getenv("OPENROUTER_API_KEY")
    return ChatOpenAI(
        model=route.model, api_key=key, base_url=url, timeout=timeout, max_retries=0,
        max_tokens=output, temperature=0, stream_usage=provider != "cloudflare",
        use_responses_api=False,
    )


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content
                       if isinstance(block, dict) and block.get("type") == "text")
    return content


def _model_call(route: Route, messages, schema: type[BaseModel] | None,
                timeout: float, output: int, deadline: float, emit: Event,
                usage: dict) -> BaseModel | str:
    model = _client(route, timeout, output)
    if schema is not None:
        if route.provider == "groq" and route.model == GROQ_REVIEW_MODEL:
            # The current LangChain adapter omits strict mode for Qwen. Bind the
            # Groq-supported schema explicitly and still validate the response locally.
            model = model.bind(response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
            })
            message = model.invoke(messages)
            usage.update(message.usage_metadata or {})
            return schema.model_validate_json(_message_text(message))
        if route.provider == "cloudflare":
            message = model.invoke(messages)
            usage.update(message.usage_metadata or {})
            text = _message_text(message).strip()
            text = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", text, flags=re.IGNORECASE)
            return schema.model_validate_json(text)
        method = "json_schema" if route.provider in {"groq", "gemini", "ollama"} else "json_mode"
        options = {"strict": True} if route.provider == "groq" else {}
        result = model.with_structured_output(schema, method=method, include_raw=True, **options).invoke(messages)
        usage.update(getattr(result.get("raw"), "usage_metadata", None) or {})
        parsed = result.get("parsed")
        if parsed is None:
            raise ModelError("invalid_response")
        try:
            return schema.model_validate(parsed)
        except ValueError as exc:
            raise ModelError("invalid_response") from exc

    chunks = []
    finish = ""
    emit({"type": "draft_reset"})
    for chunk in model.stream(messages):
        if monotonic() >= deadline:
            raise ModelError("timeout")
        content = _message_text(chunk)
        if content:
            chunks.append(content)
            emit({"type": "draft_delta", "text": content})
        if chunk.usage_metadata:
            usage.update(chunk.usage_metadata)
        finish = chunk.response_metadata.get("finish_reason") or finish
    report = "".join(chunks).strip()
    if not report or str(finish).lower() in {"length", "max_tokens", "content_filter"}:
        raise ModelError("invalid_response")
    return report


@dataclass(frozen=True)
class ModelRequest:
    tier: Tier
    instructions: str
    data: dict
    schema: type[BaseModel] | None
    deadline: float
    output_tokens: int
    max_attempts: int
    role: Role | None = None


class ModelRouter:
    def invoke(self, request: ModelRequest, on_attempt: Callable[[Attempt], None],
               emit: Event) -> BaseModel | str:
        if request.max_attempts <= 0:
            raise ModelError("budget")
        routes = _routes(request.tier, request.role)
        if not routes:
            raise ModelError("configuration")
        data = json.dumps(request.data, ensure_ascii=False, separators=(",", ":"))
        instructions = request.instructions
        if request.schema:
            # JSON-mode gateways also need the shape in the prompt. Only append it for those routes.
            schema_text = "\nReturn a JSON object matching this schema:\n" + json.dumps(
                request.schema.model_json_schema(), separators=(",", ":")
            )
        else:
            schema_text = ""
        attempts = 0
        last = ModelError("rate_limit")
        capacity_errors: list[ModelError] = []
        for route in routes:
            route_key = route.model if route.provider == "groq" else route.provider
            human_prompt = instructions + (
                schema_text if route.provider in {"openrouter", "cloudflare"} else ""
            )
            messages = _PROMPT.format_messages(instructions=human_prompt, data=data)
            estimate = estimate_tokens(human_prompt + data) + 32
            if request.schema and route.provider not in {"openrouter", "cloudflare"}:
                estimate += estimate_tokens(schema_text)
            for retry in range(2):
                remaining = request.deadline - monotonic()
                if attempts >= request.max_attempts:
                    raise last
                if remaining < 2:
                    raise ModelError("budget")
                try:
                    reserved = _reserve(route, estimate + request.output_tokens)
                except ModelError as exc:
                    last = exc
                    if exc.kind == "rate_limit":
                        capacity_errors.append(exc)
                    break
                attempts += 1
                started = monotonic()
                usage: dict = {}
                outcome = "ok"
                try:
                    result = _model_call(
                        route, messages, request.schema, min(remaining, 24), request.output_tokens,
                        request.deadline, emit, usage,
                    )
                    return result
                except Exception as exc:
                    last = _classify(exc)
                    outcome = last.kind
                finally:
                    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
                    if isinstance(inputs, int) and isinstance(outputs, int):
                        with _LOCK:
                            reserved.tokens = inputs + outputs
                    elif outcome == "rate_limit":
                        # A rejected request may count against RPM, but no generation took place.
                        with _LOCK:
                            reserved.tokens = 0
                    on_attempt(Attempt(
                        provider=route.provider, model=route.model, outcome=outcome,
                        seconds=round(monotonic() - started, 3), input_tokens=inputs,
                        output_tokens=outputs, estimated_input=estimate,
                    ))
                if last.kind == "rate_limit":
                    delay = last.retry_after
                    cooldown = max(5, min(delay, 3_600))
                    with _LOCK:
                        _COOLDOWN[route_key] = monotonic() + cooldown
                    last.retry_after = cooldown
                    capacity_errors.append(last)
                    if len(routes) > 1 or retry or delay <= 0 or delay > 2:
                        break
                    if request.deadline - monotonic() < delay + 3:
                        break
                    # The provider gave a short retry window and this is the only eligible route.
                    with _LOCK:
                        _COOLDOWN[route_key] = monotonic() + delay
                    sleep(delay)
                elif last.kind == "unavailable" and len(routes) == 1 and not retry:
                    if request.deadline - monotonic() < 4:
                        break
                else:
                    break
        if capacity_errors and last.kind in {"rate_limit", "configuration", "budget"}:
            raise min(capacity_errors, key=lambda error: error.retry_after)
        raise last
