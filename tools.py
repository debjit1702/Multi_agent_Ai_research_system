"""Bounded search and public-page retrieval for the research graph."""

from __future__ import annotations

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from http.client import HTTPConnection, HTTPException, HTTPResponse
from ipaddress import ip_address
import os
import re
import socket
import ssl
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Callable, TypedDict, TypeVar
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import requests

SEARCH_URL = "https://api.tavily.com/search"
MAX_PAGE_BYTES = 512 * 1024
PAGE_CACHE_TTL = 900
PAGE_CACHE_SIZE = 64
_WORKERS = ThreadPoolExecutor(max_workers=6, thread_name_prefix="research-io")
_CAPACITY = BoundedSemaphore(12)
_LOCK = Lock()
_PAGES: OrderedDict[str, tuple[float, str]] = OrderedDict()
_HOSTS: Counter[str] = Counter()
_SEARCH_CREDITS: Counter[str] = Counter()
_TLS = ssl.create_default_context()
_TLS.load_verify_locations(cafile=requests.certs.where())


class Candidate(TypedDict):
    url: str
    title: str
    snippet: str
    score: float
    host: str


class Source(Candidate):
    id: str


class Passage(TypedDict):
    source_id: str
    kind: str
    text: str


class Evidence(Passage):
    facet: str


class ToolError(RuntimeError):
    """A retrieval error safe to surface in the app."""


T = TypeVar("T")


def clean_text(text: str) -> str:
    return " ".join(unescape(text).split())


def estimate_tokens(text: str) -> int:
    """Estimate prompt size without downloading a provider tokenizer."""
    ascii_chars = sum(char.isascii() for char in text)
    extra_bytes = len(text.encode("utf-8")) - ascii_chars
    return max(len(re.findall(r"\w+|[^\w\s]", text)), (ascii_chars + 2) // 3) + extra_bytes


def trim_tokens(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].strip()


def public_url(value: str) -> str:
    """Validate the URL syntax. The fetcher separately validates all DNS answers."""
    if len(value) > 2_048 or re.search(r'[\x00-\x20\x7f\\<>"]', value):
        raise ToolError("Invalid source URL.")
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").rstrip(".").lower()
        if parts.scheme not in {"http", "https"} or not host or parts.username is not None:
            raise ValueError
        expected_port = 443 if parts.scheme == "https" else 80
        if parts.port not in {None, expected_port}:
            raise ValueError
        if ":" in host:
            address = ip_address(host)
            if address.scope_id or not _public_address(host):
                raise ValueError
            host = "[" + address.compressed + "]"
        else:
            host = host.encode("idna").decode("ascii")
            if not re.fullmatch(r"[a-z0-9.-]+", host) or "." not in host:
                raise ValueError
            if all(char in "0123456789." for char in host) and not _public_address(host):
                raise ValueError
        return urlunsplit((parts.scheme, host, parts.path or "/", parts.query, ""))
    except (ValueError, UnicodeError) as exc:
        raise ToolError("Only public HTTP or HTTPS source URLs are supported.") from exc


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    tracking = {"fbclid", "gclid", "mc_cid", "mc_eid"}
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in tracking]
    return urlunsplit(("", parts.netloc.removeprefix("www."), parts.path.rstrip("/"),
                       urlencode(sorted(query)), ""))


def _public_address(host: str) -> bool:
    address = ip_address(host)
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast and not address.is_reserved


def _resolve_public(host: str, port: int) -> str:
    addresses = list(dict.fromkeys(
        row[4][0] for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ))
    if not addresses or any(not _public_address(address) for address in addresses):
        raise ToolError("Source host did not resolve exclusively to public addresses.")
    return addresses[0]


def _remaining(deadline: float, cap: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ToolError("The retrieval deadline expired.")
    return min(remaining, cap)


def _tavily_search(query: str, max_results: int, deadline: float) -> list[Candidate]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise ToolError("Web search is unavailable. Configure TAVILY_API_KEY.")
    timeout = _remaining(deadline, 8.0)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    with _LOCK:
        for previous in tuple(_SEARCH_CREDITS):
            if previous != month:
                del _SEARCH_CREDITS[previous]
        if _SEARCH_CREDITS[month] >= 1_000:
            raise ToolError("The process has reached its monthly search allowance.")
        _SEARCH_CREDITS[month] += 1
    try:
        response = requests.post(
            SEARCH_URL, headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "search_depth": "basic", "max_results": max_results,
                  "auto_parameters": False, "include_answer": False, "include_raw_content": False},
            timeout=(min(timeout, 3), timeout),
        )
        if response.status_code in {429, 432, 433}:
            raise ToolError("Search quota is temporarily unavailable.")
        if response.status_code in {401, 403}:
            raise ToolError("Tavily rejected the configured credentials or permissions.")
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise ToolError("Tavily search timed out.") from exc
    except (requests.RequestException, ValueError) as exc:
        raise ToolError("Tavily search was unavailable.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ToolError("Tavily returned an invalid search response.")
    candidates = []
    for entry in payload["results"][:max_results]:
        if not isinstance(entry, dict):
            continue
        try:
            url = public_url(str(entry.get("url", "")))
            score = min(1.0, max(0.0, float(entry.get("score") or 0)))
        except (ToolError, TypeError, ValueError):
            continue
        host = urlsplit(url).hostname or ""
        candidates.append(Candidate(
            url=url, title=clean_text(str(entry.get("title") or host))[:180],
            snippet=clean_text(str(entry.get("content") or ""))[:1_200], score=score, host=host,
        ))
    return candidates


def _page_html(url: str, deadline: float) -> tuple[bytes, str]:
    for _ in range(4):
        parts = urlsplit(public_url(url))
        host = parts.hostname or ""
        port = 443 if parts.scheme == "https" else 80
        address = _resolve_public(host, port)
        path = quote(urlunsplit(("", "", parts.path, parts.query, "")), safe="/?=%&;:+,@!$'()*[]~-._")
        timeout = _remaining(deadline, 4)
        with _LOCK:
            if _HOSTS[host] >= 2:
                raise ToolError("This source host is busy.")
            _HOSTS[host] += 1
        connection = HTTPConnection(host, port, timeout=timeout)
        sock = None
        try:
            sock = socket.create_connection((address, port), timeout=_remaining(deadline, 3))
            if parts.scheme == "https":
                sock = _TLS.wrap_socket(sock, server_hostname=host)
            connection.sock = sock
            sock.settimeout(_remaining(deadline, 4))
            connection.request("GET", path, headers={
                "User-Agent": "ResearchRelay/1.0", "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity", "Connection": "close",
            })
            # The connection owns the socket until the read finishes. HTTPConnection.getresponse()
            # closes its socket early on Connection: close responses.
            with HTTPResponse(sock) as response:
                response.begin()
                if response.status in {301, 302, 303, 307, 308} and response.getheader("Location"):
                    url = urljoin(url, response.getheader("Location"))
                    continue
                if not 200 <= response.status < 300:
                    raise ToolError("The page was unavailable.")
                media = response.headers.get_content_type()
                size = response.getheader("Content-Length")
                if (media not in {"text/html", "application/xhtml+xml"}
                        or response.getheader("Content-Encoding", "identity").lower() != "identity"
                        or (size and size.isdecimal() and int(size) > MAX_PAGE_BYTES)):
                    raise ToolError("The page format or size is outside the reading budget.")
                body = bytearray()
                while len(body) <= MAX_PAGE_BYTES:
                    sock.settimeout(_remaining(deadline, 4))
                    chunk = response.read1(min(32_768, MAX_PAGE_BYTES + 1 - len(body)))
                    if not chunk:
                        return bytes(body), response.headers.get_content_charset() or "utf-8"
                    body.extend(chunk)
                raise ToolError("The page exceeded the reading budget.")
        finally:
            connection.close()
            if sock is not None:
                sock.close()
            with _LOCK:
                _HOSTS[host] -= 1
                if not _HOSTS[host]:
                    del _HOSTS[host]
    raise ToolError("The page exceeded the redirect limit.")


def _read_page(url: str, deadline: float) -> str:
    with _LOCK:
        cached = _PAGES.get(url)
        if cached and cached[0] > monotonic():
            _PAGES.move_to_end(url)
            return cached[1]
    try:
        body, charset = _page_html(url, min(deadline, monotonic() + 8))
        text = body.decode(charset, errors="replace")
    except ssl.SSLCertVerificationError as exc:
        raise ToolError("A source failed TLS certificate verification.") from exc
    except TimeoutError as exc:
        raise ToolError("A source timed out.") from exc
    except (OSError, HTTPException, LookupError, ValueError) as exc:
        raise ToolError("A source could not be read.") from exc
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    article = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = article.find_all(["h1", "h2", "h3", "p", "li"], limit=160)
    paragraphs = [clean_text(block.get_text(" ", strip=True)) for block in blocks]
    extracted = "\n".join(paragraph for paragraph in paragraphs if len(paragraph) >= 24)[:24_000]
    if not extracted:
        raise ToolError("The page had no readable passages.")
    with _LOCK:
        _PAGES[url] = (monotonic() + PAGE_CACHE_TTL, extracted)
        _PAGES.move_to_end(url)
        while len(_PAGES) > PAGE_CACHE_SIZE:
            _PAGES.popitem(last=False)
    return extracted


@dataclass
class Batch:
    attempted: int = 0
    cached: int = 0
    errors: list[str] = field(default_factory=list)


def _parallel(items: list[str], action: Callable[[str], T], deadline: float,
              result: Batch, publish: Callable[[str, T], None]) -> None:
    pending = {}
    for key in items:
        if not _CAPACITY.acquire(blocking=False):
            result.errors.append("The retrieval workers are busy.")
            break
        try:
            future = _WORKERS.submit(action, key)
        except RuntimeError:
            _CAPACITY.release()
            result.errors.append("The retrieval workers are unavailable.")
            break
        future.add_done_callback(lambda _: _CAPACITY.release())
        pending[future] = key
        result.attempted += 1
    try:
        for future in as_completed(pending, timeout=max(0, deadline - monotonic())):
            try:
                publish(pending[future], future.result())
            except ToolError as exc:
                result.errors.append(str(exc))
    except TimeoutError:
        result.errors.append("Some retrieval requests exceeded the stage deadline.")
    finally:
        for future in pending:
            future.cancel()


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[^\W_]+", text.casefold()) if len(term) > 2}


def rank_sources(candidates: list[Source], question: str) -> list[Source]:
    terms = _terms(question)

    def score(item: Source) -> float:
        overlap = len(terms & _terms(item["title"] + " " + item["snippet"]))
        return item["score"] + min(overlap, 6) * 0.06

    # Give different publishers a chance before spending multiple slots on one host.
    ranked = sorted(candidates, key=score, reverse=True)
    seen: set[str] = set()
    first, extra = [], []
    for item in ranked:
        host = item["host"].removeprefix("www.")
        (extra if host in seen else first).append(item)
        seen.add(host)
    return first + extra


def rank_passages(source: Source, page: str, focus: str, token_limit: int) -> list[Passage]:
    terms = _terms(focus)
    blocks = sorted(page.splitlines(), key=lambda line: len(terms & _terms(line)), reverse=True)
    passages = []
    for kind, blocks_of_kind in (("page", blocks[:4]), ("search snippet", [source["snippet"]])):
        for block in blocks_of_kind:
            excerpt = trim_tokens(block, min(token_limit, 200))
            if len(excerpt) >= 28:
                passages.append(Passage(source_id=source["id"], kind=kind, text=excerpt))
                token_limit -= estimate_tokens(excerpt)
    return passages


class ResearchTools:
    """The search cache belongs to one session run. The bounded page cache is shared."""

    def __init__(self) -> None:
        self._search_cache: dict[tuple[str, int], list[Candidate]] = {}

    def check_search_configuration(self) -> None:
        if not os.getenv("TAVILY_API_KEY"):
            raise ToolError("Web search is unavailable. Configure TAVILY_API_KEY.")

    def search(self, queries: list[str], max_results: int, deadline: float,
               on_result: Callable[[Candidate], None]) -> Batch:
        result = Batch()
        pending = []

        def publish(query: str, candidates: list[Candidate]) -> None:
            self._search_cache[(query, max_results)] = candidates
            for candidate in candidates:
                on_result(candidate)

        for query in queries:
            cached = self._search_cache.get((query, max_results))
            if cached is None:
                pending.append(query)
            else:
                result.cached += 1
                publish(query, cached)

        _parallel(pending, lambda query: _tavily_search(query, max_results, deadline),
                  deadline, result, publish)
        return result

    def read(self, sources: list[Source], deadline: float,
             on_result: Callable[[str, str], None]) -> Batch:
        result = Batch()
        urls = []
        by_url = {source["url"]: source["id"] for source in sources}
        for url, source_id in by_url.items():
            with _LOCK:
                cached = _PAGES.get(url)
                if cached and cached[0] > monotonic():
                    _PAGES.move_to_end(url)
            if cached and cached[0] > monotonic():
                result.cached += 1
                on_result(source_id, cached[1])
            else:
                urls.append(url)

        _parallel(urls, lambda url: _read_page(url, deadline), deadline, result,
                  lambda url, text: on_result(by_url[url], text))
        return result
