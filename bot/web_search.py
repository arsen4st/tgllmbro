from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlsplit

import httpx


SEARCH_URL = "https://lite.duckduckgo.com/lite/"
MAX_RESULTS = 6
_LINK_RE = re.compile(
    r"<a[^>]+href=(?P<quote>['\"])(?P<href>.+?)(?P=quote)[^>]*"
    r"class=(?P<classquote>['\"])result-link(?P=classquote)[^>]*>"
    r"(?P<title>.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r"<td[^>]+class=(?P<quote>['\"])result-snippet(?P=quote)[^>]*>"
    r"(?P<snippet>.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


async def search_public_web(query: str) -> dict[str, object]:
    normalized = " ".join(query.split())[:500]
    if not normalized:
        return {"ok": False, "error": "Пустой поисковый запрос"}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=6.0),
        follow_redirects=True,
        trust_env=False,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131 Safari/537.36"
            )
        },
    ) as client:
        response = await client.get(SEARCH_URL, params={"q": normalized})
        response.raise_for_status()

    links = list(_LINK_RE.finditer(response.text))
    snippets = list(_SNIPPET_RE.finditer(response.text))
    results: list[dict[str, str]] = []
    for index, link in enumerate(links[:MAX_RESULTS]):
        url = _unwrap_url(html.unescape(link.group("href")))
        if not url:
            continue
        title = _clean_html(link.group("title"))
        snippet = _clean_html(snippets[index].group("snippet")) if index < len(snippets) else ""
        results.append({"title": title or url, "url": url, "snippet": snippet})

    if not results:
        return {"ok": False, "error": "Поиск не вернул результатов"}

    answer = "\n\n".join(
        f"{index}. {item['title']}\n{item['snippet']}\n{item['url']}"
        for index, item in enumerate(results, start=1)
    )
    return {
        "ok": True,
        "provider": "duckduckgo-lite",
        "answer": answer,
        "sources": [
            {"title": item["title"], "url": item["url"]}
            for item in results
        ],
    }


def _clean_html(value: str) -> str:
    text = _TAG_RE.sub(" ", value)
    return " ".join(html.unescape(text).split())


def _unwrap_url(value: str) -> str:
    candidate = value
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    parsed = urlsplit(candidate)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        wrapped = parse_qs(parsed.query).get("uddg", [])
        if wrapped:
            candidate = unquote(wrapped[0])
            parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    return candidate
