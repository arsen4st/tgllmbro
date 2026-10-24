from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx


MAX_DOWNLOAD_BYTES = 1_000_000
MAX_TEXT_CHARS = 14_000


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data)


async def read_public_url(url: str) -> dict[str, object]:
    current = url.strip()
    headers = {"User-Agent": "tgllmbro/1.0 (+Telegram assistant)"}
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(12.0, connect=6.0),
        trust_env=False,
        headers=headers,
    ) as client:
        for _ in range(4):
            await _validate_public_url(current)
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Редирект без адреса")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if not any(
                    allowed in content_type
                    for allowed in ("text/", "json", "xml", "xhtml")
                ):
                    raise ValueError(f"Неподдерживаемый тип страницы: {content_type or 'неизвестен'}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Страница больше допустимого лимита 1 МБ")
                encoding = response.encoding or "utf-8"
                raw_text = bytes(body).decode(encoding, errors="replace")
                if "html" in content_type or "xhtml" in content_type:
                    parser = _VisibleTextParser()
                    parser.feed(raw_text)
                    raw_text = "\n".join(parser.parts)
                text = re.sub(r"[ \t]+", " ", raw_text)
                text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
                return {
                    "ok": True,
                    "url": str(response.url),
                    "content_type": content_type.split(";", 1)[0],
                    "text": text[:MAX_TEXT_CHARS],
                    "truncated": len(text) > MAX_TEXT_CHARS,
                }
    raise ValueError("Слишком много редиректов")


async def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Нужна обычная публичная http/https-ссылка")
    if parsed.username or parsed.password:
        raise ValueError("Ссылки с логином или паролем запрещены")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise ValueError("Не удалось разрешить адрес сайта") from error
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Локальные и служебные адреса запрещены")
