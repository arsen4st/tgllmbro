"""Small public-page adapters; no inline-bot impersonation or private endpoints."""
from __future__ import annotations

import re
from html import unescape
from urllib.parse import quote, urlsplit

import httpx

from bot.media import run_media_command


def _allowed(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname in {
        "media.tenor.com", "media1.tenor.com", "www.myinstants.com",
        "tenor.com",
    } and not parsed.username and not parsed.password and parsed.port in (None, 443)


async def _download(url: str, limit: int) -> bytes:
    if not _allowed(url):
        raise ValueError("Источник медиа не разрешён")
    async with httpx.AsyncClient(timeout=15, follow_redirects=False, trust_env=False,
                                 headers={"User-Agent": "tgllmbro/1.0"}) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            result = bytearray()
            async for chunk in response.aiter_bytes():
                result.extend(chunk)
                if len(result) > limit:
                    raise ValueError("Медиа превышает лимит загрузки")
            return bytes(result)


def parse_tenor(page: str, kind: str) -> list[dict]:
    results = []
    for block in re.findall(r'<figure\b[^>]*>(.*?)</figure>', page, re.S):
        link = re.search(r'href="(/view/[^"<>]+)"', block)
        image = re.search(r'<img\b[^>]*src="([^"<>]+)"[^>]*alt="([^"<>]*)"', block)
        clip = re.search(r'https://media\d*\.tenor\.com/[^"<>\s]+\.mp4', block)
        if not link or not image:
            continue
        url = clip.group(0) if clip else unescape(image.group(1))
        if not _allowed(url):
            continue
        results.append({"kind": kind, "url": url, "description": unescape(image.group(2))[:500],
                        "source": "https://tenor.com" + unescape(link.group(1))})
        if len(results) == 6:
            break
    return results


def parse_sounds(page: str) -> list[dict]:
    results = []
    for tag in re.findall(r'<button\b[^>]*>', page):
        call = re.search(r"play\('(/media/sounds/[^']+)',\s*'[^']*',\s*'([^']+)'\)", unescape(tag))
        title = re.search(r'title="Play (.*?) sound"', tag)
        if call and title:
            results.append({"kind": "voice", "url": "https://www.myinstants.com" + call.group(1),
                            "description": unescape(title.group(1))[:500],
                            "source": "https://www.myinstants.com/en/instant/" + quote(call.group(2), safe="") + "/"})
        if len(results) == 6:
            break
    return results


async def search_public_media(query: str, kind: str) -> list[dict]:
    query = query.strip()[:160]
    if not query:
        return []
    if kind in {"voice", "audio"}:
        url = "https://www.myinstants.com/en/search/?name=" + quote(query, safe="")
        return parse_sounds((await _download(url, 2_000_000)).decode("utf-8", errors="replace"))
    if kind not in {"photo", "animation"}:
        raise ValueError("В интернете ищу картинки, GIF и звуки; стикеры — в своей базе")
    url = "https://tenor.com/search/" + quote(query.replace(" ", "-"), safe="") + "-gifs"
    return parse_tenor((await _download(url, 2_000_000)).decode("utf-8", errors="replace"), kind)


async def download_public_media(item: dict) -> bytes:
    data = await _download(item["url"], 10_000_000)
    if item["kind"] == "photo":
        # Public GIF search can also supply a static reaction picture.
        return await run_media_command("ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                                       "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1", data=data)
    return data
