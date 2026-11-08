from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import monotonic
from typing import Any

import httpx

from bot.web_reader import read_public_url
from bot.web_search import search_public_web


ARENA_BASES = (
    "https://arena.fastapicloud.dev/api",
    "https://arena.rone.dev/api",
)
MLBB_GG = "https://mlbb.gg"
OFFICIAL_PATCH_NOTES = "https://www.mobilelegends.com/news/patch-notes"
_catalog_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
_ALIASES = {
    "миа": "Miya",
    "ханаби": "Hanabi",
    "пакито": "Paquito",
    "белерик": "Belerick",
    "глу": "Gloo",
    "эстес": "Estes",
    "фанни": "Fanny",
    "сейбер": "Saber",
    "аргус": "Argus",
    "рафаэла": "Rafaela",
    "линг": "Ling",
    "мартис": "Martis",
    "грейнджер": "Granger",
    "алукард": "Alucard",
    "зилонг": "Zilong",
}


async def search_mlbb_live(query: str) -> dict[str, object]:
    normalized = " ".join(query.split())[:300]
    catalog, version, ranking = await asyncio.gather(
        _safe_hero_catalog(),
        _safe_arena_get("/academy/meta/version"),
        _safe_arena_get(
            "/heroes/rank",
            params={
                "days": 7,
                "rank": "all",
                "sort_field": "win_rate",
                "size": 10,
            },
        ),
    )
    hero = _match_hero(normalized, catalog)
    arena_results: dict[str, object] = {
        "version_endpoint": _compact(version, 5),
        "top_win_rate_7d": _compact(ranking, 10),
    }
    mlbb_gg_url = f"{MLBB_GG}/tierlist"
    if hero is not None:
        hero_id = int(hero["id"])
        hero_name = str(hero["name"])
        detail, stats, counters, builds = await asyncio.gather(
            _safe_arena_get(f"/heroes/{hero_id}"),
            _safe_arena_get(f"/heroes/{hero_id}/stats"),
            _safe_arena_get(f"/heroes/{hero_id}/counters", params={"days": 7, "rank": "all"}),
            _safe_arena_get(f"/academy/heroes/{hero_id}/builds"),
        )
        arena_results.update(
            {
                "matched_hero": {"id": hero_id, "name": hero_name},
                "hero": _compact(detail, 3),
                "stats": _compact(stats, 5),
                "counters": _compact(counters, 8),
                "builds": _compact(builds, 5),
            }
        )
        mlbb_gg_url = f"{MLBB_GG}/heroes/{hero_id}"

    mlbb_gg: dict[str, object]
    try:
        page = await read_public_url(mlbb_gg_url)
        mlbb_gg = {
            "url": page.get("url", mlbb_gg_url),
            "text": str(page.get("text") or "")[:7000],
            "community_source": True,
        }
    except Exception as error:
        mlbb_gg = {"url": mlbb_gg_url, "error": str(error)[:300]}

    official_search: dict[str, object] = {"ok": False, "reason": "not needed"}
    if any(
        marker in normalized.casefold()
        for marker in ("патч", "верси", "обнов", "мета", "актуал", "current", "patch")
    ):
        try:
            official_search = await search_public_web(
                f"site:mobilelegends.com/news/patch-notes {normalized}"
            )
        except Exception as error:
            official_search = {"ok": False, "error": str(error)[:300]}

    return {
        "ok": True,
        "query": normalized,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "arena": arena_results,
        "mlbb_gg": mlbb_gg,
        "official_patch_search": official_search,
        "sources": [
            {
                "name": "Rone Arena API",
                "url": "https://arena.rone.dev/",
                "kind": "community API over public game data",
            },
            {
                "name": "MLBB.GG",
                "url": mlbb_gg_url,
                "kind": "community statistics and builds",
            },
            {
                "name": "Official MLBB patch notes",
                "url": OFFICIAL_PATCH_NOTES,
                "kind": "official; authoritative for balance changes",
            },
        ],
        "version_warning": (
            "Arena version metadata may lag behind its live hero/stat data. "
            "Do not call a patch current unless official patch notes or another dated source agrees."
        ),
    }


async def _hero_catalog() -> list[dict[str, Any]]:
    global _catalog_cache
    cached_at, cached = _catalog_cache
    if cached and monotonic() - cached_at < 600:
        return cached
    payload = await _arena_get("/heroes", params={"size": 200})
    catalog: list[dict[str, Any]] = []
    for record in _records(payload):
        data = record.get("data") if isinstance(record, dict) else None
        hero = data.get("hero") if isinstance(data, dict) else None
        hero_data = hero.get("data") if isinstance(hero, dict) else None
        hero_id = data.get("hero_id") if isinstance(data, dict) else None
        name = hero_data.get("name") if isinstance(hero_data, dict) else None
        if isinstance(hero_id, int) and isinstance(name, str):
            catalog.append({"id": hero_id, "name": name})
    _catalog_cache = (monotonic(), catalog)
    return catalog


async def _safe_hero_catalog() -> list[dict[str, Any]]:
    try:
        return await _hero_catalog()
    except Exception:
        return []


def _match_hero(query: str, catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    folded = query.casefold()
    for alias, name in _ALIASES.items():
        if alias in folded:
            folded += " " + name.casefold()
    for item in sorted(catalog, key=lambda value: len(str(value["name"])), reverse=True):
        if str(item["name"]).casefold() in folded:
            return item
    return None


async def _arena_get(path: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    error: Exception | None = None
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=6.0),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "tgllmbro/1.0 (+Telegram assistant)"},
    ) as client:
        for base in ARENA_BASES:
            try:
                response = await client.get(base + path, params=params)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("code") not in {0, None}:
                    raise RuntimeError("Arena вернула ошибку")
                return payload
            except Exception as current:
                error = current
    raise RuntimeError(f"Rone Arena недоступна: {error}")


async def _safe_arena_get(
    path: str, params: dict[str, object] | None = None
) -> dict[str, Any]:
    try:
        return await _arena_get(path, params)
    except Exception as error:
        return {"error": str(error)[:300], "data": {"records": []}}


def _records(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    records = data.get("records") if isinstance(data, dict) else None
    return [item for item in records or [] if isinstance(item, dict)]


def _compact(payload: object, limit: int) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"records": []}
    records = [_prune(item) for item in _records(payload)[:limit]]
    data = payload.get("data")
    total = data.get("total") if isinstance(data, dict) else None
    result: dict[str, object] = {"records": records, "total": total}
    if isinstance(payload.get("error"), str):
        result["error"] = payload["error"]
    return result


def _prune(value: object, depth: int = 0) -> object:
    if depth >= 6:
        return "[nested data omitted]"
    if isinstance(value, dict):
        ignored = {
            "head", "smallmap", "icon", "image", "wallpaper", "createdAt",
            "updatedAt", "uin", "vote_all", "form", "hero_channel",
            "main_hero_channel",
        }
        return {
            str(key): _prune(item, depth + 1)
            for key, item in value.items()
            if key not in ignored
        }
    if isinstance(value, list):
        return [_prune(item, depth + 1) for item in value[:8]]
    if isinstance(value, str):
        return value[:500]
    return value
