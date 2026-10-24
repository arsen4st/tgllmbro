from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bot.config import Settings  # noqa: E402
from bot.database import MessageStore  # noqa: E402
from bot.gemini import GeminiAssistant  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_SUFFIXES = {".txt", ".md"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Возобновляемый импорт мемов и примеров живой речи в SQLite."
    )
    parser.add_argument("path", type=Path, help="Файл или папка для импорта")
    parser.add_argument(
        "--mode",
        choices=("auto", "memes", "style", "knowledge"),
        default="auto",
        help="Тип контента; auto определяет его по расширению",
    )
    parser.add_argument(
        "--author",
        action="append",
        default=[],
        help="Автор ответов в Telegram JSON; можно указать несколько раз",
    )
    parser.add_argument("--limit", type=int, default=None, help="Лимит новых записей")
    parser.add_argument(
        "--context-messages",
        type=int,
        default=4,
        help="Сколько предыдущих реплик брать как контекст Telegram-ответа",
    )
    return parser.parse_args()


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from (item for item in path.rglob("*") if item.is_file())


def text_from_telegram(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def style_blocks(text: str) -> Iterable[tuple[str, str]]:
    for block in re.split(r"(?:\r?\n){2,}", text):
        block = block.strip()
        if not block:
            continue
        if re.search(r"\r?\n>>>\r?\n", block):
            context, response = re.split(r"\r?\n>>>\r?\n", block, maxsplit=1)
            if response.strip():
                yield context.strip(), response.strip()
        else:
            yield "", block


def stable_ref(prefix: str, *parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()}"


async def import_image(
    *,
    path: Path,
    store: MessageStore,
    assistant: GeminiAssistant,
) -> bool:
    resolved = path.resolve()
    file_ref = f"local:{resolved}"
    if await store.meme_ref_exists(file_ref):
        return False
    if path.stat().st_size > MAX_IMAGE_BYTES:
        print(f"skip >10 MB: {path}")
        return False
    data = path.read_bytes()
    content_sha256 = hashlib.sha256(data).hexdigest()
    if await store.meme_hash_exists(content_sha256):
        print(f"duplicate: {path}")
        return False
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    description = await assistant.describe_meme(
        data=data,
        mime_type=mime_type,
        file_name=path.name,
    )
    embedding = await assistant.embed_document(description, title=f"meme {path.name}")
    await store.add_meme(
        file_id=file_ref,
        description=description,
        embedding=embedding,
        content_sha256=content_sha256,
    )
    print(f"meme: {path}")
    return True


async def add_style(
    *,
    source_ref: str,
    context: str,
    response: str,
    store: MessageStore,
    assistant: GeminiAssistant,
) -> bool:
    if await store.style_ref_exists(source_ref):
        return False
    searchable = f"контекст: {context}\nответ: {response}".strip()
    embedding = await assistant.embed_document(searchable, title="dialog style example")
    await store.add_style_example(
        source_ref=source_ref,
        context=context,
        response=response,
        embedding=embedding,
    )
    return True


async def import_text_file(
    *,
    path: Path,
    store: MessageStore,
    assistant: GeminiAssistant,
    remaining: int | None,
) -> int:
    text = path.read_text(encoding="utf-8-sig")
    imported = 0
    for context, response in style_blocks(text):
        if remaining is not None and imported >= remaining:
            break
        source_ref = stable_ref("text", context, response)
        if await add_style(
            source_ref=source_ref,
            context=context,
            response=response,
            store=store,
            assistant=assistant,
        ):
            imported += 1
            print(f"style: {path.name} #{imported}")
    return imported


async def import_knowledge_file(
    *,
    path: Path,
    store: MessageStore,
    assistant: GeminiAssistant,
    remaining: int | None,
) -> int:
    text = path.read_text(encoding="utf-8-sig")
    imported = 0
    for index, raw_block in enumerate(re.split(r"(?:\r?\n){2,}", text), start=1):
        block = raw_block.strip()
        if not block or block.startswith("<!--"):
            continue
        if remaining is not None and imported >= remaining:
            break
        lines = block.splitlines()
        title = lines[0].lstrip("# ").strip()
        body_lines = lines[1:]
        source_url = ""
        if body_lines and body_lines[-1].startswith("Источник: "):
            source_url = body_lines.pop().removeprefix("Источник: ").strip()
        body = "\n".join(body_lines).strip()
        if not title or not body:
            continue
        source_ref = stable_ref("knowledge", str(path.resolve()), str(index), title, body)
        if await store.knowledge_ref_exists(source_ref):
            continue
        embedding = await assistant.embed_document(
            f"{title}\n{body}",
            title="MLBB knowledge",
        )
        await store.add_knowledge(
            source_ref=source_ref,
            title=title,
            body=body,
            source_url=source_url,
            embedding=embedding,
        )
        imported += 1
        print(f"knowledge: {path.name} #{index}")
    return imported


async def import_telegram_json(
    *,
    path: Path,
    authors: set[str],
    context_messages: int,
    store: MessageStore,
    assistant: GeminiAssistant,
    remaining: int | None,
) -> int:
    if not authors:
        raise ValueError("для Telegram JSON укажи хотя бы один --author")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        raise ValueError(f"{path}: не похож на Telegram result.json")

    normalized_authors = {author.strip().casefold() for author in authors if author.strip()}
    history: list[tuple[str, str]] = []
    imported = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("type", "message") != "message":
            continue
        text = text_from_telegram(message.get("text"))
        if not text:
            continue
        sender = str(
            message.get("from")
            or message.get("actor")
            or message.get("from_id")
            or "unknown"
        ).strip()
        sender_keys = {
            sender.casefold(),
            str(message.get("from_id") or "").strip().casefold(),
        }
        if normalized_authors.intersection(sender_keys):
            if remaining is not None and imported >= remaining:
                break
            context = "\n".join(
                f"{name}: {body}" for name, body in history[-context_messages:]
            )
            message_id = str(message.get("id", index))
            source_ref = f"telegram:{path.resolve()}:{message_id}"
            if await add_style(
                source_ref=source_ref,
                context=context,
                response=text,
                store=store,
                assistant=assistant,
            ):
                imported += 1
                print(f"telegram style: {path.name} id={message_id}")
        history.append((sender, text[:2000]))
    return imported


async def run(args: argparse.Namespace) -> None:
    source = args.path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit должен быть больше нуля")
    if args.context_messages <= 0:
        raise ValueError("--context-messages должен быть больше нуля")

    settings = Settings.from_env()
    store = MessageStore(settings.database_path)
    assistant = GeminiAssistant(
        api_key=settings.gemini_api_key,
        anymodel_api_key=settings.anymodel_api_key,
        anymodel_model=settings.anymodel_model,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_model=settings.openrouter_model,
        model=settings.gemini_model,
        embedding_model=settings.gemini_embedding_model,
        tts_model=settings.gemini_tts_model,
        tts_voice=settings.gemini_tts_voice,
        max_output_tokens=settings.max_output_tokens,
    )
    imported = 0
    failed = 0
    await store.connect()
    try:
        for path in iter_files(source):
            if args.limit is not None and imported >= args.limit:
                break
            suffix = path.suffix.casefold()
            mode = args.mode
            is_image = suffix in IMAGE_SUFFIXES and mode in {"auto", "memes"}
            is_style = suffix in TEXT_SUFFIXES | {".json"} and mode in {"auto", "style"}
            is_knowledge = suffix in TEXT_SUFFIXES and mode == "knowledge"
            if not is_image and not is_style and not is_knowledge:
                continue
            remaining = None if args.limit is None else args.limit - imported
            try:
                if is_image:
                    imported += int(
                        await import_image(path=path, store=store, assistant=assistant)
                    )
                elif is_knowledge:
                    imported += await import_knowledge_file(
                        path=path,
                        store=store,
                        assistant=assistant,
                        remaining=remaining,
                    )
                elif suffix == ".json":
                    imported += await import_telegram_json(
                        path=path,
                        authors=set(args.author),
                        context_messages=args.context_messages,
                        store=store,
                        assistant=assistant,
                        remaining=remaining,
                    )
                else:
                    imported += await import_text_file(
                        path=path,
                        store=store,
                        assistant=assistant,
                        remaining=remaining,
                    )
            except Exception as error:
                failed += 1
                print(f"error: {path}: {error}", file=sys.stderr)
    finally:
        await assistant.close()
        await store.close()
    print(f"готово: добавлено {imported}, файлов с ошибкой {failed}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
