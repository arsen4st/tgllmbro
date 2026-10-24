from __future__ import annotations

import asyncio
import io
import logging
import math
import random
import re
import secrets
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable, TypeVar

from aiogram import Bot
from aiogram.types import (
    BufferedInputFile,
    FSInputFile,
    ReactionTypeEmoji,
    ReplyParameters,
)
from aiogram.utils.chat_action import ChatActionSender
from pypdf import PdfReader

from bot.batcher import MessageBatch
from bot.database import (
    ChatEvent,
    KnowledgeItem,
    MemeItem,
    MemoryItem,
    MessageStore,
    SoundItem,
    StickerItem,
    StyleExample,
    UserProfileFact,
)
from bot.gemini import AgentAction, GeminiAssistant
from bot.web_reader import read_public_url
from bot.media_search import search_public_media, download_public_media


logger = logging.getLogger(__name__)
TELEGRAM_MESSAGE_LIMIT = 4000
MAX_ATTACHMENT_BYTES = 4_000_000
MAX_ATTACHMENT_TEXT = 18_000
ALLOWED_REACTIONS = {"👍", "👎", "❤", "🔥", "😁", "🤡", "🤔", "😢", "🎉", "🤮", "💩", "👀"}
UNSOLICITED_COOLDOWN_SECONDS = 600.0
BACKGROUND_REACTION_COOLDOWN_SECONDS = 180.0
MIN_INTERJECTION_SCORE = 0.92
CONVERSATION_WINDOW_SECONDS = 75.0
CONVERSATION_FOLLOWUP_MESSAGES = 3
BACKGROUND_BUDGET_WINDOW_SECONDS = 3600.0
ACTIVITY_WINDOW_SECONDS = 60.0
URL_RE = re.compile(r"https?://", re.IGNORECASE)
VOICE_REQUEST_RE = re.compile(r"\b(?:войс|голосом|голосовое|гс)\b", re.IGNORECASE)
BACKGROUND_SIGNAL_RE = re.compile(
    r"(?:[!?]{2,}|\b(?:пиздец|охуеть|нахуя|ебать|заебал|казино|армия)\b)",
    re.IGNORECASE,
)
T = TypeVar("T")


class GroupAgent:
    def __init__(
        self,
        *,
        bot: Bot,
        bot_id: int,
        bot_username: str,
        store: MessageStore,
        assistant: GeminiAssistant,
        recent_events: int,
        memory_results: int,
        style_results: int,
        owner_telegram_id: int = 0,
    ) -> None:
        self._bot = bot
        self._bot_id = bot_id
        self._bot_username = bot_username
        self._store = store
        self._assistant = assistant
        self._recent_events = recent_events
        self._memory_results = memory_results
        self._style_results = style_results
        self._owner_id = owner_telegram_id
        self._public_media: dict[tuple[int, str], tuple[float, dict]] = {}
        self._chat_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_interjection_at: dict[int, float] = {}
        self._last_background_reaction_at: dict[int, float] = {}
        self._activity: defaultdict[int, deque[tuple[float, int]]] = defaultdict(deque)
        self._background_evaluations: defaultdict[int, deque[float]] = defaultdict(deque)
        self._conversation_open_until: dict[int, float] = {}
        self._conversation_followups_left: dict[int, int] = {}
        self._conversation_participants: dict[int, int] = {}

    def claim_conversation_followup(self, chat_id: int, sender_id: int, text: str) -> bool:
        if self._conversation_participants.get(chat_id) != sender_id:
            return False
        now = monotonic()
        if now >= self._conversation_open_until.get(chat_id, 0.0):
            self._conversation_followups_left.pop(chat_id, None)
            return False
        remaining = self._conversation_followups_left.get(chat_id, 0)
        if remaining <= 0:
            return False
        if re.fullmatch(r"(?:(?:ок+|окей|понял[аи]?|спасибо|спс|ясно|ага|ахаха+)[\s,.!👍]*)+", text.strip(), re.IGNORECASE):
            self._conversation_followups_left[chat_id] = 0
            return False
        self._conversation_followups_left[chat_id] = remaining - 1
        return True

    async def process(self, batch: MessageBatch) -> None:
        directed_events = [event for event in batch.events if event.directed_to_bot]
        try:
            async with self._chat_locks[batch.chat_id]:
                if directed_events:
                    async with ChatActionSender.typing(
                        bot=self._bot,
                        chat_id=batch.chat_id,
                    ):
                        await self._process_locked(batch)
                else:
                    await self._process_locked(batch)
        except Exception:
            logger.exception("Failed group turn in chat %s", batch.chat_id)
            if directed_events:
                try:
                    await self._bot.send_message(
                        chat_id=batch.chat_id,
                        text="мозг провайдера опять отвалился. попробуй чуть позже",
                        reply_parameters=_reply_parameters(
                            directed_events[-1].telegram_message_id
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Failed to report group error in chat %s",
                        batch.chat_id,
                    )

    async def _process_locked(self, batch: MessageBatch) -> None:
        directed = any(event.directed_to_bot for event in batch.events)
        continuation = any(event.continuation_candidate for event in batch.events)
        self._record_activity(batch)
        if not directed and not continuation and not self._should_evaluate_background(batch):
            await self._store.append_chat_events(
                chat_id=batch.chat_id,
                events=batch.events,
            )
            logger.info(
                "group_batch skipped_before_llm chat_id=%s messages=%d",
                batch.chat_id,
                len(batch.events),
            )
            return

        recent = await self._store.get_recent_chat_events(
            chat_id=batch.chat_id,
            limit=self._recent_events,
        )
        # Reuse observations so a follow-up about a voice/photo keeps its meaning.
        for index, event in enumerate(recent):
            if event.file_id:
                observed = await self._store.get_attachment_description(batch.chat_id, event.file_id)
                if observed:
                    recent[index] = replace(event, text=event.text + "\n[Ранее прочитанное вложение]\n" + observed[:2000])
        await self._store.append_chat_events(chat_id=batch.chat_id, events=batch.events)
        batch = await self._prepare_context(batch)
        participant_ids = {
            event.sender_id
            for event in batch.events
            if event.role == "user" and event.sender_id != self._bot_id
        }
        profiles = await self._store.get_user_profile_facts(
            chat_id=batch.chat_id,
            user_ids=participant_ids,
        )
        rendered_new_messages = _render_events(batch.events, max_text=5500)
        style_examples = await self._retrieve_style_examples("\n".join(event.text for event in batch.events))
        available_tools = await self._available_tools(batch, rendered_new_messages)

        retrieved_memory_ids: set[int] = set()
        lookup_limits: Counter[str] = Counter()

        async def execute_lookup(action: AgentAction) -> dict[str, object]:
            lookup_limits[action.name] += 1
            if lookup_limits[action.name] > 3:
                return {"ok": False, "error": "Лимит повторных вызовов этого инструмента"}
            return await self._execute_lookup(
                batch=batch,
                action=action,
                retrieved_memory_ids=retrieved_memory_ids,
            )

        actions = await self._assistant.plan_group_actions(
            recent_context=_render_events(recent),
            user_profiles=f"Создатель/владелец: user_id={self._owner_id}. Только этот ID имеет права сохранения.\n" + _render_profiles(profiles),
            style_examples=_render_style_examples(style_examples),
            new_messages=rendered_new_messages,
            available_tools=available_tools,
            tool_executor=execute_lookup,
        )
        await self._execute_actions(batch, actions)
        await self._store.touch_memories(list(retrieved_memory_ids))

    def _should_evaluate_background(self, batch: MessageBatch) -> bool:
        now = monotonic()
        activity = self._activity[batch.chat_id]
        while activity and now - activity[0][0] > ACTIVITY_WINDOW_SECONDS:
            activity.popleft()
        messages_per_minute = len(activity)
        active_people = len({sender_id for _, sender_id in activity})

        evaluations = self._background_evaluations[batch.chat_id]
        while evaluations and now - evaluations[0] > BACKGROUND_BUDGET_WINDOW_SECONDS:
            evaluations.popleft()

        if messages_per_minute >= 12 or active_people >= 4:
            hourly_budget = 3
            chance = 0.48
        elif messages_per_minute >= 6 or active_people >= 3:
            hourly_budget = 2
            chance = 0.30
        else:
            hourly_budget = 1
            chance = 0.14
        if len(evaluations) >= hourly_budget:
            return False

        text = "\n".join(event.text for event in batch.events)
        has_attachment = any(event.file_id for event in batch.events)
        strong_signal = (
            has_attachment
            or len(text) >= 120
            or bool(URL_RE.search(text))
            or bool(BACKGROUND_SIGNAL_RE.search(text))
        )
        if strong_signal:
            chance = max(chance, 0.68)
        if random.random() >= chance:
            return False
        evaluations.append(now)
        logger.info(
            "background_evaluation chat_id=%s chance=%.2f activity=%d people=%d budget=%d/%d",
            batch.chat_id,
            chance,
            messages_per_minute,
            active_people,
            len(evaluations),
            hourly_budget,
        )
        return True

    def _record_activity(self, batch: MessageBatch) -> None:
        now = monotonic()
        activity = self._activity[batch.chat_id]
        for event in batch.events:
            if event.role == "user" and event.sender_id != self._bot_id:
                activity.append((now, event.sender_id))
        while activity and now - activity[0][0] > ACTIVITY_WINDOW_SECONDS:
            activity.popleft()

    def _open_conversation(self, chat_id: int, event: ChatEvent) -> None:
        # A continuation may extend time, but cannot replenish its own budget.
        if not event.directed_to_bot and self._conversation_participants.get(chat_id) == event.sender_id:
            return
        self._conversation_participants[chat_id] = event.sender_id
        self._conversation_open_until[chat_id] = (
            monotonic() + CONVERSATION_WINDOW_SECONDS
        )
        self._conversation_followups_left[chat_id] = CONVERSATION_FOLLOWUP_MESSAGES

    async def _available_tools(
        self,
        batch: MessageBatch,
        rendered_messages: str,
    ) -> set[str]:
        directed = any(event.directed_to_bot for event in batch.events)
        continuation = any(event.continuation_candidate for event in batch.events)
        conversational = directed or continuation
        tools = {
            "stay_silent",
            "react_to_message",
            "remember",
            "update_profile",
            "search_memory",
            "search_archive",
            "search_mlbb_knowledge",
            "search_web",
            "get_creator_history",
        }
        if conversational:
            tools.update({"reply_to_message", "search_media", "search_public_media", "send_media"})
            if any(event.sender_id == self._owner_id and _owner_save_request(event.text) for event in batch.events):
                tools.add("save_media")
        else:
            tools.update({"interject_to_message", "interject_to_chat"})

        if any(event.file_id or (event.reply_attachment and event.reply_attachment.file_id) for event in batch.events):
            tools.add("read_attachment")
        if URL_RE.search(rendered_messages):
            tools.add("read_url")
        if conversational and VOICE_REQUEST_RE.search(rendered_messages):
            tools.add("speak_to_message")

        if conversational:
            has_memes, has_sounds, has_stickers = await asyncio.gather(
                self._store.has_memes(),
                self._store.has_sounds(),
                self._store.has_stickers(),
            )
            if has_memes:
                tools.update({"search_memes", "send_meme"})
            if has_sounds:
                tools.update({"search_sounds", "send_sound"})
            if has_stickers:
                tools.update({"search_stickers", "send_sticker"})
        return tools

    async def add_meme(self, *, file_id: str, description: str) -> None:
        embedding = await self._assistant.embed_document(description, title="telegram meme")
        await self._store.add_meme(
            file_id=file_id,
            description=description,
            embedding=embedding,
        )

    async def add_sticker(
        self,
        *,
        file_id: str,
        description: str,
        emoji: str | None,
    ) -> None:
        searchable = f"{emoji or ''} {description}".strip()
        embedding = await self._assistant.embed_document(
            searchable,
            title="telegram sticker",
        )
        await self._store.add_sticker(
            file_id=file_id,
            description=description,
            emoji=emoji,
            embedding=embedding,
        )

    async def add_sound(
        self,
        *,
        file_id: str,
        description: str,
        media_kind: str,
    ) -> None:
        embedding = await self._assistant.embed_document(
            description,
            title="telegram sound meme",
        )
        await self._store.add_sound(
            file_id=file_id,
            description=description,
            media_kind=media_kind,
            embedding=embedding,
        )

    async def add_mlbb_knowledge(
        self,
        *,
        source_ref: str,
        title: str,
        body: str,
        source_url: str = "",
    ) -> None:
        searchable = f"{title}\n{body}"
        embedding = await self._assistant.embed_document(
            searchable,
            title="MLBB knowledge",
        )
        await self._store.add_knowledge(
            source_ref=source_ref,
            title=title,
            body=body,
            source_url=source_url,
            embedding=embedding,
        )

    async def read_attachment(self, event: ChatEvent) -> dict[str, object]:
        return await self._read_attachment(event)

    async def handle_owner_command(self, chat_id: int, event: ChatEvent) -> bool:
        if not self._owner_id or event.sender_id != self._owner_id:
            return False
        command = _owner_command_text(event.text)
        if _owner_save_request(event.text):
            result = await self.save_owner_media(chat_id, event)
            text = str(result.get("message") or result.get("error"))
        elif command.startswith("запомни обо мне"):
            payload = command.removeprefix("запомни обо мне").lstrip(" :,")
            public = payload.startswith("для всех:")
            if public:
                payload = payload.removeprefix("для всех:").strip()
            key, separator, value = payload.partition("=")
            if not separator:
                key, value = payload[:60], payload
            if not value.strip():
                text = "напиши факт: «запомни обо мне: любимая игра = MLBB». для общей памяти добавь «для всех:»"
            else:
                await self._store.set_creator_note(key=key.strip(), value=value.strip(), public=public)
                text = "запомнил для всех чатов" if public else "запомнил, эта запись доступна только в твоей личке"
        else:
            return False
        await self._store.append_chat_events(chat_id=chat_id, events=[event])
        await self._send_reply(chat_id, event.telegram_message_id, text)
        return True

    async def save_owner_media(self, chat_id: int, event: ChatEvent, description: str = "") -> dict[str, object]:
        if not self._owner_id or event.sender_id != self._owner_id or not _owner_save_request(event.text):
            return {"ok": False, "error": "Сохранение требует явной команды настоящего владельца"}
        source = event.reply_attachment or event
        if not source.media_file_id or not source.media_kind:
            return {"ok": False, "error": "ответь этой командой на GIF, стикер, фото, видео, кружок, голосовое, аудио или файл"}
        shared = bool(re.search(r"\b(?:для всех|в общую базу)\b", event.text, re.I))
        label = re.search(r"(?:название|назови|описание|под названием)\s*[:=—-]?\s*(.+)", event.text, re.I | re.S)
        description = (description or (label.group(1) if label else "")).strip()
        if not description:
            try:
                inspected = await self._cached_attachment(chat_id, source)
                description = str(inspected.get("text") or "")[:1500]
            except Exception:
                logger.warning("media library description unavailable")
        description = description or source.text or source.media_kind
        media_id = await self._store.save_media(scope_chat=0 if shared else chat_id,
            kind=source.media_kind, file_id=source.media_file_id, unique_id=source.media_unique_id,
            description=description, source_chat=chat_id, source_message=source.telegram_message_id)
        scope = "для всех чатов" if shared else "для этого чата"
        return {"ok": True, "media_id": media_id, "message": f"сохранил {scope}: #{media_id} — {description[:180]}"}

    async def import_sticker_pack(self, chat_id: int, sender_id: int, name: str) -> dict:
        if not self._owner_id or sender_id != self._owner_id:
            return {"ok": False, "error": "Только владелец может добавлять наборы"}
        name = name.removeprefix("https://t.me/addstickers/").strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
            return {"ok": False, "error": "Формат: /pack название_набора или ссылка t.me/addstickers/..."}
        try:
            pack = await self._bot.get_sticker_set(name)
            for item in pack.stickers[:120]:
                await self._store.save_media(scope_chat=chat_id, kind="sticker", file_id=item.file_id,
                    unique_id=item.file_unique_id, description=f"{pack.title} {item.emoji or ''} стикер из {name}",
                    source_chat=chat_id, source_message=0)
            return {"ok": True, "message": f"добавил {min(len(pack.stickers), 120)} стикеров из {pack.title} для этого чата. описания пока по названию набора и эмодзи"}
        except Exception:
            logger.exception("Sticker pack import failed")
            return {"ok": False, "error": "не получилось загрузить набор. проверь название и попробуй позже"}

    async def _cached_attachment(self, chat_id: int, event: ChatEvent) -> dict[str, object]:
        if event.file_id:
            cached = await self._store.get_attachment_description(chat_id=chat_id, file_id=event.file_id)
            if cached:
                return {"ok": True, "text": cached, "cached": True}
        result = await self._read_attachment(event)
        if result.get("ok") and event.file_id:
            await self._store.cache_attachment(chat_id=chat_id, file_id=event.file_id, description=str(result["text"]))
        return result

    async def _prepare_context(self, batch: MessageBatch) -> MessageBatch:
        events = []
        media_budget = 2
        for event in sorted(batch.events, key=lambda item: item.telegram_message_id):
            parent = event.reply_attachment
            if parent is None and event.reply_to_message_id is not None:
                parent = await self._store.get_chat_event(chat_id=batch.chat_id, message_id=event.reply_to_message_id)
            if parent:
                event = replace(event, reply_attachment=parent, reply_text=event.reply_text or parent.text,
                                reply_sender=event.reply_sender or parent.sender_name)
            if event.directed_to_bot or event.continuation_candidate:
                attachment = event if event.file_id else parent
                if attachment and attachment.file_id and media_budget:
                    media_budget -= 1
                    try:
                        result = await self._cached_attachment(batch.chat_id, attachment)
                    except Exception as error:
                        logger.warning("media inspection failed type=%s", type(error).__name__)
                        result = {"ok": False, "error": "сервис чтения вложения недоступен"}
                    label = "Вложение цитаты" if attachment is parent else "Вложение сообщения"
                    observed = str(result.get("text") or result.get("error", "не прочитано"))[:4000]
                    event = replace(event, text=event.text + f"\n[{label}; прочитано={bool(result.get('ok'))}]\n" + observed)
            events.append(event)
        return replace(batch, events=events)

    async def _execute_lookup(
        self,
        *,
        batch: MessageBatch,
        action: AgentAction,
        retrieved_memory_ids: set[int],
    ) -> dict[str, object]:
        if action.name == "get_creator_history":
            history = Path(__file__).resolve().parent.parent / "data" / "creator_history.md"
            notes = await self._store.get_creator_notes(private=batch.chat_id == self._owner_id)
            return {"ok": True, "history": history.read_text(encoding="utf-8")[:4000], "notes": notes,
                    "owner_user_id": self._owner_id}
        if action.name == "save_media":
            target = _target_id(action, {event.telegram_message_id for event in batch.events})
            event = next((event for event in batch.events if event.telegram_message_id == target), None)
            if event is None:
                return {"ok": False, "error": "Нет такой команды в текущей пачке"}
            return await self.save_owner_media(batch.chat_id, event, str(action.arguments.get("description", ""))[:1500])
        if action.name == "search_media":
            results = await self._store.search_media(chat_id=batch.chat_id, query=_required_text(action, "query"),
                                                    kind=str(action.arguments.get("kind", "any")))
            return {"ok": True, "results": results}
        if action.name == "search_public_media":
            items = await search_public_media(_required_text(action, "query"), str(action.arguments.get("kind", "animation")))
            now = monotonic()
            self._public_media = {key: value for key, value in self._public_media.items() if now - value[0] < 600}
            results = []
            for item in items:
                token = secrets.token_urlsafe(9)
                self._public_media[(batch.chat_id, token)] = (now, item)
                results.append({"web_id": token, "kind": item["kind"], "description": item["description"], "source": item["source"]})
            return {"ok": True, "results": results, "note": "Это описания источника, а не просмотр содержимого моделью"}
        if action.name == "search_memory":
            query = _required_text(action, "query")
            limit = _bounded_int(action.arguments.get("limit"), 1, 10, self._memory_results)
            memories = await self._retrieve_memories(batch.chat_id, query, limit)
            retrieved_memory_ids.update(item.id for item in memories)
            return {
                "ok": True,
                "results": [
                    {"memory_id": item.id, "text": item.text, "importance": item.importance}
                    for item in memories
                ],
            }

        if action.name == "search_archive":
            query = _required_text(action, "query")
            limit = _bounded_int(action.arguments.get("limit"), 1, 12, 8)
            events = await self._store.search_chat_events(
                chat_id=batch.chat_id,
                query=query,
                limit=limit,
            )
            return {
                "ok": True,
                "results": [
                    {
                        "message_id": event.telegram_message_id,
                        "sender": event.sender_name,
                        "role": event.role,
                        "created_at": event.created_at,
                        "text": event.text[:1500],
                    }
                    for event in events
                ],
            }

        if action.name == "search_memes":
            query = _required_text(action, "query")
            limit = _bounded_int(action.arguments.get("limit"), 1, 6, 4)
            memes = await self._find_memes(query, limit)
            return {
                "ok": True,
                "results": [
                    {"meme_id": item.id, "description": item.description}
                    for item in memes
                ],
            }

        if action.name == "search_sounds":
            query = _required_text(action, "query")
            limit = _bounded_int(action.arguments.get("limit"), 1, 6, 4)
            sounds = await self._find_sounds(query, limit)
            return {
                "ok": True,
                "results": [
                    {
                        "sound_id": item.id,
                        "description": item.description,
                        "media_kind": item.media_kind,
                    }
                    for item in sounds
                ],
            }

        if action.name == "search_stickers":
            query = _required_text(action, "query")
            limit = _bounded_int(action.arguments.get("limit"), 1, 6, 4)
            stickers = await self._find_stickers(query, limit)
            return {
                "ok": True,
                "results": [
                    {
                        "sticker_id": item.id,
                        "description": item.description,
                        "emoji": item.emoji,
                    }
                    for item in stickers
                ],
            }

        if action.name == "search_mlbb_knowledge":
            query = _required_text(action, "query")
            limit = _bounded_int(action.arguments.get("limit"), 1, 8, 5)
            items = await self._find_knowledge(query, limit)
            return {
                "ok": True,
                "results": [
                    {
                        "title": item.title,
                        "body": item.body[:3000],
                        "source_url": item.source_url,
                    }
                    for item in items
                ],
            }

        if action.name == "search_web":
            query = _required_text(action, "query")
            return await self._assistant.search_web(query)

        if action.name == "read_url":
            return await read_public_url(_required_text(action, "url"))

        if action.name == "read_attachment":
            target = _target_id(action, {event.telegram_message_id for event in batch.events})
            if target is None:
                return {"ok": False, "error": "Такого id нет в новых сообщениях"}
            event = next(
                event for event in batch.events if event.telegram_message_id == target
            )
            if action.arguments.get("source") == "reply":
                if event.reply_attachment is None:
                    return {"ok": False, "error": "У цитаты нет доступного вложения"}
                event = event.reply_attachment
            result = await self._cached_attachment(batch.chat_id, event)
            if result.get("ok"):
                return {**result, "text": str(result.get("text", ""))[:6000]}
            return result

        return {"ok": False, "error": "Неизвестный поисковый инструмент"}

    async def _retrieve_memories(
        self,
        chat_id: int,
        query: str,
        limit: int,
    ) -> list[MemoryItem]:
        candidates = await self._store.get_memory_candidates(chat_id=chat_id)
        if not candidates:
            return []
        embedding, fts_ids = await asyncio.gather(
            self._assistant.embed_query(query),
            self._store.search_memory_ids(chat_id=chat_id, query=query),
        )
        return _hybrid_rank(
            query=query,
            query_embedding=embedding,
            items=candidates,
            text_getter=lambda item: item.text,
            embedding_getter=lambda item: item.embedding,
            importance_getter=lambda item: item.importance,
            limit=limit,
            fts_ids=fts_ids,
        )

    async def _find_memes(self, query: str, limit: int) -> list[MemeItem]:
        fts_ids = set(await self._store.search_meme_ids(query=query))
        candidates = await self._store.get_meme_candidates(
            meme_ids=list(fts_ids) or None,
            limit=1000,
        )
        if not candidates:
            return []
        embedding = await self._assistant.embed_query(query)
        return _hybrid_rank(
            query=query,
            query_embedding=embedding,
            items=candidates,
            text_getter=lambda item: item.description,
            embedding_getter=lambda item: item.embedding,
            importance_getter=lambda item: 0.5,
            limit=limit,
            fts_ids=fts_ids,
        )

    async def _find_sounds(self, query: str, limit: int) -> list[SoundItem]:
        fts_ids = set(await self._store.search_sound_ids(query=query))
        candidates = await self._store.get_sound_candidates(
            sound_ids=list(fts_ids) or None,
            limit=1000,
        )
        if not candidates:
            return []
        embedding = await self._assistant.embed_query(query)
        return _hybrid_rank(
            query=query,
            query_embedding=embedding,
            items=candidates,
            text_getter=lambda item: item.description,
            embedding_getter=lambda item: item.embedding,
            importance_getter=lambda item: 0.5,
            limit=limit,
            fts_ids=fts_ids,
        )

    async def _find_knowledge(self, query: str, limit: int) -> list[KnowledgeItem]:
        fts_ids = set(await self._store.search_knowledge_ids(query=query))
        candidates = await self._store.get_knowledge_candidates(
            item_ids=list(fts_ids) or None,
            limit=1000,
        )
        if not candidates:
            return []
        embedding = await self._assistant.embed_query(query)
        return _hybrid_rank(
            query=query,
            query_embedding=embedding,
            items=candidates,
            text_getter=lambda item: f"{item.title}\n{item.body}",
            embedding_getter=lambda item: item.embedding,
            importance_getter=lambda item: 0.7,
            limit=limit,
            fts_ids=fts_ids,
        )

    async def _retrieve_style_examples(self, query: str) -> list[StyleExample]:
        if not await self._store.has_style_examples():
            return []
        fts_ids = set(await self._store.search_style_ids(query=query))
        if not fts_ids:
            return []
        candidates = await self._store.get_style_candidates(
            style_ids=list(fts_ids) or None,
            limit=1000,
        )
        if not candidates:
            return []
        return _hybrid_rank(
            query=query,
            query_embedding=[],
            items=candidates,
            text_getter=lambda item: f"{item.context}\n{item.response}",
            embedding_getter=lambda item: item.embedding,
            importance_getter=lambda item: 0.5,
            limit=min(self._style_results, 2),
            fts_ids=fts_ids,
        )

    async def _find_stickers(self, query: str, limit: int) -> list[StickerItem]:
        candidates = await self._store.get_sticker_candidates()
        if not candidates:
            return []
        embedding = await self._assistant.embed_query(query)
        return _hybrid_rank(
            query=query,
            query_embedding=embedding,
            items=candidates,
            text_getter=lambda item: f"{item.emoji or ''} {item.description}",
            embedding_getter=lambda item: item.embedding,
            importance_getter=lambda item: 0.5,
            limit=limit,
        )

    async def _read_attachment(self, event: ChatEvent) -> dict[str, object]:
        if not event.file_id:
            return {"ok": False, "error": "В сообщении нет читаемого вложения"}
        if event.file_size and event.file_size > MAX_ATTACHMENT_BYTES:
            return {"ok": False, "error": "Файл больше лимита 4 МБ"}
        telegram_file = await self._bot.get_file(event.file_id)
        if not telegram_file.file_path:
            return {"ok": False, "error": "Telegram не отдал путь к файлу"}
        buffer = io.BytesIO()
        await self._bot.download_file(telegram_file.file_path, destination=buffer)
        data = buffer.getvalue()
        if len(data) > MAX_ATTACHMENT_BYTES:
            return {"ok": False, "error": "Файл больше лимита 4 МБ"}

        file_name = event.file_name or "attachment"
        mime_type = (event.mime_type or "").lower()
        if mime_type == "application/pdf" or file_name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        elif mime_type.startswith("text/") or file_name.lower().endswith(
            (".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".py", ".js", ".ts")
        ):
            text = _decode_text(data)
        elif mime_type.startswith(("image/", "audio/", "video/")):
            text = await self._assistant.inspect_media(
                data=data,
                mime_type=mime_type,
                file_name=file_name,
            )
        else:
            return {
                "ok": False,
                "error": "Пока читаю текст, код, JSON/CSV/XML/YAML, PDF, изображения, аудио и видео",
            }
        text = text.strip()
        if file_name == "sticker_preview.webp":
            text = "[Только статичное превью TGS-стикера; анимация не просмотрена.]\n" + text
        return {
            "ok": True,
            "file_name": file_name,
            "text": text[:MAX_ATTACHMENT_TEXT],
            "truncated": len(text) > MAX_ATTACHMENT_TEXT,
        }

    async def _execute_actions(
        self,
        batch: MessageBatch,
        actions: list[AgentAction],
    ) -> None:
        valid_targets = {event.telegram_message_id for event in batch.events}
        directed_targets = {
            event.telegram_message_id
            for event in batch.events
            if event.directed_to_bot
        }
        continuation_targets = {
            event.telegram_message_id
            for event in batch.events
            if event.continuation_candidate
        }
        conversational_targets = directed_targets | continuation_targets
        limits: Counter[str] = Counter()
        replied_targets: set[int] = set()
        for action in actions:
            sent_before = sum(limits[key] for key in ("reply", "interject", "meme", "sticker", "sound", "voice"))
            try:
                if action.name == "reply_to_message" and limits["reply"] < 2:
                    target = _target_id(action, conversational_targets)
                    text = str(action.arguments.get("text", "")).strip()
                    if target is not None and target not in replied_targets and text:
                        async with ChatActionSender.typing(
                            bot=self._bot,
                            chat_id=batch.chat_id,
                        ):
                            await self._send_reply(batch.chat_id, target, text)
                        limits["reply"] += 1
                        replied_targets.add(target)

                elif (
                    action.name == "interject_to_message"
                    and limits["interject"] < 1
                    and not conversational_targets
                ):
                    target = _target_id(action, valid_targets)
                    text = str(action.arguments.get("text", "")).strip()
                    score = float(action.arguments.get("score", 0.0))
                    last_at = self._last_interjection_at.get(batch.chat_id, 0.0)
                    if (
                        target is not None
                        and text
                        and score >= MIN_INTERJECTION_SCORE
                        and monotonic() - last_at >= UNSOLICITED_COOLDOWN_SECONDS
                    ):
                        async with ChatActionSender.typing(
                            bot=self._bot,
                            chat_id=batch.chat_id,
                        ):
                            await self._send_reply(batch.chat_id, target, text)
                        self._last_interjection_at[batch.chat_id] = monotonic()
                        limits["interject"] += 1

                elif (
                    action.name == "interject_to_chat"
                    and limits["interject"] < 1
                    and not conversational_targets
                ):
                    text = str(action.arguments.get("text", "")).strip()
                    score = float(action.arguments.get("score", 0.0))
                    last_at = self._last_interjection_at.get(batch.chat_id, 0.0)
                    if (
                        text
                        and score >= MIN_INTERJECTION_SCORE
                        and monotonic() - last_at >= UNSOLICITED_COOLDOWN_SECONDS
                    ):
                        async with ChatActionSender.typing(
                            bot=self._bot,
                            chat_id=batch.chat_id,
                        ):
                            for chunk in _split_text(text):
                                sent = await self._bot.send_message(
                                    chat_id=batch.chat_id,
                                    text=chunk,
                                )
                                await self._record_bot_message(
                                    sent.message_id,
                                    batch.chat_id,
                                    chunk,
                                )
                        self._last_interjection_at[batch.chat_id] = monotonic()
                        limits["interject"] += 1

                elif action.name == "react_to_message" and limits["reaction"] < 2:
                    target = _target_id(action, valid_targets)
                    emoji = str(action.arguments.get("emoji", ""))
                    background_allowed = (
                        target in conversational_targets
                        or (
                            limits["background_reaction"] < 1
                            and monotonic()
                            - self._last_background_reaction_at.get(batch.chat_id, 0.0)
                            >= BACKGROUND_REACTION_COOLDOWN_SECONDS
                        )
                    )
                    if (
                        target is not None
                        and emoji in ALLOWED_REACTIONS
                        and background_allowed
                    ):
                        await self._bot.set_message_reaction(
                            chat_id=batch.chat_id,
                            message_id=target,
                            reaction=[ReactionTypeEmoji(emoji=emoji)],
                        )
                        limits["reaction"] += 1
                        if target not in conversational_targets:
                            limits["background_reaction"] += 1
                            self._last_background_reaction_at[batch.chat_id] = monotonic()

                elif action.name == "send_meme" and limits["meme"] < 1:
                    target = _target_id(action, conversational_targets)
                    meme_id = _optional_int(action.arguments.get("meme_id"))
                    caption = str(action.arguments.get("caption", "")).strip()[:1000]
                    meme = await self._store.get_meme(meme_id) if meme_id else None
                    if target is not None and meme is not None:
                        sent = await self._bot.send_photo(
                            chat_id=batch.chat_id,
                            photo=_meme_photo(meme.file_id),
                            caption=caption or None,
                            reply_parameters=_reply_parameters(target),
                        )
                        await self._record_bot_message(
                            sent.message_id,
                            batch.chat_id,
                            caption or f"[мем: {meme.description}]",
                            target=target,
                        )
                        limits["meme"] += 1

                elif action.name == "send_sticker" and limits["sticker"] < 1:
                    target = _target_id(action, conversational_targets)
                    sticker_id = _optional_int(action.arguments.get("sticker_id"))
                    sticker = (
                        await self._store.get_sticker(sticker_id) if sticker_id else None
                    )
                    if target is not None and sticker is not None:
                        sent = await self._bot.send_sticker(
                            chat_id=batch.chat_id,
                            sticker=sticker.file_id,
                            reply_parameters=_reply_parameters(target),
                        )
                        await self._record_bot_message(
                            sent.message_id,
                            batch.chat_id,
                            f"[стикер: {sticker.emoji or ''} {sticker.description}]",
                            target=target,
                        )
                        limits["sticker"] += 1

                elif action.name == "send_sound" and limits["sound"] < 1:
                    target = _target_id(action, conversational_targets)
                    sound_id = _optional_int(action.arguments.get("sound_id"))
                    caption = str(action.arguments.get("caption", "")).strip()[:1000]
                    sound = await self._store.get_sound(sound_id) if sound_id else None
                    if target is not None and sound is not None:
                        if sound.media_kind == "voice":
                            sent = await self._bot.send_voice(
                                chat_id=batch.chat_id,
                                voice=sound.file_id,
                                caption=caption or None,
                                reply_parameters=_reply_parameters(target),
                            )
                        else:
                            sent = await self._bot.send_audio(
                                chat_id=batch.chat_id,
                                audio=sound.file_id,
                                caption=caption or None,
                                reply_parameters=_reply_parameters(target),
                            )
                        await self._record_bot_message(
                            sent.message_id,
                            batch.chat_id,
                            caption or f"[звуковой мем: {sound.description}]",
                            target=target,
                        )
                        limits["sound"] += 1

                elif action.name == "speak_to_message" and limits["voice"] < 1:
                    target = _target_id(action, conversational_targets)
                    text = str(action.arguments.get("text", "")).strip()[:700]
                    if target is not None and text:
                        async with ChatActionSender.record_voice(
                            bot=self._bot,
                            chat_id=batch.chat_id,
                        ):
                            try:
                                audio = await self._assistant.synthesize_speech(text)
                                ogg = await _wav_to_ogg(audio)
                            except Exception:
                                logger.exception("Voice generation failed; sending text")
                                await self._send_reply(batch.chat_id, target, text + "\n\nголос сейчас не собрался, отправляю текстом")
                                limits["voice"] += 1
                                event = next(item for item in batch.events if item.telegram_message_id == target)
                                self._open_conversation(batch.chat_id, event)
                                continue
                            sent = await self._bot.send_voice(
                                chat_id=batch.chat_id,
                                voice=BufferedInputFile(ogg, filename="sych.ogg"),
                                reply_parameters=_reply_parameters(target),
                            )
                        await self._record_bot_message(
                            sent.message_id,
                            batch.chat_id,
                            f"[голосовое] {text}",
                            target=target,
                        )
                        limits["voice"] += 1

                elif action.name == "remember" and limits["memory"] < 2:
                    fact = str(action.arguments.get("fact", "")).strip()[:1000]
                    importance = float(action.arguments.get("importance", 0.5))
                    if fact:
                        embedding = await self._assistant.embed_document(
                            fact,
                            title="group memory",
                        )
                        await self._store.add_memory(
                            chat_id=batch.chat_id,
                            text=fact,
                            embedding=embedding,
                            importance=max(0.0, min(1.0, importance)),
                        )
                        limits["memory"] += 1

                elif action.name == "update_profile" and limits["profile"] < 3:
                    user_id = _optional_int(action.arguments.get("user_id"))
                    key = str(action.arguments.get("key", "")).strip().casefold()[:80]
                    value = str(action.arguments.get("value", "")).strip()[:500]
                    names = {
                        event.sender_id: event.sender_name
                        for event in batch.events
                        if event.role == "user" and event.sender_id != self._bot_id
                    }
                    if user_id in names and key and value:
                        await self._store.upsert_user_profile_fact(
                            chat_id=batch.chat_id,
                            user_id=user_id,
                            display_name=names[user_id],
                            key=key,
                            value=value,
                        )
                        limits["profile"] += 1
                sent_after = sum(limits[key] for key in ("reply", "interject", "meme", "sticker", "sound", "voice"))
                if sent_after > sent_before:
                    anchor = _target_id(action, valid_targets)
                    event = next((item for item in batch.events if item.telegram_message_id == anchor), batch.events[-1])
                    self._open_conversation(batch.chat_id, event)
            except Exception:
                logger.exception(
                    "Failed group action %s in chat %s",
                    action.name,
                    batch.chat_id,
                )

    async def _send_reply(self, chat_id: int, target: int, text: str) -> None:
        for index, chunk in enumerate(_split_text(text)):
            sent = await self._bot.send_message(
                chat_id=chat_id,
                text=chunk,
                reply_parameters=_reply_parameters(target) if index == 0 else None,
            )
            await self._record_bot_message(sent.message_id, chat_id, chunk, target=target)

    async def _record_bot_message(self, message_id: int, chat_id: int, text: str, *, target: int | None = None) -> None:
        parent = await self._store.get_chat_event(chat_id=chat_id, message_id=target) if target is not None else None
        await self._store.append_chat_events(
            chat_id=chat_id,
            events=[
                ChatEvent(
                    telegram_message_id=message_id,
                    sender_id=self._bot_id,
                    sender_name=f"@{self._bot_username}",
                    text=text,
                    role="bot",
                    reply_to_message_id=target,
                    reply_text=parent.text[:2000] if parent else "",
                    reply_sender=parent.sender_name if parent else "",
                )
            ],
        )


def _owner_command_text(text: str) -> str:
    text = re.sub(r"^\[[^\]\n]+\]\s*", "", text.strip())
    return re.sub(r"^(?:@[\w]+|лисун(?:\s+син)?|yss)[\s,:]*", "", text, flags=re.I).strip().casefold()


def _owner_save_request(text: str) -> bool:
    return bool(re.match(r"^(?:сохрани|сохраняй|добавь\s+в\s+(?:базу|коллекцию|медиатеку))\b", _owner_command_text(text)))


def _render_events(events: Iterable[ChatEvent], *, max_text: int = 1500) -> str:
    lines = []
    for event in events:
        directed = " directed_to=bot" if event.directed_to_bot else ""
        continuation = (
            " continuation_candidate=bot" if event.continuation_candidate else ""
        )
        reply = (
            f" reply_to={event.reply_to_message_id}"
            if event.reply_to_message_id is not None
            else ""
        )
        attachment = f" file={event.file_name or event.mime_type}" if event.file_id else ""
        text = event.text[:max_text]
        lines.append(
            f"[id={event.telegram_message_id} role={event.role} user_id={event.sender_id}"
            f"{directed}{continuation}{reply}{attachment}] "
            f"{event.sender_name}: {text}"
        )
        if event.reply_text:
            quoted_file = " file_in_reply=true" if event.reply_attachment and event.reply_attachment.file_id else ""
            lines.append(f"  [цитата reply{quoted_file}] {event.reply_sender}: {event.reply_text[:2000]}")
    return "\n".join(lines)


def _render_profiles(facts: Iterable[UserProfileFact]) -> str:
    return "\n".join(
        f"[user_id={fact.user_id}] {fact.display_name}: {fact.key} = {fact.value}"
        for fact in facts
    )


def _render_style_examples(examples: Iterable[StyleExample]) -> str:
    blocks = []
    for index, example in enumerate(examples, start=1):
        context = example.context.replace("\n", " ").strip()[:1200]
        response = example.response.replace("\n", " ").strip()[:1200]
        blocks.append(f"{index}. контекст: {context or '(нет)'}\nответ: {response}")
    return "\n\n".join(blocks)


def _required_text(action: AgentAction, key: str) -> str:
    value = str(action.arguments.get(key, "")).strip()
    if not value:
        raise ValueError(f"Пустой параметр {key}")
    return value


def _target_id(action: AgentAction, valid_targets: set[int]) -> int | None:
    target = _optional_int(action.arguments.get("target_message_id"))
    return target if target in valid_targets else None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _bounded_int(value: object, minimum: int, maximum: int, default: int) -> int:
    parsed = _optional_int(value)
    return max(minimum, min(maximum, parsed if parsed is not None else default))


def _reply_parameters(target: int) -> ReplyParameters:
    return ReplyParameters(message_id=target, allow_sending_without_reply=True)


def _meme_photo(file_ref: str) -> str | FSInputFile:
    if file_ref.startswith("local:"):
        return FSInputFile(Path(file_ref.removeprefix("local:")))
    return file_ref


def _hybrid_rank(
    *,
    query: str,
    query_embedding: list[float],
    items: list[T],
    text_getter: Callable[[T], str],
    embedding_getter: Callable[[T], list[float]],
    importance_getter: Callable[[T], float],
    limit: int,
    fts_ids: set[int] | None = None,
) -> list[T]:
    query_tokens = _tokens(query)
    scored: list[tuple[float, T]] = []
    for item in items:
        semantic = _cosine(query_embedding, embedding_getter(item))
        lexical = _lexical_score(query_tokens, _tokens(text_getter(item)))
        importance = float(importance_getter(item))
        fts_bonus = 0.1 if fts_ids and getattr(item, "id", None) in fts_ids else 0.0
        scored.append(
            (0.62 * semantic + 0.23 * lexical + 0.05 * importance + fts_bonus, item)
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:limit] if score > 0.12]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w-]{2,}", text.casefold()))


def _lexical_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


async def _wav_to_ogg(audio: bytes) -> bytes:
    from bot.media import run_media_command

    return await run_media_command(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-c:a", "libopus", "-b:a", "48k", "-f", "ogg", "pipe:1",
        data=audio,
    )


def _split_text(text: str) -> list[str]:
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= TELEGRAM_MESSAGE_LIMIT:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        if split_at < TELEGRAM_MESSAGE_LIMIT // 2:
            split_at = remaining.rfind(" ", 0, TELEGRAM_MESSAGE_LIMIT)
        if split_at <= 0:
            split_at = TELEGRAM_MESSAGE_LIMIT
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks
