from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import replace

from aiogram import F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import ErrorEvent, Message
from aiogram.utils.chat_action import ChatActionSender

from bot.batcher import GroupBatcher, MessageBatch
from bot.database import ChatEvent, MessageStore
from bot.gemini import GeminiAssistant
from bot.group_agent import GroupAgent


logger = logging.getLogger(__name__)
BOT_NAME_RE = re.compile(r"^(?:лисун(?:\s+син)?|лисунсин|yss|лсс)(?=$|[\s,!?\.:])", re.IGNORECASE)
TELEGRAM_MESSAGE_LIMIT = 4000
WEB_SEARCH_REQUEST_RE = re.compile(
    r"(?:\b(?:погугли|загугли|поищи|найди\s+в\s+интернете|свежие\s+новости|"
    r"что\s+сейчас|актуальн\w*|последн\w*\s+новост\w*)\b|https?://)",
    re.IGNORECASE,
)


def build_router(
    *,
    store: MessageStore,
    assistant: GeminiAssistant,
    group_agent: GroupAgent,
    group_batcher: GroupBatcher,
    history_messages: int,
    owner_telegram_id: int,
    bot_id: int,
    bot_username: str,
) -> Router:
    router = Router()
    private_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def private_access_allowed(message: Message) -> bool:
        return (
            message.chat.type != ChatType.PRIVATE
            or (
                message.from_user is not None
                and message.from_user.id == owner_telegram_id
            )
        )

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        if not private_access_allowed(message):
            return
        await message.answer(
            "Я Лисун Син. В личке работаю только с владельцем, а в группах читаю "
            "поток, собираю сообщения в пачки и вмешиваюсь по ситуации.\n\n"
            "/help — справка, /reset — стереть память текущего чата."
        )

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        if not private_access_allowed(message):
            return
        await message.answer(
            "В личке отвечаю только владельцу. В группе сообщения собираются "
            "в один батч; для гарантированного ответа упомяни меня через "
            f"@{bot_username} или ответь на моё сообщение.\n"
            "/reset — владелец удаляет историю и память текущего чата.\n"
            "/addmeme описание — ответом на картинку добавить мем в базу.\n"
            "/addsticker описание — ответом на стикер добавить его в базу.\n"
            "/addsound описание — ответом на войс или аудио добавить звуковой мем.\n"
            "/addmlbb заголовок | факт | ссылка — добавить проверенный факт по MLBB."
        )

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        if message.from_user is None or message.from_user.id != owner_telegram_id:
            return
        context_id = _context_id(message, owner_telegram_id)
        await store.clear_history(chat_id=message.chat.id, user_id=context_id)
        await message.answer("Контекст и долговременная память этого чата стёрты.")

    @router.message(Command("addmeme"))
    async def add_meme(message: Message) -> None:
        if message.from_user is None or message.from_user.id != owner_telegram_id:
            return
        source = message.reply_to_message
        if source is None or not source.photo:
            await message.answer(
                "Ответь командой /addmeme описание на сообщение с картинкой."
            )
            return
        description = (message.text or "").partition(" ")[2].strip()
        if not description:
            description = (source.caption or "").strip()
        if not description:
            description = await _describe_for_library(message, source, group_agent)
            if not description:
                return
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await group_agent.add_meme(
                file_id=source.photo[-1].file_id,
                description=description,
            )
        await message.answer("Мем добавлен в базу. Теперь есть чем позориться.")

    @router.message(Command("addsticker"))
    async def add_sticker(message: Message) -> None:
        if message.from_user is None or message.from_user.id != owner_telegram_id:
            return
        source = message.reply_to_message
        if source is None or source.sticker is None:
            await message.answer(
                "Ответь командой /addsticker описание на сообщение со стикером."
            )
            return
        description = (message.text or "").partition(" ")[2].strip()
        if not description:
            description = await _describe_for_library(message, source, group_agent)
            if not description:
                return
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await group_agent.add_sticker(
                file_id=source.sticker.file_id,
                description=description,
                emoji=source.sticker.emoji,
            )
        await message.answer("Стикер добавлен. Арсенал социальной неловкости растёт.")

    @router.message(Command("addsound"))
    async def add_sound(message: Message) -> None:
        if message.from_user is None or message.from_user.id != owner_telegram_id:
            return
        source = message.reply_to_message
        if source is None or (source.voice is None and source.audio is None):
            await message.answer(
                "Ответь командой /addsound описание на войс или аудиофайл."
            )
            return
        description = (message.text or "").partition(" ")[2].strip()
        if not description:
            description = (source.caption or "").strip()
        if not description:
            description = await _describe_for_library(message, source, group_agent)
            if not description:
                return
        item = source.voice or source.audio
        assert item is not None
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await group_agent.add_sound(
                file_id=item.file_id,
                description=description,
                media_kind="voice" if source.voice is not None else "audio",
            )
        await message.answer("Звуковой мем добавлен. Теперь можно отвечать не только ртом.")

    @router.message(Command("addmlbb"))
    async def add_mlbb(message: Message) -> None:
        if message.from_user is None or message.from_user.id != owner_telegram_id:
            return
        payload = (message.text or "").partition(" ")[2].strip()
        if not payload and message.reply_to_message is not None:
            payload = (
                message.reply_to_message.text
                or message.reply_to_message.caption
                or ""
            ).strip()
        parts = [part.strip() for part in payload.split("|", maxsplit=2)]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            await message.answer(
                "Формат: /addmlbb заголовок | проверенный факт | ссылка на источник"
            )
            return
        title, body = parts[:2]
        source_url = parts[2] if len(parts) == 3 else ""
        source_ref = f"telegram:{message.chat.id}:{message.message_id}"
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await group_agent.add_mlbb_knowledge(
                source_ref=source_ref,
                title=title[:300],
                body=body[:6000],
                source_url=source_url[:1000],
            )
        await message.answer("Факт по MLBB записан. Теперь хотя бы не из воздуха.")

    @router.message(Command("pic", "gif", "sound", "sticker", "save", "creator", "pack"))
    async def media_command(message: Message) -> None:
        if not private_access_allowed(message):
            return
        command, _, query = (message.text or "").partition(" ")
        name = command.split("@")[0].lstrip("/")
        if name == "pack":
            if message.from_user is None or message.from_user.id != owner_telegram_id:
                return
            result = await group_agent.import_sticker_pack(message.chat.id, message.from_user.id, query.strip())
            await message.answer(str(result.get("message") or result.get("error")))
            return
        phrases = {"pic": "найди и отправь картинку ", "gif": "найди и отправь гифку ",
                   "sound": "найди и отправь звуковой мем ", "sticker": "найди и отправь стикер ",
                   "save": "сохрани ", "creator": "расскажи историю своего создателя "}
        event = replace(_event_from_message(message), text=phrases[name] + query, directed_to_bot=True)
        if await group_agent.handle_owner_command(message.chat.id, event):
            return
        if name == "save":
            return
        await group_agent.process(MessageBatch(chat_id=message.chat.id, events=[event], priority=True))

    content_filter = F.text | F.caption | F.document | F.photo | F.sticker | F.voice | F.audio | F.video | F.animation | F.video_note

    @router.message(content_filter)
    @router.channel_post(content_filter)
    async def incoming_message(message: Message) -> None:
        text = _message_text(message)
        if not text or text.startswith("/"):
            return
        if not private_access_allowed(message):
            return
        if await group_agent.handle_owner_command(message.chat.id, _event_from_message(message)):
            return

        if message.chat.type == ChatType.PRIVATE:
            if not private_access_allowed(message) or message.from_user is None:
                return
            await _handle_private_message(
                message=message,
                text=text,
                store=store,
                assistant=assistant,
                group_agent=group_agent,
                history_messages=history_messages,
                locks=private_locks,
            )
            return

        if message.from_user is not None and message.from_user.id == bot_id:
            return
        sender_id, sender_name = _sender_identity(message)
        file_id, file_name, mime_type, file_size = _attachment_metadata(message)
        directed_to_bot = _is_directed_to_bot(
            message,
            bot_id=bot_id,
            bot_username=bot_username,
        )
        continuation_candidate = (
            not directed_to_bot
            and message.reply_to_message is None
            and group_agent.claim_conversation_followup(message.chat.id, sender_id, text)
        )
        replied = message.reply_to_message
        await group_batcher.submit(
            chat_id=message.chat.id,
            priority=directed_to_bot or continuation_candidate,
            event=ChatEvent(
                telegram_message_id=message.message_id,
                sender_id=sender_id,
                sender_name=sender_name,
                text=text,
                reply_to_message_id=(
                    message.reply_to_message.message_id
                    if message.reply_to_message is not None
                    else None
                ),
                file_id=file_id,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                directed_to_bot=directed_to_bot,
                continuation_candidate=continuation_candidate,
                reply_text=_message_text(replied)[:2000] if replied else "",
                reply_sender=_sender_identity(replied)[1] if replied else "",
                reply_attachment=_event_from_message(replied) if replied else None,
                **_media_identity(message),
            ),
        )
        if directed_to_bot:
            try:
                await message.bot.send_chat_action(
                    chat_id=message.chat.id,
                    action=ChatAction.TYPING,
                )
            except Exception:
                logger.debug(
                    "Failed to send immediate typing action in chat %s",
                    message.chat.id,
                    exc_info=True,
                )

    @router.error()
    async def error_handler(event: ErrorEvent) -> bool:
        exception = event.exception
        logger.error(
            "Unhandled bot error",
            exc_info=(type(exception), exception, exception.__traceback__),
        )
        return True

    return router


async def _describe_for_library(message: Message, source: Message, agent: GroupAgent) -> str:
    try:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            result = await agent.read_attachment(_event_from_message(source))
        if result.get("ok") and result.get("text"):
            return str(result["text"])[:2000]
    except Exception:
        logger.exception("Could not describe library item")
    await message.answer("Не получилось разобрать файл. Повтори команду с коротким описанием смысла.")
    return ""


async def _handle_private_message(
    *,
    message: Message,
    text: str,
    store: MessageStore,
    assistant: GeminiAssistant,
    group_agent: GroupAgent,
    history_messages: int,
    locks: defaultdict[int, asyncio.Lock],
) -> None:
    # Keep the lock object: deleting it while another message waits creates two locks.
    async with locks[message.chat.id]:
        recent = await store.get_recent_chat_events(chat_id=message.chat.id, limit=1)
        if not recent:
            history = await store.get_history(chat_id=message.chat.id, user_id=message.chat.id, limit=history_messages)
            legacy = [
                ChatEvent(
                    telegram_message_id=-len(history) + index,
                    sender_id=message.chat.id if item[0] == "user" else message.bot.id,
                    sender_name=message.from_user.full_name if item[0] == "user" and message.from_user else "Лисун",
                    text=item[1], role="user" if item[0] == "user" else "bot",
                )
                for index, item in enumerate(history)
            ]
            if legacy:
                await store.append_chat_events(chat_id=message.chat.id, events=legacy)
        event = replace(_event_from_message(message), text=text, directed_to_bot=True)
        await group_agent.process(MessageBatch(chat_id=message.chat.id, events=[event], priority=True))


def _context_id(message: Message, owner_telegram_id: int) -> int:
    if message.chat.type == ChatType.PRIVATE:
        return owner_telegram_id
    return message.chat.id


def _sender_identity(message: Message) -> tuple[int, str]:
    if message.from_user is not None:
        name = message.from_user.full_name
        if message.from_user.username:
            name += f" (@{message.from_user.username})"
        return message.from_user.id, name
    if message.sender_chat is not None:
        return message.sender_chat.id, message.sender_chat.title or str(message.sender_chat.id)
    return message.chat.id, message.chat.title or str(message.chat.id)


def _is_directed_to_bot(
    message: Message,
    *,
    bot_id: int,
    bot_username: str,
) -> bool:
    replied = message.reply_to_message
    if (
        replied is not None
        and replied.from_user is not None
        and replied.from_user.id == bot_id
    ):
        return True
    raw_text = message.text or message.caption or ""
    return (
        f"@{bot_username}".casefold() in raw_text.casefold()
        or bool(BOT_NAME_RE.match(raw_text.strip()))
    )


def _message_text(message: Message) -> str:
    text = (message.text or message.caption or "").strip()
    if message.animation is not None:
        marker = "[GIF/анимация]"
    elif message.video_note is not None:
        marker = "[видеокружок]"
    elif message.document is not None:
        marker = f"[файл: {message.document.file_name or 'без имени'}]"
    elif message.photo:
        marker = "[фотография]"
    elif message.sticker is not None:
        preview = "; доступно только статичное превью" if message.sticker.is_animated else ""
        marker = f"[стикер {message.sticker.emoji or ''}{preview}]"
    elif message.voice is not None:
        marker = "[голосовое сообщение]"
    elif message.audio is not None:
        marker = f"[аудио: {message.audio.file_name or message.audio.title or 'без названия'}]"
    elif message.video is not None:
        marker = f"[видео: {message.video.file_name or 'без названия'}]"
    else:
        marker = ""
    return f"{marker} {text}".strip()


def _attachment_metadata(
    message: Message,
) -> tuple[str | None, str | None, str | None, int | None]:
    if message.animation is not None:
        item = message.animation
        return item.file_id, item.file_name or "animation.mp4", item.mime_type or "video/mp4", item.file_size
    if message.video_note is not None:
        item = message.video_note
        return item.file_id, "video_note.mp4", "video/mp4", item.file_size
    if message.sticker is not None:
        item = message.sticker
        if item.is_animated:
            if item.thumbnail is not None:
                thumb = item.thumbnail
                return thumb.file_id, "sticker_preview.webp", "image/webp", thumb.file_size
            return None, None, None, None
        suffix, mime = ("webm", "video/webm") if item.is_video else ("webp", "image/webp")
        return item.file_id, f"sticker.{suffix}", mime, item.file_size
    if message.document is not None:
        item = message.document
        return item.file_id, item.file_name, item.mime_type, item.file_size
    if message.photo:
        item = message.photo[-1]
        return item.file_id, f"photo_{message.message_id}.jpg", "image/jpeg", item.file_size
    if message.voice is not None:
        item = message.voice
        return item.file_id, f"voice_{message.message_id}.ogg", item.mime_type or "audio/ogg", item.file_size
    if message.audio is not None:
        item = message.audio
        return item.file_id, item.file_name or f"audio_{message.message_id}.mp3", item.mime_type or "audio/mpeg", item.file_size
    if message.video is not None:
        item = message.video
        return item.file_id, item.file_name or f"video_{message.message_id}.mp4", item.mime_type or "video/mp4", item.file_size
    return None, None, None, None


def _event_from_message(message: Message, *, include_reply: bool = True) -> ChatEvent:
    sender_id, sender_name = _sender_identity(message)
    file_id, file_name, mime_type, file_size = _attachment_metadata(message)
    replied = message.reply_to_message if include_reply else None
    return ChatEvent(
        telegram_message_id=message.message_id, sender_id=sender_id,
        sender_name=sender_name, text=_message_text(message), file_id=file_id,
        file_name=file_name, mime_type=mime_type, file_size=file_size,
        reply_to_message_id=replied.message_id if replied else None,
        reply_text=_message_text(replied)[:2000] if replied else "",
        reply_sender=_sender_identity(replied)[1] if replied else "",
        reply_attachment=_event_from_message(replied, include_reply=False) if replied else None,
        **_media_identity(message),
    )


def _media_identity(message: Message) -> dict[str, str]:
    for kind in ("animation", "sticker", "voice", "audio", "video_note", "video", "photo", "document"):
        item = getattr(message, kind, None)
        if item:
            if kind == "photo":
                item = item[-1]
            return {"media_kind": kind, "media_file_id": item.file_id, "media_unique_id": item.file_unique_id}
    return {}


async def _send_long_message(message: Message, text: str) -> None:
    remaining = text
    while remaining:
        if len(remaining) <= TELEGRAM_MESSAGE_LIMIT:
            chunk = remaining
            remaining = ""
        else:
            split_at = remaining.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
            if split_at < TELEGRAM_MESSAGE_LIMIT // 2:
                split_at = remaining.rfind(" ", 0, TELEGRAM_MESSAGE_LIMIT)
            if split_at <= 0:
                split_at = TELEGRAM_MESSAGE_LIMIT
            chunk = remaining[:split_at].rstrip()
            remaining = remaining[split_at:].lstrip()
        await message.answer(chunk)
