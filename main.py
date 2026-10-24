from __future__ import annotations

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aiogram import Bot, Dispatcher

from bot.batcher import GroupBatcher
from bot.config import Settings
from bot.database import MessageStore
from bot.gemini import GeminiAssistant
from bot.group_agent import GroupAgent
from bot.handlers import build_router


async def main() -> None:
    settings = Settings.from_env()
    log_directory = Path("logs")
    log_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                log_directory / "bot-runtime.log",
                maxBytes=5_000_000,
                backupCount=5,
                encoding="utf-8",
            ),
        ],
    )

    bot = Bot(token=settings.telegram_bot_token)
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
    await store.connect()
    group_batcher: GroupBatcher | None = None
    try:
        bot_info = await bot.get_me()
        if not bot_info.username:
            raise RuntimeError("У Telegram-бота не задан username")
        if not bot_info.can_read_all_group_messages:
            logging.getLogger(__name__).warning(
                "Telegram Privacy Mode включён: обычные сообщения групп не видны боту"
            )

        group_agent = GroupAgent(
            bot=bot,
            bot_id=bot_info.id,
            bot_username=bot_info.username,
            store=store,
            assistant=assistant,
            recent_events=settings.group_recent_events,
            memory_results=settings.memory_results,
            style_results=settings.style_results,
            owner_telegram_id=settings.owner_telegram_id,
        )
        group_batcher = GroupBatcher(
            processor=group_agent.process,
            debounce_seconds=settings.group_debounce_seconds,
            direct_debounce_seconds=settings.group_direct_debounce_seconds,
            max_batch_seconds=settings.group_max_batch_seconds,
            max_batch_messages=settings.group_max_batch_messages,
            workers=settings.group_queue_workers,
        )
        await group_batcher.start()

        dispatcher = Dispatcher()
        dispatcher.include_router(
            build_router(
                store=store,
                assistant=assistant,
                group_agent=group_agent,
                group_batcher=group_batcher,
                history_messages=settings.history_messages,
                owner_telegram_id=settings.owner_telegram_id,
                bot_id=bot_info.id,
                bot_username=bot_info.username,
            )
        )
        await bot.delete_webhook(drop_pending_updates=False)
        await dispatcher.start_polling(bot)
    finally:
        if group_batcher is not None:
            await group_batcher.close()
        await assistant.close()
        await store.close()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
