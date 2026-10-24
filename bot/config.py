from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    anymodel_api_key: str | None
    anymodel_model: str
    openrouter_api_key: str | None
    openrouter_model: str
    owner_telegram_id: int
    gemini_model: str
    gemini_embedding_model: str
    gemini_tts_model: str
    gemini_tts_voice: str
    database_path: Path
    history_messages: int
    max_output_tokens: int
    group_debounce_seconds: float
    group_direct_debounce_seconds: float
    group_max_batch_seconds: float
    group_max_batch_messages: int
    group_queue_workers: int
    group_recent_events: int
    memory_results: int
    style_results: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", telegram_bot_token),
                ("GEMINI_API_KEY", gemini_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Не заполнены обязательные переменные: " + ", ".join(missing)
            )

        history_messages = _positive_int("HISTORY_MESSAGES", default=20)
        max_output_tokens = _positive_int("MAX_OUTPUT_TOKENS", default=500)
        owner_telegram_id = _positive_int("OWNER_TELEGRAM_ID")
        group_debounce_seconds = _positive_float(
            "GROUP_DEBOUNCE_SECONDS", default=1.2
        )
        group_direct_debounce_seconds = _positive_float(
            "GROUP_DIRECT_DEBOUNCE_SECONDS", default=0.2
        )
        group_max_batch_seconds = _positive_float(
            "GROUP_MAX_BATCH_SECONDS", default=8.0
        )
        if group_max_batch_seconds < group_debounce_seconds:
            raise RuntimeError(
                "GROUP_MAX_BATCH_SECONDS не может быть меньше GROUP_DEBOUNCE_SECONDS"
            )

        return cls(
            telegram_bot_token=telegram_bot_token,
            gemini_api_key=gemini_api_key,
            anymodel_api_key=os.getenv("ANYMODEL_API_KEY", "").strip() or None,
            anymodel_model=os.getenv("ANYMODEL_MODEL", "am/free").strip(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip() or None,
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL", "google/gemini-3.8-flash"
            ).strip(),
            owner_telegram_id=owner_telegram_id,
            gemini_model=os.getenv(
                "GEMINI_MODEL", "gemini-3.8-flash"
            ).strip(),
            gemini_embedding_model=os.getenv(
                "GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"
            ).strip(),
            gemini_tts_model=os.getenv(
                "GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts"
            ).strip(),
            gemini_tts_voice=os.getenv("GEMINI_TTS_VOICE", "Puck").strip(),
            database_path=Path(
                os.getenv("DATABASE_PATH", "data/bot.sqlite3")
            ),
            history_messages=history_messages,
            max_output_tokens=max_output_tokens,
            group_debounce_seconds=group_debounce_seconds,
            group_direct_debounce_seconds=group_direct_debounce_seconds,
            group_max_batch_seconds=group_max_batch_seconds,
            group_max_batch_messages=_positive_int(
                "GROUP_MAX_BATCH_MESSAGES", default=50
            ),
            group_queue_workers=_positive_int("GROUP_QUEUE_WORKERS", default=2),
            group_recent_events=_positive_int("GROUP_RECENT_EVENTS", default=15),
            memory_results=_positive_int("MEMORY_RESULTS", default=8),
            style_results=_positive_int("STYLE_RESULTS", default=4),
        )


def _positive_int(name: str, *, default: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        if default is None:
            raise RuntimeError(f"Не заполнена обязательная переменная: {name}")
        raw_value = str(default)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} должен быть целым числом") from error

    if value <= 0:
        raise RuntimeError(f"{name} должен быть больше нуля")
    return value


def _positive_float(name: str, *, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} должен быть числом") from error
    if value <= 0:
        raise RuntimeError(f"{name} должен быть больше нуля")
    return value

