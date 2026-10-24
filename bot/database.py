from __future__ import annotations

import asyncio
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import aiosqlite


@dataclass(frozen=True, slots=True)
class ChatEvent:
    telegram_message_id: int
    sender_id: int
    sender_name: str
    text: str
    reply_to_message_id: int | None = None
    role: str = "user"
    file_id: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    file_size: int | None = None
    directed_to_bot: bool = False
    continuation_candidate: bool = False
    reply_text: str = ""
    reply_sender: str = ""
    reply_attachment: ChatEvent | None = None
    created_at: str = ""
    media_kind: str = ""
    media_file_id: str = ""
    media_unique_id: str = ""


@dataclass(frozen=True, slots=True)
class MemoryItem:
    id: int
    text: str
    embedding: list[float]
    importance: float


@dataclass(frozen=True, slots=True)
class MemeItem:
    id: int
    file_id: str
    description: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class SoundItem:
    id: int
    file_id: str
    description: str
    media_kind: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class StickerItem:
    id: int
    file_id: str
    description: str
    emoji: str | None
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class StyleExample:
    id: int
    context: str
    response: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class KnowledgeItem:
    id: int
    title: str
    body: str
    source_url: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class UserProfileFact:
    user_id: int
    display_name: str
    key: str
    value: str


class MessageStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._database_path)
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'model')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_messages_context
            ON messages (chat_id, user_id, id DESC);

            CREATE TABLE IF NOT EXISTS chat_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                telegram_message_id INTEGER NOT NULL,
                sender_id INTEGER NOT NULL,
                sender_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'bot')),
                text TEXT NOT NULL,
                reply_to_message_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (chat_id, telegram_message_id, role)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_events_recent
            ON chat_events (chat_id, id DESC);

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_used_at TEXT,
                UNIQUE (chat_id, text)
            );

            CREATE INDEX IF NOT EXISTS idx_memories_chat
            ON memories (chat_id, id DESC);

            CREATE TABLE IF NOT EXISTS memes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                content_sha256 TEXT,
                embedding BLOB NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                media_kind TEXT NOT NULL CHECK (media_kind IN ('voice', 'audio')),
                embedding BLOB NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS stickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                emoji TEXT,
                embedding BLOB NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS style_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_ref TEXT NOT NULL UNIQUE,
                context TEXT NOT NULL DEFAULT '',
                response TEXT NOT NULL,
                embedding BLOB NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_ref TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT '',
                embedding BLOB NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chat_id, user_id, key)
            );

            CREATE INDEX IF NOT EXISTS idx_user_profiles_chat_user
            ON user_profiles (chat_id, user_id);

            CREATE TABLE IF NOT EXISTS attachment_cache (
                chat_id INTEGER NOT NULL,
                file_id TEXT NOT NULL,
                description TEXT NOT NULL,
                PRIMARY KEY (chat_id, file_id)
            );
            CREATE TABLE IF NOT EXISTS media_library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_chat INTEGER NOT NULL,
                kind TEXT NOT NULL,
                file_id TEXT NOT NULL,
                unique_id TEXT NOT NULL,
                description TEXT NOT NULL,
                source_chat INTEGER NOT NULL,
                source_message INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scope_chat, unique_id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS media_library_fts USING fts5(description, content='media_library', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS media_library_ai AFTER INSERT ON media_library BEGIN
                INSERT INTO media_library_fts(rowid, description) VALUES(new.id, new.description);
            END;
            CREATE TRIGGER IF NOT EXISTS media_library_au AFTER UPDATE ON media_library BEGIN
                INSERT INTO media_library_fts(media_library_fts, rowid, description) VALUES('delete', old.id, old.description);
                INSERT INTO media_library_fts(rowid, description) VALUES(new.id, new.description);
            END;
            CREATE TABLE IF NOT EXISTS creator_notes (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                public INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        await self._ensure_schema_extensions()
        await self._connection.commit()

    async def _ensure_schema_extensions(self) -> None:
        connection = self._require_connection()
        cursor = await connection.execute("PRAGMA table_info(chat_events)")
        columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        for name, sql_type in (
            ("file_id", "TEXT"),
            ("file_name", "TEXT"),
            ("mime_type", "TEXT"),
            ("file_size", "INTEGER"),
            ("reply_text", "TEXT NOT NULL DEFAULT ''"),
            ("reply_sender", "TEXT NOT NULL DEFAULT ''"),
            ("media_kind", "TEXT NOT NULL DEFAULT ''"),
            ("media_file_id", "TEXT NOT NULL DEFAULT ''"),
            ("media_unique_id", "TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in columns:
                await connection.execute(
                    f"ALTER TABLE chat_events ADD COLUMN {name} {sql_type}"
                )

        cursor = await connection.execute("PRAGMA table_info(memes)")
        meme_columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        if "content_sha256" not in meme_columns:
            await connection.execute(
                "ALTER TABLE memes ADD COLUMN content_sha256 TEXT"
            )
        await connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memes_content_sha256
            ON memes (content_sha256)
            WHERE content_sha256 IS NOT NULL
            """
        )

        await connection.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                chat_id UNINDEXED,
                text,
                content='memories',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, chat_id, text)
                VALUES (new.id, new.chat_id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, chat_id, text)
                VALUES ('delete', old.id, old.chat_id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, chat_id, text)
                VALUES ('delete', old.id, old.chat_id, old.text);
                INSERT INTO memories_fts(rowid, chat_id, text)
                VALUES (new.id, new.chat_id, new.text);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS chat_events_fts USING fts5(
                chat_id UNINDEXED,
                sender_name,
                text,
                content='chat_events',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chat_events_fts_ai AFTER INSERT ON chat_events BEGIN
                INSERT INTO chat_events_fts(rowid, chat_id, sender_name, text)
                VALUES (new.id, new.chat_id, new.sender_name, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chat_events_fts_ad AFTER DELETE ON chat_events BEGIN
                INSERT INTO chat_events_fts(
                    chat_events_fts, rowid, chat_id, sender_name, text
                ) VALUES ('delete', old.id, old.chat_id, old.sender_name, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chat_events_fts_au AFTER UPDATE ON chat_events BEGIN
                INSERT INTO chat_events_fts(
                    chat_events_fts, rowid, chat_id, sender_name, text
                ) VALUES ('delete', old.id, old.chat_id, old.sender_name, old.text);
                INSERT INTO chat_events_fts(rowid, chat_id, sender_name, text)
                VALUES (new.id, new.chat_id, new.sender_name, new.text);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS memes_fts USING fts5(
                description,
                content='memes',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS memes_fts_ai AFTER INSERT ON memes BEGIN
                INSERT INTO memes_fts(rowid, description)
                VALUES (new.id, new.description);
            END;
            CREATE TRIGGER IF NOT EXISTS memes_fts_ad AFTER DELETE ON memes BEGIN
                INSERT INTO memes_fts(memes_fts, rowid, description)
                VALUES ('delete', old.id, old.description);
            END;
            CREATE TRIGGER IF NOT EXISTS memes_fts_au AFTER UPDATE ON memes BEGIN
                INSERT INTO memes_fts(memes_fts, rowid, description)
                VALUES ('delete', old.id, old.description);
                INSERT INTO memes_fts(rowid, description)
                VALUES (new.id, new.description);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS sounds_fts USING fts5(
                description,
                content='sounds',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS sounds_fts_ai AFTER INSERT ON sounds BEGIN
                INSERT INTO sounds_fts(rowid, description)
                VALUES (new.id, new.description);
            END;
            CREATE TRIGGER IF NOT EXISTS sounds_fts_ad AFTER DELETE ON sounds BEGIN
                INSERT INTO sounds_fts(sounds_fts, rowid, description)
                VALUES ('delete', old.id, old.description);
            END;
            CREATE TRIGGER IF NOT EXISTS sounds_fts_au AFTER UPDATE ON sounds BEGIN
                INSERT INTO sounds_fts(sounds_fts, rowid, description)
                VALUES ('delete', old.id, old.description);
                INSERT INTO sounds_fts(rowid, description)
                VALUES (new.id, new.description);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS style_examples_fts USING fts5(
                context,
                response,
                content='style_examples',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS style_examples_fts_ai
            AFTER INSERT ON style_examples BEGIN
                INSERT INTO style_examples_fts(rowid, context, response)
                VALUES (new.id, new.context, new.response);
            END;
            CREATE TRIGGER IF NOT EXISTS style_examples_fts_ad
            AFTER DELETE ON style_examples BEGIN
                INSERT INTO style_examples_fts(
                    style_examples_fts, rowid, context, response
                ) VALUES ('delete', old.id, old.context, old.response);
            END;
            CREATE TRIGGER IF NOT EXISTS style_examples_fts_au
            AFTER UPDATE ON style_examples BEGIN
                INSERT INTO style_examples_fts(
                    style_examples_fts, rowid, context, response
                ) VALUES ('delete', old.id, old.context, old.response);
                INSERT INTO style_examples_fts(rowid, context, response)
                VALUES (new.id, new.context, new.response);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_items_fts USING fts5(
                title,
                body,
                content='knowledge_items',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS knowledge_items_fts_ai
            AFTER INSERT ON knowledge_items BEGIN
                INSERT INTO knowledge_items_fts(rowid, title, body)
                VALUES (new.id, new.title, new.body);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_items_fts_ad
            AFTER DELETE ON knowledge_items BEGIN
                INSERT INTO knowledge_items_fts(
                    knowledge_items_fts, rowid, title, body
                ) VALUES ('delete', old.id, old.title, old.body);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_items_fts_au
            AFTER UPDATE ON knowledge_items BEGIN
                INSERT INTO knowledge_items_fts(
                    knowledge_items_fts, rowid, title, body
                ) VALUES ('delete', old.id, old.title, old.body);
                INSERT INTO knowledge_items_fts(rowid, title, body)
                VALUES (new.id, new.title, new.body);
            END;
            """
        )
        await connection.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
        await connection.execute(
            "INSERT INTO chat_events_fts(chat_events_fts) VALUES ('rebuild')"
        )
        await connection.execute("INSERT INTO memes_fts(memes_fts) VALUES ('rebuild')")
        await connection.execute("INSERT INTO sounds_fts(sounds_fts) VALUES ('rebuild')")
        await connection.execute(
            "INSERT INTO style_examples_fts(style_examples_fts) VALUES ('rebuild')"
        )
        await connection.execute(
            "INSERT INTO knowledge_items_fts(knowledge_items_fts) VALUES ('rebuild')"
        )

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def get_history(
        self,
        *,
        chat_id: int,
        user_id: int,
        limit: int,
    ) -> list[tuple[str, str]]:
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id = ? AND user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, user_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(reversed(rows))

    async def append_exchange(
        self,
        *,
        chat_id: int,
        user_id: int,
        user_text: str,
        model_text: str,
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.executemany(
                """
                INSERT INTO messages (chat_id, user_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (
                    (chat_id, user_id, "user", user_text),
                    (chat_id, user_id, "model", model_text),
                ),
            )
            await connection.commit()

    async def append_chat_events(
        self,
        *,
        chat_id: int,
        events: Iterable[ChatEvent],
    ) -> None:
        rows = [
            (
                chat_id,
                event.telegram_message_id,
                event.sender_id,
                event.sender_name,
                event.role,
                event.text,
                event.reply_to_message_id,
                event.file_id,
                event.file_name,
                event.mime_type,
                event.file_size,
                event.reply_text,
                event.reply_sender,
                event.media_kind,
                event.media_file_id,
                event.media_unique_id,
            )
            for event in events
        ]
        if not rows:
            return
        connection = self._require_connection()
        async with self._write_lock:
            await connection.executemany(
                """
                INSERT OR IGNORE INTO chat_events (
                    chat_id, telegram_message_id, sender_id, sender_name,
                    role, text, reply_to_message_id, file_id, file_name,
                    mime_type, file_size, reply_text, reply_sender, media_kind, media_file_id, media_unique_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            await connection.commit()

    async def get_recent_chat_events(
        self,
        *,
        chat_id: int,
        limit: int,
    ) -> list[ChatEvent]:
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT telegram_message_id, sender_id, sender_name, text,
                   reply_to_message_id, role, file_id, file_name, mime_type,
                   file_size, reply_text, reply_sender, created_at, media_kind, media_file_id, media_unique_id
            FROM chat_events
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_chat_event_from_row(row) for row in reversed(rows)]

    async def get_chat_event(self, *, chat_id: int, message_id: int) -> ChatEvent | None:
        cursor = await self._require_connection().execute(
            """
            SELECT telegram_message_id, sender_id, sender_name, text,
                   reply_to_message_id, role, file_id, file_name, mime_type,
                   file_size, reply_text, reply_sender, created_at, media_kind, media_file_id, media_unique_id
            FROM chat_events WHERE chat_id = ? AND telegram_message_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (chat_id, message_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _chat_event_from_row(row) if row else None

    async def get_attachment_description(self, chat_id: int, file_id: str) -> str | None:
        cursor = await self._require_connection().execute(
            "SELECT description FROM attachment_cache WHERE chat_id = ? AND file_id = ?",
            (chat_id, file_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row[0] if row else None

    async def cache_attachment(self, chat_id: int, file_id: str, description: str) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                "INSERT OR REPLACE INTO attachment_cache VALUES (?, ?, ?)",
                (chat_id, file_id, description[:18000]),
            )
            await connection.commit()

    async def search_chat_events(
        self,
        *,
        chat_id: int,
        query: str,
        limit: int,
    ) -> list[ChatEvent]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT e.telegram_message_id, e.sender_id, e.sender_name, e.text,
                   e.reply_to_message_id, e.role, e.file_id, e.file_name,
                   e.mime_type, e.file_size, e.reply_text, e.reply_sender, e.created_at, e.media_kind, e.media_file_id, e.media_unique_id
            FROM chat_events_fts
            JOIN chat_events AS e ON e.id = chat_events_fts.rowid
            WHERE chat_events_fts MATCH ? AND chat_events_fts.chat_id = ?
            ORDER BY bm25(chat_events_fts)
            LIMIT ?
            """,
            (fts_query, chat_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_chat_event_from_row(row) for row in rows]

    async def add_memory(
        self,
        *,
        chat_id: int,
        text: str,
        embedding: list[float],
        importance: float,
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                """
                INSERT INTO memories (chat_id, text, embedding, importance)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, text) DO UPDATE SET
                    embedding = excluded.embedding,
                    importance = MAX(memories.importance, excluded.importance)
                """,
                (chat_id, text, _pack_embedding(embedding), importance),
            )
            await connection.commit()

    async def get_memory_candidates(
        self,
        *,
        chat_id: int,
        limit: int = 250,
    ) -> list[MemoryItem]:
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT id, text, embedding, importance
            FROM memories
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            MemoryItem(row[0], row[1], _unpack_embedding(row[2]), row[3])
            for row in rows
        ]

    async def search_memory_ids(
        self,
        *,
        chat_id: int,
        query: str,
        limit: int = 50,
    ) -> set[int]:
        fts_query = _fts_query(query)
        if not fts_query:
            return set()
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT rowid
            FROM memories_fts
            WHERE memories_fts MATCH ? AND memories_fts.chat_id = ?
            ORDER BY bm25(memories_fts)
            LIMIT ?
            """,
            (fts_query, chat_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return {int(row[0]) for row in rows}

    async def touch_memories(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        connection = self._require_connection()
        placeholders = ",".join("?" for _ in memory_ids)
        async with self._write_lock:
            await connection.execute(
                f"UPDATE memories SET last_used_at = CURRENT_TIMESTAMP "
                f"WHERE id IN ({placeholders})",
                memory_ids,
            )
            await connection.commit()

    async def add_meme(
        self,
        *,
        file_id: str,
        description: str,
        embedding: list[float],
        content_sha256: str | None = None,
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                """
                INSERT INTO memes (file_id, description, content_sha256, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    description = excluded.description,
                    content_sha256 = COALESCE(excluded.content_sha256, memes.content_sha256),
                    embedding = excluded.embedding,
                    enabled = 1
                """,
                (file_id, description, content_sha256, _pack_embedding(embedding)),
            )
            await connection.commit()

    async def meme_ref_exists(self, file_id: str) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM memes WHERE file_id = ?)",
            (file_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def meme_hash_exists(self, content_sha256: str) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM memes WHERE content_sha256 = ?)",
            (content_sha256,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def search_meme_ids(self, *, query: str, limit: int = 500) -> list[int]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT m.id
            FROM memes_fts
            JOIN memes AS m ON m.id = memes_fts.rowid
            WHERE memes_fts MATCH ? AND m.enabled = 1
            ORDER BY bm25(memes_fts)
            LIMIT ?
            """,
            (fts_query, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [int(row[0]) for row in rows]

    async def get_meme_candidates(
        self,
        *,
        meme_ids: list[int] | None = None,
        limit: int = 500,
    ) -> list[MemeItem]:
        connection = self._require_connection()
        where = "WHERE enabled = 1"
        params: list[object] = []
        if meme_ids:
            placeholders = ",".join("?" for _ in meme_ids)
            where += f" AND id IN ({placeholders})"
            params.extend(meme_ids)
        params.append(limit)
        cursor = await connection.execute(
            f"""
            SELECT id, file_id, description, embedding
            FROM memes
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            MemeItem(row[0], row[1], row[2], _unpack_embedding(row[3]))
            for row in rows
        ]

    async def has_memes(self) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM memes WHERE enabled = 1)"
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def get_meme(self, meme_id: int) -> MemeItem | None:
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT id, file_id, description, embedding
            FROM memes
            WHERE id = ? AND enabled = 1
            """,
            (meme_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return MemeItem(row[0], row[1], row[2], _unpack_embedding(row[3]))

    async def add_sound(
        self,
        *,
        file_id: str,
        description: str,
        media_kind: str,
        embedding: list[float],
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                """
                INSERT INTO sounds (file_id, description, media_kind, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    description = excluded.description,
                    media_kind = excluded.media_kind,
                    embedding = excluded.embedding,
                    enabled = 1
                """,
                (file_id, description, media_kind, _pack_embedding(embedding)),
            )
            await connection.commit()

    async def search_sound_ids(self, *, query: str, limit: int = 500) -> list[int]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT s.id
            FROM sounds_fts
            JOIN sounds AS s ON s.id = sounds_fts.rowid
            WHERE sounds_fts MATCH ? AND s.enabled = 1
            ORDER BY bm25(sounds_fts)
            LIMIT ?
            """,
            (fts_query, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [int(row[0]) for row in rows]

    async def get_sound_candidates(
        self,
        *,
        sound_ids: list[int] | None = None,
        limit: int = 500,
    ) -> list[SoundItem]:
        connection = self._require_connection()
        where = "WHERE enabled = 1"
        params: list[object] = []
        if sound_ids:
            placeholders = ",".join("?" for _ in sound_ids)
            where += f" AND id IN ({placeholders})"
            params.extend(sound_ids)
        params.append(limit)
        cursor = await connection.execute(
            f"""
            SELECT id, file_id, description, media_kind, embedding
            FROM sounds
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            SoundItem(row[0], row[1], row[2], row[3], _unpack_embedding(row[4]))
            for row in rows
        ]

    async def get_sound(self, sound_id: int) -> SoundItem | None:
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT id, file_id, description, media_kind, embedding
            FROM sounds
            WHERE id = ? AND enabled = 1
            """,
            (sound_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return SoundItem(row[0], row[1], row[2], row[3], _unpack_embedding(row[4]))

    async def has_sounds(self) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM sounds WHERE enabled = 1)"
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def add_style_example(
        self,
        *,
        source_ref: str,
        context: str,
        response: str,
        embedding: list[float],
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                """
                INSERT INTO style_examples (source_ref, context, response, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_ref) DO UPDATE SET
                    context = excluded.context,
                    response = excluded.response,
                    embedding = excluded.embedding,
                    enabled = 1
                """,
                (source_ref, context, response, _pack_embedding(embedding)),
            )
            await connection.commit()

    async def style_ref_exists(self, source_ref: str) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM style_examples WHERE source_ref = ?)",
            (source_ref,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def has_style_examples(self) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM style_examples WHERE enabled = 1)"
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def search_style_ids(
        self,
        *,
        query: str,
        limit: int = 500,
    ) -> list[int]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT s.id
            FROM style_examples_fts
            JOIN style_examples AS s ON s.id = style_examples_fts.rowid
            WHERE style_examples_fts MATCH ? AND s.enabled = 1
            ORDER BY bm25(style_examples_fts)
            LIMIT ?
            """,
            (fts_query, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [int(row[0]) for row in rows]

    async def get_style_candidates(
        self,
        *,
        style_ids: list[int] | None = None,
        limit: int = 1000,
    ) -> list[StyleExample]:
        connection = self._require_connection()
        where = "WHERE enabled = 1"
        params: list[object] = []
        if style_ids:
            placeholders = ",".join("?" for _ in style_ids)
            where += f" AND id IN ({placeholders})"
            params.extend(style_ids)
        params.append(limit)
        cursor = await connection.execute(
            f"""
            SELECT id, context, response, embedding
            FROM style_examples
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            StyleExample(row[0], row[1], row[2], _unpack_embedding(row[3]))
            for row in rows
        ]

    async def add_knowledge(
        self,
        *,
        source_ref: str,
        title: str,
        body: str,
        source_url: str,
        embedding: list[float],
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                """
                INSERT INTO knowledge_items (
                    source_ref, title, body, source_url, embedding
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_ref) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    source_url = excluded.source_url,
                    embedding = excluded.embedding,
                    enabled = 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (source_ref, title, body, source_url, _pack_embedding(embedding)),
            )
            await connection.commit()

    async def knowledge_ref_exists(self, source_ref: str) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM knowledge_items WHERE source_ref = ?)",
            (source_ref,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def search_knowledge_ids(self, *, query: str, limit: int = 500) -> list[int]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT k.id
            FROM knowledge_items_fts
            JOIN knowledge_items AS k ON k.id = knowledge_items_fts.rowid
            WHERE knowledge_items_fts MATCH ? AND k.enabled = 1
            ORDER BY bm25(knowledge_items_fts)
            LIMIT ?
            """,
            (fts_query, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [int(row[0]) for row in rows]

    async def get_knowledge_candidates(
        self,
        *,
        item_ids: list[int] | None = None,
        limit: int = 1000,
    ) -> list[KnowledgeItem]:
        connection = self._require_connection()
        where = "WHERE enabled = 1"
        params: list[object] = []
        if item_ids:
            placeholders = ",".join("?" for _ in item_ids)
            where += f" AND id IN ({placeholders})"
            params.extend(item_ids)
        params.append(limit)
        cursor = await connection.execute(
            f"""
            SELECT id, title, body, source_url, embedding
            FROM knowledge_items
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            KnowledgeItem(row[0], row[1], row[2], row[3], _unpack_embedding(row[4]))
            for row in rows
        ]

    async def upsert_user_profile_fact(
        self,
        *,
        chat_id: int,
        user_id: int,
        display_name: str,
        key: str,
        value: str,
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                """
                INSERT INTO user_profiles (
                    chat_id, user_id, display_name, key, value
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, user_id, key) DO UPDATE SET
                    display_name = excluded.display_name,
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, user_id, display_name, key, value),
            )
            await connection.commit()

    async def get_user_profile_facts(
        self,
        *,
        chat_id: int,
        user_ids: set[int],
    ) -> list[UserProfileFact]:
        if not user_ids:
            return []
        connection = self._require_connection()
        placeholders = ",".join("?" for _ in user_ids)
        cursor = await connection.execute(
            f"""
            SELECT user_id, display_name, key, value
            FROM user_profiles
            WHERE chat_id = ? AND user_id IN ({placeholders})
            ORDER BY user_id, key
            """,
            [chat_id, *sorted(user_ids)],
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [UserProfileFact(*row) for row in rows]

    async def add_sticker(
        self,
        *,
        file_id: str,
        description: str,
        emoji: str | None,
        embedding: list[float],
    ) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                """
                INSERT INTO stickers (file_id, description, emoji, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    description = excluded.description,
                    emoji = excluded.emoji,
                    embedding = excluded.embedding,
                    enabled = 1
                """,
                (file_id, description, emoji, _pack_embedding(embedding)),
            )
            await connection.commit()

    async def get_sticker_candidates(self, *, limit: int = 500) -> list[StickerItem]:
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT id, file_id, description, emoji, embedding
            FROM stickers
            WHERE enabled = 1
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            StickerItem(row[0], row[1], row[2], row[3], _unpack_embedding(row[4]))
            for row in rows
        ]

    async def get_sticker(self, sticker_id: int) -> StickerItem | None:
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT id, file_id, description, emoji, embedding
            FROM stickers
            WHERE id = ? AND enabled = 1
            """,
            (sticker_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return StickerItem(row[0], row[1], row[2], row[3], _unpack_embedding(row[4]))

    async def has_stickers(self) -> bool:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT EXISTS(SELECT 1 FROM stickers WHERE enabled = 1)"
        )
        row = await cursor.fetchone()
        await cursor.close()
        return bool(row and row[0])

    async def clear_history(self, *, chat_id: int, user_id: int) -> None:
        connection = self._require_connection()
        async with self._write_lock:
            await connection.execute(
                "DELETE FROM messages WHERE chat_id = ? AND user_id = ?",
                (chat_id, user_id),
            )
            await connection.execute("DELETE FROM chat_events WHERE chat_id = ?", (chat_id,))
            await connection.execute("DELETE FROM memories WHERE chat_id = ?", (chat_id,))
            await connection.execute("DELETE FROM attachment_cache WHERE chat_id = ?", (chat_id,))
            await connection.commit()

    async def save_media(self, *, scope_chat: int, kind: str, file_id: str, unique_id: str,
                         description: str, source_chat: int, source_message: int) -> int:
        connection = self._require_connection()
        async with self._write_lock:
            cursor = await connection.execute(
                """INSERT INTO media_library(scope_chat,kind,file_id,unique_id,description,source_chat,source_message)
                   VALUES(?,?,?,?,?,?,?) ON CONFLICT(scope_chat,unique_id) DO UPDATE SET
                   file_id=excluded.file_id, description=excluded.description RETURNING id""",
                (scope_chat, kind, file_id, unique_id or file_id, description[:2000], source_chat, source_message),
            )
            row = await cursor.fetchone()
            await cursor.close()
            await connection.commit()
            return int(row[0])

    async def search_media(self, *, chat_id: int, query: str, kind: str = "any") -> list[dict]:
        fts = _fts_query(query)
        if not fts:
            return []
        cursor = await self._require_connection().execute(
            """SELECT m.id,m.kind,m.description FROM media_library_fts
               JOIN media_library m ON m.id=media_library_fts.rowid
               WHERE media_library_fts MATCH ? AND m.scope_chat IN (0,?)
               AND (?='any' OR m.kind=?) ORDER BY bm25(media_library_fts) LIMIT 8""",
            (fts, chat_id, kind, kind),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [{"media_id": row[0], "kind": row[1], "description": row[2]} for row in rows]

    async def get_media(self, *, chat_id: int, media_id: int) -> dict | None:
        cursor = await self._require_connection().execute(
            "SELECT id,kind,file_id,description FROM media_library WHERE id=? AND scope_chat IN (0,?)",
            (media_id, chat_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return dict(zip(("id", "kind", "file_id", "description"), row)) if row else None

    async def set_creator_note(self, *, key: str, value: str, public: bool = False) -> None:
        async with self._write_lock:
            await self._require_connection().execute(
                "INSERT INTO creator_notes VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,public=excluded.public",
                (key[:80], value[:500], int(public)),
            )
            await self._require_connection().commit()

    async def get_creator_notes(self, *, private: bool) -> str:
        cursor = await self._require_connection().execute(
            "SELECT key,value FROM creator_notes WHERE public=1 OR ?=1 ORDER BY rowid DESC LIMIT 20", (int(private),)
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return "\n".join(f"{key}: {value}" for key, value in rows)

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Хранилище сообщений ещё не подключено")
        return self._connection


def _chat_event_from_row(row: tuple) -> ChatEvent:
    return ChatEvent(
        *row[:10], reply_text=row[10] or "", reply_sender=row[11] or "",
        created_at=row[12] or "", media_kind=row[13] or "", media_file_id=row[14] or "", media_unique_id=row[15] or "",
    )


def _pack_embedding(values: list[float]) -> bytes:
    return array("f", values).tobytes()


def _unpack_embedding(value: bytes) -> list[float]:
    result = array("f")
    result.frombytes(value)
    return result.tolist()


def _fts_query(text: str) -> str:
    tokens = []
    for token in text.casefold().split():
        cleaned = "".join(character for character in token if character.isalnum())
        if len(cleaned) >= 2:
            tokens.append(f'"{cleaned}"')
    return " OR ".join(tokens[:20])
