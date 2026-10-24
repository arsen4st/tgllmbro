from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from itertools import count
from time import monotonic
from typing import Awaitable, Callable

from bot.database import ChatEvent


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MessageBatch:
    chat_id: int
    events: list[ChatEvent]
    priority: bool = False


@dataclass(slots=True)
class _PendingBatch:
    events: list[ChatEvent] = field(default_factory=list)
    first_at: float = field(default_factory=monotonic)
    last_at: float = field(default_factory=monotonic)
    debounce_seconds: float = 0.0
    priority: bool = False
    timer: asyncio.Task[None] | None = None


class GroupBatcher:
    def __init__(
        self,
        *,
        processor: Callable[[MessageBatch], Awaitable[None]],
        debounce_seconds: float,
        direct_debounce_seconds: float,
        max_batch_seconds: float,
        max_batch_messages: int,
        workers: int,
    ) -> None:
        self._processor = processor
        self._debounce_seconds = debounce_seconds
        self._direct_debounce_seconds = direct_debounce_seconds
        self._max_batch_seconds = max_batch_seconds
        self._max_batch_messages = max_batch_messages
        self._worker_count = workers
        self._pending: dict[int, _PendingBatch] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, MessageBatch | None]
        ] = asyncio.PriorityQueue()
        self._sequence = count()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._workers = [
            asyncio.create_task(self._worker(), name=f"group-batch-worker-{index}")
            for index in range(self._worker_count)
        ]

    async def submit(
        self,
        *,
        chat_id: int,
        event: ChatEvent,
        priority: bool = False,
    ) -> None:
        batch_to_queue: MessageBatch | None = None
        async with self._lock:
            state = self._pending.get(chat_id)
            if state is None:
                now = monotonic()
                state = _PendingBatch(
                    first_at=now,
                    last_at=now,
                    debounce_seconds=(
                        self._direct_debounce_seconds
                        if priority
                        else self._debounce_seconds
                    ),
                    priority=priority,
                )
                self._pending[chat_id] = state
            state.events.append(event)
            state.last_at = monotonic()
            if priority:
                state.priority = True
                state.debounce_seconds = min(
                    state.debounce_seconds,
                    self._direct_debounce_seconds,
                )
            if state.timer is not None:
                state.timer.cancel()
            if len(state.events) >= self._max_batch_messages:
                self._pending.pop(chat_id, None)
                batch_to_queue = MessageBatch(
                    chat_id=chat_id,
                    events=state.events,
                    priority=state.priority,
                )
                state.timer = None
            else:
                state.timer = asyncio.create_task(
                    self._wait_and_flush(chat_id, state),
                    name=f"group-debounce-{chat_id}",
                )

        if batch_to_queue is not None:
            await self._enqueue(batch_to_queue)

    async def close(self) -> None:
        async with self._lock:
            states = list(self._pending.items())
            self._pending.clear()
        timers = []
        for _, state in states:
            if state.timer is not None:
                state.timer.cancel()
                timers.append(state.timer)
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        for chat_id, state in states:
            if state.events:
                await self._enqueue(
                    MessageBatch(
                        chat_id=chat_id,
                        events=state.events,
                        priority=state.priority,
                    )
                )
        await self._queue.join()
        for _ in self._workers:
            await self._queue.put((2, next(self._sequence), None))
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def _wait_and_flush(
        self,
        chat_id: int,
        expected_state: _PendingBatch,
    ) -> None:
        try:
            while True:
                async with self._lock:
                    state = self._pending.get(chat_id)
                    if state is not expected_state:
                        return
                    now = monotonic()
                    quiet_left = state.debounce_seconds - (now - state.last_at)
                    max_left = self._max_batch_seconds - (now - state.first_at)
                    delay = min(quiet_left, max_left)
                    if delay <= 0:
                        self._pending.pop(chat_id, None)
                        batch = MessageBatch(
                            chat_id=chat_id,
                            events=state.events,
                            priority=state.priority,
                        )
                        break
                await asyncio.sleep(delay)
            await self._enqueue(batch)
        except asyncio.CancelledError:
            raise

    async def _enqueue(self, batch: MessageBatch) -> None:
        queue_priority = 0 if batch.priority else 1
        await self._queue.put((queue_priority, next(self._sequence), batch))

    async def _worker(self) -> None:
        while True:
            _, _, batch = await self._queue.get()
            try:
                if batch is None:
                    return
                await self._processor(batch)
            except Exception:
                logger.exception("Failed to process group batch")
            finally:
                self._queue.task_done()
