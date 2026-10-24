import io
import tempfile
import unittest
import wave
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram import Bot
from aiogram.types import Message

from bot.batcher import MessageBatch
from bot.database import ChatEvent, MessageStore
from bot.gemini import AgentAction
from bot.group_agent import GroupAgent, _render_events
from bot.handlers import _attachment_metadata, _event_from_message, _handle_private_message
from bot.media import wrap_pcm


@asynccontextmanager
async def no_chat_action(**kwargs):
    yield


class DialogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = MessageStore(Path(self.directory.name) / "test.sqlite3")
        await self.store.connect()
        self.bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=101)))
        self.assistant = SimpleNamespace(plan_group_actions=AsyncMock(return_value=[]))
        self.agent = GroupAgent(bot=self.bot, bot_id=99, bot_username="yss", store=self.store,
                                assistant=self.assistant, recent_events=15, memory_results=4, style_results=2)
        self.person = ChatEvent(10, 7, "Вася", "а ты что думаешь?", directed_to_bot=True)

    async def asyncTearDown(self):
        await self.store.close()
        self.directory.cleanup()

    async def test_reply_survives_storage_and_stays_in_its_chat(self):
        quote = replace(self.person, reply_to_message_id=9, reply_text="sqlite или вики?", reply_sender="Коля")
        await self.store.append_chat_events(chat_id=1, events=[quote])
        restored = await self.store.get_chat_event(chat_id=1, message_id=10)
        self.assertEqual(restored.reply_sender, "Коля")
        self.assertIn("sqlite или вики?", _render_events([restored]))
        self.assertIn("role=user", _render_events([restored]))
        self.assertIsNone(await self.store.get_chat_event(chat_id=2, message_id=10))

    async def test_unrelated_speaker_cannot_steal_followup(self):
        self.agent._open_conversation(1, self.person)
        self.assertFalse(self.agent.claim_conversation_followup(1, 8, "не про бота"))
        self.assertTrue(self.agent.claim_conversation_followup(1, 7, "почему так?"))
        self.agent._open_conversation(1, replace(self.person, directed_to_bot=False, continuation_candidate=True))
        self.assertTrue(self.agent.claim_conversation_followup(1, 7, "а ещё?"))
        self.assertTrue(self.agent.claim_conversation_followup(1, 7, "и дальше?"))
        self.assertFalse(self.agent.claim_conversation_followup(1, 7, "безлимит?"))

    async def test_closing_stops_but_correction_is_allowed(self):
        self.agent._open_conversation(1, self.person)
        self.assertTrue(self.agent.claim_conversation_followup(1, 7, "что ты несёшь?"))
        self.assertFalse(self.agent.claim_conversation_followup(1, 7, "понял, спасибо"))
        self.assertFalse(self.agent.claim_conversation_followup(1, 7, "спасибо"))
        self.assertFalse(self.agent.claim_conversation_followup(1, 7, "посторонняя тема"))

    async def test_quoted_attachment_read_once_and_scoped(self):
        photo = ChatEvent(9, 8, "Коля", "[фото]", file_id="photo", mime_type="image/jpeg")
        event = replace(self.person, reply_to_message_id=9, reply_attachment=photo)
        self.agent._read_attachment = AsyncMock(return_value={"ok": True, "text": "На картинке кот."})
        batch = MessageBatch(1, [event])
        prepared = await self.agent._prepare_context(batch)
        self.assertIn("На картинке кот", prepared.events[0].text)
        await self.agent._execute_lookup(batch=prepared, action=AgentAction("read_attachment", {
            "target_message_id": 10, "source": "reply"}), retrieved_memory_ids=set())
        self.agent._read_attachment.assert_awaited_once()
        self.assertIsNone(await self.store.get_attachment_description(2, "photo"))

    async def test_tts_failure_sends_the_answer_as_text(self):
        self.assistant.synthesize_speech = AsyncMock(side_effect=RuntimeError("offline"))
        await self.store.append_chat_events(chat_id=1, events=[self.person])
        with patch("bot.group_agent.ChatActionSender.record_voice", no_chat_action):
            await self.agent._execute_actions(MessageBatch(1, [self.person]), [
                AgentAction("speak_to_message", {"target_message_id": 10, "text": "очередь сглаживает наплыв"})])
        self.assertIn("очередь сглаживает наплыв", self.bot.send_message.call_args.kwargs["text"])
        answer = await self.store.get_chat_event(chat_id=1, message_id=101)
        self.assertEqual(answer.reply_to_message_id, 10)
        self.assertTrue(self.agent.claim_conversation_followup(1, 7, "а задержка?"))

    async def test_silence_does_not_send_or_open_episode(self):
        await self.agent._execute_actions(MessageBatch(1, [self.person]), [AgentAction("stay_silent", {})])
        self.bot.send_message.assert_not_awaited()
        self.assertFalse(self.agent.claim_conversation_followup(1, 7, "привет"))

    async def test_complete_turn_uses_quote_and_rejects_duplicate_reply(self):
        event = replace(self.person, reply_to_message_id=9, reply_text="очередь", reply_sender="Коля")
        reply = AgentAction("reply_to_message", {"target_message_id": 10, "text": "очередь сглаживает наплыв"})
        self.assistant.plan_group_actions.return_value = [reply, reply]
        with patch("bot.group_agent.ChatActionSender.typing", no_chat_action):
            await self.agent._process_locked(MessageBatch(1, [event]))
        prompt = self.assistant.plan_group_actions.call_args.kwargs["new_messages"]
        self.assertIn("Коля: очередь", prompt)
        self.bot.send_message.assert_awaited_once()
        self.assertTrue(self.agent.claim_conversation_followup(1, 7, "почему?"))
        self.assertFalse(self.agent.claim_conversation_followup(1, 8, "это другой разговор"))

    async def test_legacy_private_context_uses_same_tool_agent(self):
        from collections import defaultdict
        import asyncio

        await self.store.append_exchange(chat_id=7, user_id=7, user_text="я люблю игры", model_text="какие?")
        bot = Bot("123456:abcdefghijklmnopqrstuvwxyz123456789")
        message = Message.model_validate({"message_id": 10, "date": 0, "chat": {"id": 7, "type": "private"},
            "from": {"id": 7, "is_bot": False, "first_name": "Вася"}, "text": "ответь голосом"}, context={"bot": bot})
        self.agent.process = AsyncMock()
        try:
            await _handle_private_message(message=message, text=message.text, store=self.store, assistant=self.assistant,
                group_agent=self.agent, history_messages=10, locks=defaultdict(asyncio.Lock))
            batch = self.agent.process.call_args.args[0]
            self.assertTrue(batch.events[0].directed_to_bot)
            history = await self.store.get_recent_chat_events(chat_id=7, limit=10)
            self.assertEqual([event.role for event in history], ["user", "bot"])
        finally:
            await bot.session.close()


class MediaMetadataTests(unittest.TestCase):
    def test_animation_takes_precedence_over_document(self):
        message = Message.model_validate({"message_id": 1, "date": 0, "chat": {"id": 1, "type": "private"},
            "animation": {"file_id": "gif", "file_unique_id": "g", "width": 64, "height": 64, "duration": 2, "mime_type": "video/mp4"},
            "document": {"file_id": "doc", "file_unique_id": "d", "mime_type": "application/octet-stream"}})
        self.assertEqual(_attachment_metadata(message)[0], "gif")
        self.assertTrue(_event_from_message(message).mime_type.startswith("video/"))

    def test_raw_pcm_is_wrapped_but_mp3_is_not(self):
        data = wrap_pcm(b"\x00\x00" * 240, "audio/L16;rate=24000")
        with wave.open(io.BytesIO(data)) as stream:
            self.assertEqual(stream.getframerate(), 24000)
            self.assertEqual(stream.getnframes(), 240)
        self.assertEqual(wrap_pcm(b"ID3abc", "audio/mpeg"), b"ID3abc")


if __name__ == "__main__":
    unittest.main()
