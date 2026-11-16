from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import os
from pathlib import Path
import shutil
from time import monotonic, perf_counter
from typing import Any, Awaitable, Callable

from google.genai import types


logger = logging.getLogger(__name__)

_ISOLATED_CLI_SETTINGS = {
    "allowNonWorkspaceAccess": False,
    "toolPermission": "strict",
    "permissions": {
        "deny": [
            "read_file(*)",
            "write_file(*)",
            "command(*)",
            "read_url(*)",
            "execute_url(*)",
            "mcp(*)",
        ]
    },
}

_ISOLATED_GEMINI_SETTINGS = {
    "security": {
        "auth": {
            "selectedType": "oauth-personal",
        }
    }
}

_AUTH_FILES = (
    "oauth_creds.json",
    "google_accounts.json",
    "installation_id",
    "state.json",
)


class AntigravityClient:
    def __init__(
        self,
        *,
        model: str = "gemini-3.8-flash-low",
        effort: str = "low",
        sandbox_root: Path | None = None,
        source_gemini_dir: Path | None = None,
        max_rpm: int = 8,
        cooldown_seconds: float = 600.0,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.model = model
        self.effort = effort
        self._sandbox_root = (
            sandbox_root
            if sandbox_root is not None
            else Path("data") / ".antigravity_home"
        )
        self._source_gemini_dir = (
            source_gemini_dir
            if source_gemini_dir is not None
            else Path.home() / ".gemini"
        )
        self._max_rpm = max(1, max_rpm)
        self._cooldown_seconds = cooldown_seconds
        self._timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(2)
        self._recent_calls: deque[float] = deque()
        self._disabled_until: float = 0.0
        self._Executable = shutil.which("agy") or "agy"

    def is_available(self) -> bool:
        creds = self._source_gemini_dir / "oauth_creds.json"
        return creds.is_file() and bool(shutil.which("agy"))

    def _check_guardrails(self) -> None:
        now = monotonic()
        if now < self._disabled_until:
            remaining = int(self._disabled_until - now)
            raise RuntimeError(
                f"Antigravity временно на паузе после ошибки лимита ({remaining}с)"
            )
        while self._recent_calls and now - self._recent_calls[0] > 60.0:
            self._recent_calls.popleft()
        if len(self._recent_calls) >= self._max_rpm:
            raise RuntimeError(
                f"Превышен безопасный лимит подписки ({self._max_rpm} запр/мин)"
            )

    def _prepare_isolated_home(self) -> tuple[Path, Path]:
        home_dir = self._sandbox_root.resolve()
        gemini_dir = home_dir / ".gemini"
        cli_dir = gemini_dir / "antigravity-cli"
        work_dir = home_dir / "workspace"
        cli_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        for filename in _AUTH_FILES:
            src = self._source_gemini_dir / filename
            dst = gemini_dir / filename
            if src.is_file():
                if not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(src, dst)

        gemini_settings = gemini_dir / "settings.json"
        if not gemini_settings.is_file():
            gemini_settings.write_text(
                json.dumps(_ISOLATED_GEMINI_SETTINGS, ensure_ascii=False),
                encoding="utf-8",
            )

        cli_settings = cli_dir / "settings.json"
        cli_settings.write_text(
            json.dumps(_ISOLATED_CLI_SETTINGS, ensure_ascii=False),
            encoding="utf-8",
        )
        return home_dir, cli_settings

    async def _run_cli(
        self,
        *,
        prompt: str,
        operation: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._check_guardrails()
        async with self._semaphore:
            self._check_guardrails()
            self._recent_calls.append(monotonic())
            home_dir, cli_settings = await asyncio.to_thread(
                self._prepare_isolated_home
            )
            work_dir = home_dir / "workspace"

            args = [
                self._Executable,
                "--model",
                self.model,
                "--effort",
                self.effort,
                "--disable-slash-commands",
                "--sandbox",
                "--settings",
                str(cli_settings),
                "--output-format",
                "json",
            ]
            schema_file: Path | None = None
            if json_schema is not None:
                schema_file = home_dir / f"schema_{operation}.json"
                await asyncio.to_thread(
                    schema_file.write_text,
                    json.dumps(json_schema, ensure_ascii=False),
                    "utf-8",
                )
                args.extend(["--json-schema", str(schema_file)])

            args.append(f"--print={prompt}")

            env = os.environ.copy()
            env["USERPROFILE"] = str(home_dir)
            env["HOME"] = str(home_dir)

            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(work_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise RuntimeError("Таймаут вызова Antigravity CLI")

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            combined = f"{stdout_text}\n{stderr_text}"

            if (
                "429" in combined
                or "RESOURCE_EXHAUSTED" in combined
                or "Rate limit" in combined
                or "quota" in combined.casefold()
            ):
                self._disabled_until = monotonic() + self._cooldown_seconds
                logger.warning(
                    "Antigravity rate limit detected; pausing subscription channel for %ds",
                    int(self._cooldown_seconds),
                )
                raise RuntimeError("Лимит запросов Antigravity (429)")

            if proc.returncode != 0 and not stdout_text.startswith("{"):
                raise RuntimeError(
                    f"Antigravity CLI завершился с кодом {proc.returncode}: {stderr_text[:300]}"
                )

            payload = _parse_json_line(stdout_text)
            usage = payload.get("usage")
            if isinstance(usage, dict):
                logger.info(
                    "llm_usage model=%s operation=%s input_tokens=%s output_tokens=%s total_tokens=%s",
                    self.model,
                    operation,
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("total_tokens"),
                )
            return payload

    async def reply(
        self,
        *,
        system_prompt: str,
        history: list[tuple[str, str]],
        user_text: str,
    ) -> str:
        history_lines = []
        for role, content in history:
            speaker = "Ассистент" if role == "model" else "Пользователь"
            history_lines.append(f"{speaker}: {content}")
        history_block = "\n".join(history_lines)

        full_prompt = (
            f"СИСТЕМНАЯ ИНСТРУКЦИЯ:\n{system_prompt}\n\n"
            + (f"ИСТОРИЯ ДИАЛОГА:\n{history_block}\n\n" if history_block else "")
            + f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{user_text}\n\n"
            "Ответь только текстом реплики ассистента без пояснений."
        )
        payload = await self._run_cli(prompt=full_prompt, operation="reply")
        answer = str(payload.get("response") or "").strip()
        if not answer:
            raise RuntimeError("Antigravity вернул пустой ответ")
        return answer

    async def plan_group_actions(
        self,
        *,
        system_prompt: str,
        prompt: str,
        declarations: list[types.FunctionDeclaration],
        lookup_tools: set[str],
        tool_executor: Callable[[Any], Awaitable[dict[str, Any]]],
        action_factory: Callable[[str, dict[str, Any]], Any],
    ) -> list[Any]:
        tool_names = [decl.name for decl in declarations if decl.name]
        tool_Catalog = _format_tool_catalog(declarations)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": tool_names or ["stay_silent"],
                            },
                            "arguments": {
                                "type": "object",
                            },
                        },
                        "required": ["name", "arguments"],
                    },
                }
            },
            "required": ["actions"],
        }

        tool_history: list[str] = []
        for _ in range(3):
            history_section = (
                "\n\nРЕЗУЛЬТАТЫ ВЫЗВАННЫХ ИНСТРУМЕНТОВ:\n" + "\n".join(tool_history)
                if tool_history
                else ""
            )
            full_prompt = (
                f"СИСТЕМНАЯ ИНСТРУКЦИЯ:\n{system_prompt}\n\n"
                f"ДОСТУПНЫЕ ИНСТРУМЕНТЫ (выбирай только из них и передавай точные arguments):\n{tool_Catalog}\n\n"
                f"{prompt}{history_section}\n\n"
                "Верни JSON строго по схеме: {\"actions\": [{\"name\": \"...\", \"arguments\": {...}}]}. "
                "Не используй свои встроенные инструменты CLI, только верни JSON."
            )
            payload = await self._run_cli(
                prompt=full_prompt,
                operation="group",
                json_schema=schema,
            )
            raw_actions = _extract_actions(payload)
            actions = [
                action_factory(name, args)
                for name, args in raw_actions
                if name in tool_names
            ]
            if not actions:
                return [action_factory("stay_silent", {"reason": "no tool call"})]
            if not any(action.name in lookup_tools for action in actions):
                return actions

            for action in actions:
                if action.name in lookup_tools:
                    try:
                        result = await tool_executor(action)
                    except Exception as error:
                        result = {"ok": False, "error": str(error)[:500]}
                else:
                    result = {
                        "ok": False,
                        "error": "Финальное действие смешано с поиском; вызови его следующим ходом.",
                    }
                tool_history.append(
                    f"- {action.name}({json.dumps(action.arguments, ensure_ascii=False)}) -> "
                    f"{json.dumps(result, ensure_ascii=False)}"
                )

        return [action_factory("stay_silent", {"reason": "tool loop limit"})]


def _format_tool_catalog(declarations: list[types.FunctionDeclaration]) -> str:
    lines: list[str] = []
    for decl in declarations:
        if not decl.name:
            continue
        params = decl.parameters_json_schema or {}
        lines.append(
            f"- {decl.name}: {decl.description or ''} | params={json.dumps(params, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def _parse_json_line(stdout_text: str) -> dict[str, Any]:
    for line in reversed(stdout_text.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    parsed = json.loads(stdout_text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Antigravity CLI вернул некорректный JSON")
    return parsed


def _extract_actions(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    structured = payload.get("structured_output")
    if not isinstance(structured, dict):
        raw_response = str(payload.get("response") or "").strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.removeprefix("```json").removeprefix("```")
            raw_response = raw_response.removesuffix("```").strip()
        structured = json.loads(raw_response) if raw_response else {}

    items = structured.get("actions") if isinstance(structured, dict) else None
    if not isinstance(items, list):
        return []

    result: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        if name:
            result.append((name, arguments))
    return result
