"""LiteLLM-backed generation with an explicit mock driver for CI reproducibility."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import litellm
from litellm import acompletion

from mega_ai.core.hashing import sha256_text
from mega_ai.core.settings import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.openai_api_key is not None:
            litellm.api_key = settings.openai_api_key.get_secret_value()

    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        agent_id: str,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], int, int]:
        """Returns (parsed_json, prompt_tokens, completion_tokens)."""
        mode = self._settings.llm_mode.lower()
        if mode == "mock":
            return self._mock_json(messages, agent_id)

        model = self._settings.litellm_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if model.startswith("ollama/") and self._settings.ollama_api_base:
            kwargs["api_base"] = self._settings.ollama_api_base

        response = await acompletion(**kwargs)
        content = response.choices[0].message.content or "{}"
        usage = response.usage
        prompt_tokens = int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0
        completion_tokens = int(usage.completion_tokens) if usage and usage.completion_tokens else 0
        return json.loads(content), prompt_tokens, completion_tokens

    async def complete_text_stream(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> AsyncIterator[tuple[str, int, int]]:
        mode = self._settings.llm_mode.lower()
        if mode == "mock":
            text = self._mock_stream_text(messages)
            for ch in text:
                yield ch, 0, 0
            return

        model = self._settings.litellm_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if model.startswith("ollama/") and self._settings.ollama_api_base:
            kwargs["api_base"] = self._settings.ollama_api_base

        stream = await acompletion(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None) or ""
            if piece:
                u = getattr(chunk, "usage", None)
                pt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
                ct = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
                yield piece, pt, ct

    def _mock_json(self, messages: list[dict[str, str]], agent_id: str) -> tuple[dict[str, Any], int, int]:
        joined = "\n".join(m["content"] for m in messages)
        seed = sha256_text(joined + agent_id)
        if agent_id == "orchestrator_router":
            return (
                {
                    "chosen_agents": ["decomposition", "retrieval", "critique", "synthesis"],
                    "order": ["decomposition", "retrieval", "critique", "synthesis"],
                    "reason": "mock_router: default staged pipeline for reproducible CI.",
                    "confidence": 0.82,
                    "context_budget_tokens": {
                        "decomposition": 2048,
                        "retrieval": 4096,
                        "critique": 3072,
                        "synthesis": 4096,
                        "compression": 1024,
                    },
                },
                120,
                180,
            )
        if agent_id == "decomposition":
            return (
                {
                    "tasks": [
                        {"id": "t1", "type": "clarify", "depends_on": []},
                        {"id": "t2", "type": "retrieve", "depends_on": ["t1"]},
                    ],
                    "notes": "mock decomposition graph",
                },
                200,
                220,
            )
        if agent_id == "retrieval":
            return (
                {
                    "hop1_query": "mock hop1",
                    "hop2_query": "mock hop2",
                    "claims": [
                        {"text": "Paris is the capital of France.", "chunk_ids": ["c1", "c2"]},
                    ],
                },
                220,
                260,
            )
        if agent_id == "critique":
            return (
                {
                    "spans": [
                        {
                            "claim_span": "Paris is the capital of France.",
                            "confidence": 0.9,
                            "disagree": False,
                            "suggested_correction": None,
                        }
                    ],
                    "overall_notes": "mock critique",
                },
                180,
                200,
            )
        if agent_id == "synthesis":
            return (
                {
                    "final_answer": "Mock synthesized answer with deterministic provenance.",
                    "provenance": [
                        {
                            "sentence": "Mock synthesized answer with deterministic provenance.",
                            "source_agent": "retrieval",
                            "source_chunk_id": "c1",
                        }
                    ],
                },
                240,
                260,
            )
        if agent_id == "compression":
            return {"compressed_filler": "[compressed filler]"}, 50, 40
        if agent_id == "meta_prompt_optimizer":
            return (
                {
                    "weakest_prompt_key": "orchestrator_system",
                    "proposed_rewrite": "You are the orchestrator. Route agents explicitly.",
                    "diff": "- old\n+ new",
                    "justification": "mock meta-agent justification",
                },
                300,
                200,
            )
        if agent_id == "nl2sql":
            return {"sql": "SELECT 1 AS ok"}, 100, 40
        return {"result": f"mock:{seed[:8]}", "echo_agent": agent_id}, 50, 50

    def _mock_stream_text(self, messages: list[dict[str, str]]) -> str:
        """Deterministic short stream for mock mode."""
        topic = messages[-1]["content"] if messages else ""
        digest = sha256_text(topic)[:8]
        return f"[mock-stream:{digest}] {topic[:200]}"
