"""
CrewAI-compatible LLM factory that mirrors the Vertex AI access pattern used
by ``langraph_executor.py`` (via ``services/llm.py``).

Behaviour:
  * **Local dev**   → calls the Vertex AI REST endpoint
    (``aiplatform.googleapis.com``) using the API key fetched from GCP Secret
    Manager (exposed as ``settings.VERTEX_API_KEY`` by ``config.py``).
  * **Production**  → uses LiteLLM's ``vertex_ai/`` provider with Application
    Default Credentials — the Compute Engine VM already has Vertex AI access.
  * **Fallback**    → on any primary failure the LangChain-based wrapper
    falls back to OpenAI via ``services.llm.get_llm`` (same as langraph).

No API keys are defined or read from environment variables here — all
credentials come from ``settings`` (which sources them from Secret Manager /
ADC).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

from crewai import LLM
from crewai.llms.base_llm import BaseLLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from config import settings
from services.llm import get_llm as get_langchain_llm

logger = logging.getLogger(__name__)

_DEFAULT_VERTEX_MODEL = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash")
_GCP_PROJECT = (
    os.getenv("BIGQUERY_PROJECT_ID")
    or settings.GCP_PROJECT_ID
    or "tools-universe-29042025"
)
_GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
_IS_LOCAL = (os.getenv("DEPLOY_ENV", "local") or "local").strip().lower() == "local"


# ── CrewAI BaseLLM that delegates to the LangChain Vertex client ────────────


class _LangChainBridgeLLM(BaseLLM):
    """Adapts a LangChain ``Runnable`` chat model (Vertex REST primary +
    OpenAI fallback, identical to the one used by ``langraph_executor``) to
    CrewAI's ``BaseLLM`` interface.

    All credential handling lives in ``services.llm.get_llm`` — this class is
    pure plumbing.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_VERTEX_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
    ) -> None:
        super().__init__(model=f"vertex_ai/{model}", temperature=temperature)
        self._lc_model = get_langchain_llm(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        # CrewAgentExecutor reads these attributes directly.
        self.stream = False
        self._context_window_size = 1_000_000  # Gemini 2.5 Flash ≈ 1M tokens

    # ── CrewAI BaseLLM API ───────────────────────────────────────────────

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return self._context_window_size

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Single-turn CrewAI LLM call with internal tool-call loop.

        CrewAI expects ``call()`` to return the final assistant text, so any
        function calls returned by the model are resolved here before
        returning to the caller.
        """
        lc_messages = self._to_lc_messages(messages)

        model: Any = self._lc_model
        if tools:
            try:
                model = self._lc_model.bind_tools(tools)
            except Exception as exc:
                logger.warning(
                    "bind_tools failed (%s); calling LLM without tool schemas.",
                    exc,
                )
                model = self._lc_model

        max_iters = 10
        for _ in range(max_iters):
            response = model.invoke(lc_messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls or not available_functions:
                return self._extract_text(response)

            lc_messages.append(response)
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args", {}) if isinstance(tc, dict) else (getattr(tc, "args", {}) or {})
                call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

                fn = available_functions.get(name) if name else None
                if fn is None:
                    tool_output: Any = f"Function '{name}' is not available."
                else:
                    try:
                        tool_output = fn(**args) if isinstance(args, dict) else fn(args)
                    except Exception as exc:
                        tool_output = f"Error executing {name}: {exc}"

                lc_messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=call_id or name or "tool",
                        name=name or "",
                    )
                )

        # Tool loop exhausted — return last AI text we managed to produce.
        for msg in reversed(lc_messages):
            if isinstance(msg, AIMessage):
                return self._extract_text(msg)
        return ""

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content) if content is not None else ""

    @staticmethod
    def _to_lc_messages(messages: Union[str, List[Dict[str, Any]]]) -> List[BaseMessage]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        out: List[BaseMessage] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                out.append(SystemMessage(content=content or ""))
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    out.append(AIMessage(content=content or "", tool_calls=tool_calls))
                else:
                    out.append(AIMessage(content=content or ""))
            elif role == "tool":
                out.append(
                    ToolMessage(
                        content=content if isinstance(content, str) else str(content),
                        tool_call_id=msg.get("tool_call_id") or msg.get("name", "tool"),
                        name=msg.get("name", ""),
                    )
                )
            else:
                out.append(HumanMessage(content=content or ""))
        return out


# ── Public factories ────────────────────────────────────────────────────────


def get_crewai_llm(
    *,
    model: str = _DEFAULT_VERTEX_MODEL,
    temperature: float = 0.9,
    max_output_tokens: int = 8192,
) -> BaseLLM:
    """Return a CrewAI-compatible LLM that uses Vertex AI.

    Identical credential flow to ``langraph_executor.py``:
      * Local      — Vertex AI REST + API key from Secret Manager.
      * Production — Vertex AI via LiteLLM/ADC (Compute Engine credentials).
    """
    if _IS_LOCAL:
        return _LangChainBridgeLLM(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    # Production: route directly through CrewAI's LiteLLM-backed LLM with ADC.
    os.environ.setdefault("VERTEXAI_PROJECT", _GCP_PROJECT)
    os.environ.setdefault("VERTEXAI_LOCATION", _GCP_LOCATION)
    return LLM(
        model=f"vertex_ai/{model}",
        temperature=temperature,
        max_tokens=max_output_tokens,
    )


def _get_adc_access_token() -> str:
    """Mint a short-lived OAuth 2.0 access token from Application Default
    Credentials (Compute Engine metadata server in production).

    Chroma's ``GoogleVertexEmbeddingFunction`` only authenticates via an
    ``Authorization: Bearer <token>`` header — it has no native ADC support —
    so we resolve ADC ourselves and hand it a fresh access token. This avoids
    needing a static ``CHROMA_GOOGLE_VERTEX_API_KEY`` secret.
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token or ""


def get_crewai_embedder() -> Dict[str, Any]:
    """Return a CrewAI embedder config that uses Vertex AI.

    Local picks up ``settings.VERTEX_API_KEY`` (loaded from Secret Manager)
    and uses the Generative Language embedding endpoint. Production mints a
    short-lived OAuth token from ADC and passes it to the ``vertexai`` provider
    as its bearer credential — no static ``CHROMA_GOOGLE_VERTEX_API_KEY`` needed.

    Note: this must be called per-execution (not cached at import time) because
    the minted access token is short-lived (~1 hour).
    """
    if _IS_LOCAL:
        return dict(
            provider="google",
            config=dict(
                model="models/gemini-embedding-001",
                api_key=(settings.VERTEX_API_KEY or "").strip(),
            ),
        )

    return dict(
        provider="vertexai",
        config=dict(
            model="text-embedding-005",
            project_id=_GCP_PROJECT,
            region=_GCP_LOCATION,
            api_key=_get_adc_access_token(),
        ),
    )
