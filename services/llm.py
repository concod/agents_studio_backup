"""
LLM wrapper layer for LangChain-based agents (LangGraph, etc.).

Provides a unified factory (`get_llm`) that returns a LangChain chat model
with automatic fallback:

    Primary  → Vertex AI REST API  (API key, local dev only)
               ChatVertexAI        (ADC, deployed environments)
    Fallback → OpenAI              (ChatOpenAI)

Usage:
    from services.llm import get_llm

    intent_model = get_llm(temperature=0.0, max_output_tokens=512)
    instruction_model = get_llm(temperature=0.2, max_output_tokens=16384)
"""

import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from config import settings

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────
_DEFAULT_VERTEX_MODEL = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash")
_DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")
_DEFAULT_GCP_PROJECT = (
    os.getenv("BIGQUERY_PROJECT_ID")
    or settings.GCP_PROJECT_ID
    or "tools-universe-29042025"
)
_DEFAULT_GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
_IS_LOCAL = os.getenv("DEPLOY_ENV", "local").strip().lower() == "local"


# ── Vertex AI REST Chat Model (local dev only) ─────────────────────────────


class _VertexRestChatModel(BaseChatModel):
    """LangChain ``BaseChatModel`` that calls the Vertex AI REST API with an
    API key (``?key=…``).

    This bypasses LangChain's Vertex / Gemini integrations — which either
    route to the blocked ``generativelanguage.googleapis.com`` or ignore the
    API key in favour of ADC — and talks to
    ``aiplatform.googleapis.com`` directly.

    **Only used for local development.**
    """

    api_key: str = ""
    project: str = ""
    location: str = "global"
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 1024

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "vertex-rest-api-key"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "project": self.project,
            "location": self.location,
        }

    def _url(self) -> str:
        return (
            f"https://aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/google/models/"
            f"{self.model_name}:generateContent"
        )

    # ── message → Vertex REST conversion ────────────────────────────────

    @staticmethod
    def _content_to_parts(content: Any) -> list:
        if isinstance(content, str):
            return [{"text": content}]
        parts: list = []
        for item in content:
            if isinstance(item, str):
                parts.append({"text": item})
            elif isinstance(item, dict):
                t = item.get("type")
                if t == "text":
                    parts.append({"text": item["text"]})
                elif t == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        header, b64data = url.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                        parts.append(
                            {"inlineData": {"mimeType": mime, "data": b64data}}
                        )
        return parts or [{"text": ""}]

    def _messages_to_vertex(
        self, messages: List[BaseMessage]
    ) -> tuple:
        """Return ``(system_instruction | None, contents_list)``."""
        system_instruction: Optional[str] = None
        contents: list = []
        pending_tool_parts: list = []

        def _flush():
            nonlocal pending_tool_parts
            if pending_tool_parts:
                contents.append({"role": "user", "parts": pending_tool_parts})
                pending_tool_parts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
            elif isinstance(msg, HumanMessage):
                _flush()
                contents.append(
                    {"role": "user", "parts": self._content_to_parts(msg.content)}
                )
            elif isinstance(msg, AIMessage):
                _flush()
                parts: list = []
                if msg.content:
                    parts.append(
                        {
                            "text": msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        }
                    )
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tc["name"],
                                    "args": tc.get("args", {}),
                                }
                            }
                        )
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif isinstance(msg, ToolMessage):
                import json as _json

                name = getattr(msg, "name", None) or getattr(
                    msg, "tool_call_id", "tool"
                )
                try:
                    result = (
                        _json.loads(msg.content)
                        if isinstance(msg.content, str)
                        else msg.content
                    )
                except Exception:
                    result = {"result": msg.content}
                pending_tool_parts.append(
                    {"functionResponse": {"name": name, "response": result}}
                )

        _flush()
        return system_instruction, contents

    # ── tool conversion ─────────────────────────────────────────────────

    @staticmethod
    def _tools_to_vertex(tools: list) -> list:
        """OpenAI-format tools → Vertex ``functionDeclarations``."""
        if not tools:
            return []
        decls: list = []
        for tool in tools:
            if isinstance(tool, dict):
                if "function" in tool:
                    func = tool["function"]
                    fd: Dict[str, Any] = {"name": func["name"]}
                    if func.get("description"):
                        fd["description"] = func["description"]
                    if func.get("parameters"):
                        fd["parameters"] = func["parameters"]
                    decls.append(fd)
                elif "name" in tool:
                    decls.append(tool)
        return [{"functionDeclarations": decls}] if decls else []

    @staticmethod
    def _tool_config_to_vertex(tc: Optional[dict]) -> Optional[dict]:
        if not tc:
            return None
        fcc = tc.get("function_calling_config")
        return {"functionCallingConfig": fcc} if fcc else tc

    # ── request / response ──────────────────────────────────────────────

    def _build_body(self, messages: List[BaseMessage], **kwargs: Any) -> dict:
        sys_instr, contents = self._messages_to_vertex(messages)
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if sys_instr:
            body["systemInstruction"] = {"parts": [{"text": sys_instr}]}

        tools = kwargs.get("tools")
        if tools:
            vt = self._tools_to_vertex(tools)
            if vt:
                body["tools"] = vt

        tc = self._tool_config_to_vertex(kwargs.get("tool_config"))
        if tc:
            body["toolConfig"] = tc

        stop = kwargs.get("stop")
        if stop:
            body["generationConfig"]["stopSequences"] = stop

        return body

    def _parse_response(self, data: dict) -> ChatResult:
        candidates = data.get("candidates", [])
        if not candidates:
            # Log why there are no candidates (safety block, etc.)
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            safety_ratings = data.get("promptFeedback", {}).get("safetyRatings", [])
            logger.warning(
                "[VertexREST] No candidates in response. blockReason=%s, safetyRatings=%s, keys=%s",
                block_reason,
                safety_ratings,
                list(data.keys()),
            )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=""))]
            )

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", [])

        # Log if response was filtered or empty
        if finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
            safety = candidate.get("safetyRatings", [])
            logger.warning(
                "[VertexREST] Unexpected finishReason=%s, safetyRatings=%s",
                finish_reason,
                safety,
            )

        text_parts: list = []
        tool_calls: list = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    {
                        "name": fc["name"],
                        "args": fc.get("args", {}),
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "tool_call",
                    }
                )

        content = "".join(text_parts)
        msg = (
            AIMessage(content=content, tool_calls=tool_calls)
            if tool_calls
            else AIMessage(content=content)
        )

        gen_info: Dict[str, Any] = {}
        usage = data.get("usageMetadata")
        if usage:
            gen_info["usage_metadata"] = {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            }
        fr = candidate.get("finishReason")
        if fr:
            gen_info["finish_reason"] = fr

        return ChatResult(
            generations=[ChatGeneration(message=msg, generation_info=gen_info)]
        )

    # ── LangChain _generate / _agenerate ────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        body = self._build_body(messages, stop=stop, **kwargs)
        logger.debug(
            "[VertexREST] Request: model=%s, contents_count=%d, has_tools=%s",
            self.model_name,
            len(body.get("contents", [])),
            "tools" in body,
        )
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                self._url(), json=body, params={"key": self.api_key}
            )
            if not resp.is_success:
                raise RuntimeError(
                    f"Vertex AI REST error {resp.status_code}: {resp.text}"
                )
            resp_data = resp.json()
            result = self._parse_response(resp_data)
            content = result.generations[0].message.content if result.generations else ""
            if not content:
                logger.warning(
                    "[VertexREST] Empty response from %s. Raw keys=%s, candidates=%d",
                    self.model_name,
                    list(resp_data.keys()),
                    len(resp_data.get("candidates", [])),
                )
            return result

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        body = self._build_body(messages, stop=stop, **kwargs)
        logger.debug(
            "[VertexREST] Async request: model=%s, contents_count=%d, has_tools=%s",
            self.model_name,
            len(body.get("contents", [])),
            "tools" in body,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                self._url(), json=body, params={"key": self.api_key}
            )
            if not resp.is_success:
                raise RuntimeError(
                    f"Vertex AI REST error {resp.status_code}: {resp.text}"
                )
            resp_data = resp.json()
            result = self._parse_response(resp_data)
            content = result.generations[0].message.content if result.generations else ""
            if not content:
                import json as _json
                logger.warning(
                    "[VertexREST] Empty response from %s. Full response: %s",
                    self.model_name,
                    _json.dumps(resp_data, indent=2)[:2000],
                )
            return result

    # ── tool binding ────────────────────────────────────────────────────

    def bind_tools(self, tools: list, **kwargs: Any) -> Runnable:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)


# ── Logging fallback wrapper ────────────────────────────────────────────────


class _LoggingFallbackLLM(Runnable):
    """Tries primary LLM first, logs the exact failure reason, then falls
    back to a secondary LLM.  Delegates ``bind_tools`` /
    ``with_structured_output`` to both inner models so tool-calling and
    structured output work on either path.
    """

    def __init__(self, primary: Any, fallback: Any, primary_label: str):
        self._primary = primary
        self._fallback = fallback
        self._primary_label = primary_label

    # ── invoke / ainvoke / batch ─────────────────────────────────────────

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs) -> Any:
        try:
            return self._primary.invoke(input, config=config, **kwargs)
        except Exception as exc:
            logger.warning(
                "[LLM Fallback] %s FAILED → OpenAI. Reason: %s: %s",
                self._primary_label, type(exc).__name__, exc,
            )
            return self._fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        try:
            return await self._primary.ainvoke(input, config=config, **kwargs)
        except Exception as exc:
            logger.warning(
                "[LLM Fallback] %s FAILED → OpenAI. Reason: %s: %s",
                self._primary_label, type(exc).__name__, exc,
            )
            return await self._fallback.ainvoke(input, config=config, **kwargs)

    def batch(self, inputs, config=None, **kwargs):
        try:
            return self._primary.batch(inputs, config=config, **kwargs)
        except Exception as exc:
            logger.warning(
                "[LLM Fallback] %s batch FAILED → OpenAI. Reason: %s: %s",
                self._primary_label, type(exc).__name__, exc,
            )
            return self._fallback.batch(inputs, config=config, **kwargs)

    # ── streaming ────────────────────────────────────────────────────────

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs
    ) -> AsyncIterator:
        try:
            async for chunk in self._primary.astream(input, config=config, **kwargs):
                yield chunk
        except Exception as exc:
            logger.warning(
                "[LLM Fallback] %s stream FAILED → OpenAI. Reason: %s: %s",
                self._primary_label, type(exc).__name__, exc,
            )
            async for chunk in self._fallback.astream(input, config=config, **kwargs):
                yield chunk

    # ── delegation helpers ───────────────────────────────────────────────

    def bind_tools(self, *args, **kwargs):
        new_primary = self._primary.bind_tools(*args, **kwargs)
        # Strip Vertex-specific kwargs before forwarding to OpenAI
        fallback_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("tool_config",)
        }
        new_fallback = self._fallback.bind_tools(*args, **fallback_kwargs)
        return _LoggingFallbackLLM(new_primary, new_fallback, self._primary_label)

    def with_structured_output(self, *args, **kwargs):
        new_primary = self._primary.with_structured_output(*args, **kwargs)
        new_fallback = self._fallback.with_structured_output(*args, **kwargs)
        return _LoggingFallbackLLM(new_primary, new_fallback, self._primary_label)

    @property
    def InputType(self):
        return self._primary.InputType

    @property
    def OutputType(self):
        return self._primary.OutputType


# ── Public factory ──────────────────────────────────────────────────────────


def get_llm(
    *,
    model: str = _DEFAULT_VERTEX_MODEL,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    max_retries: int = 2,
    fallback_model: str = _DEFAULT_OPENAI_MODEL,
):
    """Return a Vertex AI chat model with an OpenAI fallback.

    The returned object is a LangChain ``Runnable`` — pipe it into prompt
    chains, call ``.invoke()`` / ``.ainvoke()``, use ``.bind_tools()`` and
    ``.with_structured_output()`` transparently.

    **Local dev** uses a direct REST call to
    ``aiplatform.googleapis.com?key=…`` — no ADC required.

    **Deployed** uses ``ChatVertexAI`` with ADC (unchanged).
    """
    if _IS_LOCAL:
        api_key = (settings.VERTEX_API_KEY or "").strip()
        logger.info(
            "VERTEX_API_KEY debug: length=%d, starts_with=%s",
            len(api_key),
            repr(api_key[:10]) if api_key else "<empty>",
        )
        primary = _VertexRestChatModel(
            api_key=api_key,
            project=_DEFAULT_GCP_PROJECT,
            location="global",
            model_name=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        primary_label = f"VertexRestAPI(model={model})"
    else:
        primary = ChatVertexAI(
            model_name=model,
            project=_DEFAULT_GCP_PROJECT,
            location=_DEFAULT_GCP_LOCATION,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
        )
        primary_label = f"ChatVertexAI(model={model}, ADC)"

    fallback = ChatOpenAI(
        model=fallback_model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        api_key=settings.OPENAI_API_KEY,
        max_retries=max_retries,
    )

    llm_with_fallback = _LoggingFallbackLLM(primary, fallback, primary_label)

    logger.info(
        "LLM initialised: primary=%s, fallback=%s (OpenAI)",
        primary_label,
        fallback_model,
    )
    return llm_with_fallback
