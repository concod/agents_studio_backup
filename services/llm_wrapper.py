"""
LLM Wrapper with Vertex AI (Gemini) as primary and OpenAI as fallback.

Uses Vertex AI via the ``google-genai`` SDK with Application Default
Credentials (ADC) — no API key required when running on an authorised
GCP VM.

Usage:
    from services.llm_wrapper import llm_client

    # Async text generation (most common)
    text = await llm_client.generate(prompt, temperature=0.3, max_tokens=800)

    # Sync text generation
    text = llm_client.generate_sync(prompt, temperature=0.0, max_tokens=1024)

    # With multimodal parts (e.g., PDF)
    from google.genai import types
    text = llm_client.generate_sync(
        prompt,
        parts=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")],
        temperature=0.0,
        max_tokens=65536,
    )

    # Override model
    text = await llm_client.generate(prompt, model="gemini-2.5-flash-lite")
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OPENAI_BASE = "https://api.openai.com/v1/chat/completions"

_DEFAULT_VERTEX_MODEL = settings.VERTEX_AI_MODEL or "gemini-2.5-flash"
_DEFAULT_OPENAI_MODEL = settings.OPENAI_FALLBACK_MODEL or "gpt-4o"

_GCP_PROJECT_ID = settings.BIGQUERY_PROJECT_ID or settings.GCP_PROJECT_ID or "tools-universe-29042025"
_GCP_LOCATION = settings.GCP_LOCATION or "us-central1"
_IS_LOCAL = (settings.DEPLOY_ENV or "local").strip().lower() == "local"

# Map Gemini models → OpenAI fallback equivalents
_OPENAI_FALLBACK_MAP: Dict[str, str] = {
    "gemini-2.5-flash": "gpt-4o",
    "gemini-2.5-flash-lite": "gpt-4o",
    "gemini-2.5-pro": "gpt-4o",
    "gemini-2.5-flash-exp": "gpt-4o",
    "gemini-2.5-flash-image": "gpt-4o",
    "gemini-3-flash-preview": "gpt-4o",
    "gemini-3-pro-image-preview": "gpt-4o",
}


def _mask_key_for_logs(key: str) -> str:
    """Return a safe key fingerprint for debugging (first 4 + last 4 chars)."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


# ---------------------------------------------------------------------------
# REST-based Vertex AI wrapper (local dev with API key)
# ---------------------------------------------------------------------------

class _RestResponse:
    """Wraps Vertex AI REST JSON to match google-genai response interface."""

    __slots__ = ("_data",)

    def __init__(self, data: dict):
        self._data = data

    @property
    def text(self) -> str:
        candidates = self._data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p["text"] for p in parts if "text" in p]
        return "".join(texts)


class _VertexRestClient:
    """Lightweight REST client for Vertex AI with API-key auth.

    Routes through aiplatform.googleapis.com (Vertex AI endpoint) instead of
    generativelanguage.googleapis.com, avoiding org-policy blocks.

    Implements the subset of the genai.Client interface used by LLMWrapper:
      client.aio.models.generate_content(model=, contents=, config=)
      client.models.generate_content(model=, contents=, config=)
    """

    def __init__(self, api_key: str, project: str, location: str):
        self._api_key = api_key
        self._project = project
        self._location = location
        self.aio = self._Aio(self)
        self.models = self._SyncModels(self)

    def _url(self, model: str, method: str) -> str:
        return (
            f"https://aiplatform.googleapis.com/v1/projects/{self._project}"
            f"/locations/{self._location}/publishers/google/models/{model}:{method}"
        )

    @staticmethod
    def _config_to_gen_config(config) -> dict:
        """Convert a GenerateContentConfig object to REST generationConfig dict."""
        if config is None:
            return {}
        gc: dict = {}
        for attr, key in [
            ("max_output_tokens", "maxOutputTokens"),
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("top_k", "topK"),
        ]:
            v = getattr(config, attr, None)
            if v is not None:
                gc[key] = v
        return gc

    @staticmethod
    def _normalise_contents(contents) -> list:
        """Convert SDK Content/Part objects or plain strings to REST dicts."""
        import base64 as _b64

        if isinstance(contents, str):
            return [{"role": "user", "parts": [{"text": contents}]}]
        if not isinstance(contents, list):
            return [{"role": "user", "parts": [{"text": str(contents)}]}]

        result: list = []
        flat_parts: list = []

        for item in contents:
            if isinstance(item, str):
                flat_parts.append({"text": item})
            elif isinstance(item, dict) and "role" in item:
                if flat_parts:
                    result.append({"role": "user", "parts": flat_parts})
                    flat_parts = []
                result.append(item)
            elif hasattr(item, "role") and hasattr(item, "parts"):
                # SDK Content object
                if flat_parts:
                    result.append({"role": "user", "parts": flat_parts})
                    flat_parts = []
                parts = []
                for p in item.parts:
                    if hasattr(p, "text") and p.text:
                        parts.append({"text": p.text})
                    elif hasattr(p, "inline_data") and p.inline_data:
                        d = p.inline_data.data
                        if isinstance(d, bytes):
                            d = _b64.b64encode(d).decode()
                        parts.append({"inlineData": {"mimeType": p.inline_data.mime_type, "data": d}})
                result.append({"role": item.role, "parts": parts})
            elif hasattr(item, "inline_data") and item.inline_data:
                # SDK Part with image/binary data
                d = item.inline_data.data
                if isinstance(d, bytes):
                    d = _b64.b64encode(d).decode()
                flat_parts.append({"inlineData": {"mimeType": item.inline_data.mime_type, "data": d}})
            elif hasattr(item, "text") and item.text is not None:
                flat_parts.append({"text": item.text})
            elif isinstance(item, dict):
                flat_parts.append(item)

        if flat_parts:
            result.append({"role": "user", "parts": flat_parts})
        return result or [{"role": "user", "parts": [{"text": ""}]}]

    def _build_body(self, contents, config=None) -> dict:
        body: dict = {"contents": self._normalise_contents(contents)}
        if config is not None:
            gc = self._config_to_gen_config(config)
            if gc:
                body["generationConfig"] = gc
        return body

    class _Aio:
        def __init__(self, client: "_VertexRestClient"):
            self.models = _VertexRestClient._AsyncModels(client)

    class _AsyncModels:
        def __init__(self, client: "_VertexRestClient"):
            self._c = client

        async def generate_content(self, *, model: str, contents, config=None):
            import httpx as _httpx

            url = self._c._url(model, "generateContent")
            body = self._c._build_body(contents, config)
            async with _httpx.AsyncClient(timeout=120) as http:
                r = await http.post(url, json=body, params={"key": self._c._api_key})
                if not r.is_success:
                    raise RuntimeError(
                        f"Vertex AI REST generateContent {r.status_code}: {r.text}"
                    )
                return _RestResponse(r.json())

    class _SyncModels:
        def __init__(self, client: "_VertexRestClient"):
            self._c = client

        def generate_content(self, *, model: str, contents, config=None):
            import httpx as _httpx

            url = self._c._url(model, "generateContent")
            body = self._c._build_body(contents, config)
            with _httpx.Client(timeout=120) as http:
                r = http.post(url, json=body, params={"key": self._c._api_key})
                if not r.is_success:
                    raise RuntimeError(
                        f"Vertex AI REST generateContent {r.status_code}: {r.text}"
                    )
                return _RestResponse(r.json())


def _build_vertex_client():
    """Create a Vertex AI client.

    Local dev: uses _VertexRestClient with API key, routing through
               aiplatform.googleapis.com (Vertex AI endpoint).
    Deployed:  uses google-genai Client with ADC / service-account.
    """
    if _IS_LOCAL:
        api_key = (settings.VERTEX_API_KEY or "").strip()
        logger.info(
            "[LLMWrapper] VERTEX_API_KEY: length=%d, starts_with=%s, has_newline=%s, has_cr=%s, is_json=%s",
            len(api_key),
            repr(api_key[:10]) if api_key else "<empty>",
            "\n" in api_key,
            "\r" in api_key,
            api_key.startswith("{"),
        )
        # Route through aiplatform.googleapis.com (Vertex AI endpoint)
        # instead of generativelanguage.googleapis.com which is blocked
        # by org policy.
        return _VertexRestClient(
            api_key=api_key,
            project=_GCP_PROJECT_ID,
            location=_GCP_LOCATION,
        )

    # In deployed envs we must use Vertex ADC. If GEMINI_API_KEY leaks in from
    # the shell/runtime environment, google-genai emits a noisy warning when
    # both GOOGLE_API_KEY and GEMINI_API_KEY exist.
    if os.getenv("GEMINI_API_KEY"):
        logger.info("[LLMWrapper] Clearing GEMINI_API_KEY env var in Vertex ADC mode")
        os.environ.pop("GEMINI_API_KEY", None)

    return genai.Client(
        vertexai=True,
        project=_GCP_PROJECT_ID,
        location=_GCP_LOCATION,
    )


class LLMWrapper:
    """Unified LLM client: tries Vertex AI (Gemini) first, falls back to OpenAI."""

    def __init__(self) -> None:
        self._async_client: Optional[httpx.AsyncClient] = None
        self._vertex_client: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_vertex_client(self):
        if self._vertex_client is None:
            self._vertex_client = _build_vertex_client()
        return self._vertex_client

    def _get_openai_key(self) -> str:
        return settings.OPENAI_API_KEY or ""

    def _resolve_openai_model(self, gemini_model: str) -> str:
        return _OPENAI_FALLBACK_MAP.get(gemini_model, _DEFAULT_OPENAI_MODEL)

    def _get_async_http_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=50,
                    keepalive_expiry=30.0,
                ),
                http2=True,
            )
        return self._async_client

    # ------------------------------------------------------------------
    # Vertex AI (Gemini) helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_vertex_config(
        *,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )

    @staticmethod
    def _build_vertex_contents(
        prompt: str,
        parts: Optional[List[Any]] = None,
    ) -> list:
        """Build the contents list for Vertex AI generate_content."""
        content_parts: list = []
        if parts:
            content_parts.extend(parts)
        content_parts.append(prompt)
        return content_parts

    @staticmethod
    def _extract_vertex_text(response: Any) -> str:
        """Extract text from a Vertex AI GenerateContentResponse."""
        return (response.text or "").strip()

    # ------------------------------------------------------------------
    # OpenAI call helpers
    # ------------------------------------------------------------------
    def _build_openai_payload(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        top_p: Optional[float] = None,
    ) -> dict:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        return payload

    def _build_openai_vision_payload(
        self,
        prompt: str,
        *,
        image_base64: str,
        image_mime_type: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        top_p: Optional[float] = None,
    ) -> dict:
        """Build an OpenAI payload with an inline base64 image for vision models."""
        data_url = f"data:{image_mime_type};base64,{image_base64}"
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        return payload

    @staticmethod
    def _extract_openai_text(data: dict) -> str:
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content", "").strip()

    # ------------------------------------------------------------------
    # Async generate  (primary entry point)
    # ------------------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        model: str = _DEFAULT_VERTEX_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        parts: Optional[List[Any]] = None,
        timeout: float = 120.0,
        image_base64: Optional[str] = None,
        image_mime_type: Optional[str] = None,
    ) -> str:
        """Generate text. Tries Vertex AI first; on failure falls back to OpenAI.

        Returns the generated text string.
        Raises ``RuntimeError`` only if **both** providers fail.
        """
        # --- Try Vertex AI ---
        try:
            client = self._get_vertex_client()
            config = self._build_vertex_config(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            contents = self._build_vertex_contents(prompt, parts)
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = self._extract_vertex_text(response)
            if text:
                return text
            logger.warning("[LLMWrapper] Vertex AI returned empty text, falling back to OpenAI")
        except Exception as exc:
            logger.warning("[LLMWrapper] Vertex AI failed (%s: %s), falling back to OpenAI", type(exc).__name__, exc)

        # --- Fallback to OpenAI ---
        return await self._openai_generate_async(
            prompt, model=model, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p, timeout=timeout,
            client=self._get_async_http_client(),
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )

    async def _openai_generate_async(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float],
        timeout: float,
        client: httpx.AsyncClient,
        image_base64: Optional[str] = None,
        image_mime_type: Optional[str] = None,
    ) -> str:
        openai_key = self._get_openai_key()
        if not openai_key:
            raise RuntimeError("[LLMWrapper] Both Vertex AI and OpenAI are unavailable")

        logger.info(
            "[LLMWrapper] OpenAI fallback auth key fingerprint=%s (len=%d)",
            _mask_key_for_logs(openai_key),
            len(openai_key),
        )

        openai_model = self._resolve_openai_model(model)
        # If image data is provided, use vision-capable model and multimodal content
        if image_base64 and image_mime_type:
            openai_model = "gpt-4o"  # Force vision-capable model
            payload = self._build_openai_vision_payload(
                prompt, image_base64=image_base64, image_mime_type=image_mime_type,
                model=openai_model, temperature=temperature,
                max_tokens=max_tokens, top_p=top_p,
            )
        else:
            payload = self._build_openai_payload(
                prompt, model=openai_model, temperature=temperature,
                max_tokens=max_tokens, top_p=top_p,
            )
        resp = await client.post(
            _OPENAI_BASE,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        text = self._extract_openai_text(resp.json())
        if not text:
            raise RuntimeError("[LLMWrapper] OpenAI also returned empty text")
        logger.info("[LLMWrapper] OpenAI fallback succeeded (model=%s)", openai_model)
        return text

    # ------------------------------------------------------------------
    # Sync generate  (for non-async callers)
    # ------------------------------------------------------------------
    def generate_sync(
        self,
        prompt: str,
        *,
        model: str = _DEFAULT_VERTEX_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        parts: Optional[List[Any]] = None,
        timeout: float = 120.0,
    ) -> str:
        """Synchronous version — tries Vertex AI first, falls back to OpenAI."""
        # --- Try Vertex AI ---
        try:
            client = self._get_vertex_client()
            config = self._build_vertex_config(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            contents = self._build_vertex_contents(prompt, parts)
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = self._extract_vertex_text(response)
            if text:
                return text
            logger.warning("[LLMWrapper] Vertex AI returned empty, falling back to OpenAI")
        except Exception as exc:
            logger.warning("[LLMWrapper] Vertex AI failed (%s: %s), falling back to OpenAI", type(exc).__name__, exc)

        # --- Fallback to OpenAI ---
        return self._openai_generate_sync(
            prompt, model=model, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p, timeout=timeout,
        )

    def _openai_generate_sync(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float],
        timeout: float,
    ) -> str:
        openai_key = self._get_openai_key()
        if not openai_key:
            raise RuntimeError("[LLMWrapper] Both Vertex AI and OpenAI are unavailable")

        logger.info(
            "[LLMWrapper] OpenAI sync fallback auth key fingerprint=%s (len=%d)",
            _mask_key_for_logs(openai_key),
            len(openai_key),
        )

        openai_model = self._resolve_openai_model(model)
        payload = self._build_openai_payload(
            prompt, model=openai_model, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p,
        )
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                _OPENAI_BASE,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}",
                },
                json=payload,
            )
            resp.raise_for_status()
            text = self._extract_openai_text(resp.json())
            if not text:
                raise RuntimeError("[LLMWrapper] OpenAI also returned empty text")
            logger.info("[LLMWrapper] OpenAI sync fallback succeeded (model=%s)", openai_model)
            return text

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    async def close(self) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
llm_client = LLMWrapper()
