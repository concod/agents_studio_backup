"""
MCP (Model Context Protocol) client integration.

Provides:
- `test_mcp_connection(url, headers)`: open a one-shot session, list tools, return status.
- `MCPClientManager`: keeps streamable-HTTP sessions open for the lifetime of an
  agent execution and exposes the discovered tools as LangChain `StructuredTool`s.

Sessions are managed via `AsyncExitStack` so a single `close()` tears down every
underlying transport cleanly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


def _python_type_for_json_schema(prop_spec: Dict[str, Any]):
    t = (prop_spec or {}).get("type", "string")
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(t, str)


def _default_for_type(py_type):
    return {str: "", int: 0, float: 0.0, bool: False, list: [], dict: {}}.get(py_type, "")


def _build_args_schema(tool_name: str, input_schema: Optional[Dict[str, Any]]):
    """Convert a JSON-Schema-style input spec into a Pydantic model.

    Uses plain (non-Optional) types with safe defaults for Vertex AI / Gemini
    function-calling compatibility — matches the convention used elsewhere in
    `langraph_executor._build_tool_args_schema`.
    """

    class _EmptyInput(BaseModel):
        pass

    if not input_schema or not isinstance(input_schema, dict):
        return _EmptyInput

    props = input_schema.get("properties") or {}
    if not props:
        return _EmptyInput

    fields: Dict[str, Any] = {}
    for prop_name, prop_spec in props.items():
        if not isinstance(prop_spec, dict):
            prop_spec = {}
        py_type = _python_type_for_json_schema(prop_spec)
        default_val = _default_for_type(py_type)
        desc = prop_spec.get("description") or f"Parameter: {prop_name}"
        # Sanitize the field name (Pydantic disallows certain characters)
        safe_name = prop_name if prop_name.isidentifier() else prop_name.replace("-", "_")
        fields[safe_name] = (py_type, Field(default=default_val, description=desc))

    model_name = "".join(w.capitalize() for w in tool_name.replace("-", "_").split("_") if w) + "McpInput"
    if not model_name:
        model_name = "McpInput"
    try:
        return create_model(model_name, **fields)
    except Exception as e:
        logger.warning(f"[MCP] Failed to build pydantic args schema for {tool_name}: {e}")
        return _EmptyInput


def _content_to_text(content_list: Any) -> str:
    """Flatten MCP tool response content (list of TextContent/ImageContent/etc.) to a string."""
    if not content_list:
        return ""
    out_parts: List[str] = []
    for c in content_list:
        text = getattr(c, "text", None)
        if text is not None:
            out_parts.append(text)
            continue
        if hasattr(c, "data"):
            mime = getattr(c, "mimeType", "unknown")
            out_parts.append(f"[binary content type={mime}]")
            continue
        try:
            out_parts.append(json.dumps(c.model_dump()))
        except Exception:
            out_parts.append(str(c))
    return "\n".join(out_parts)


async def test_mcp_connection(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Open an MCP streamable-HTTP session, initialize it, list tools, and close.

    Returns {"success": bool, "tools": [...], "server_info": {...}, "error": str?}.
    """
    if not url:
        return {"success": False, "error": "Missing MCP server URL"}

    try:
        from mcp.client.streamable_http import streamablehttp_client
        from mcp.client.session import ClientSession
    except ImportError as e:
        return {"success": False, "error": f"MCP client library not installed: {e}"}

    async def _run() -> Dict[str, Any]:
        async with streamablehttp_client(url, headers=headers or {}, timeout=timeout) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                tools_result = await session.list_tools()
                tools: List[Dict[str, Any]] = []
                for t in getattr(tools_result, "tools", []) or []:
                    tools.append({
                        "name": getattr(t, "name", ""),
                        "description": getattr(t, "description", "") or "",
                    })
                server_info: Dict[str, Any] = {}
                si = getattr(init, "serverInfo", None)
                if si is not None:
                    server_info = {
                        "name": getattr(si, "name", "") or "",
                        "version": getattr(si, "version", "") or "",
                    }
                return {"success": True, "tools": tools, "server_info": server_info}

    try:
        return await asyncio.wait_for(_run(), timeout=timeout + 5)
    except asyncio.TimeoutError:
        return {"success": False, "error": f"Connection timed out after {timeout}s"}
    except BaseExceptionGroup as eg:  # anyio TaskGroup wraps the real cause here
        err_msg = _format_mcp_error(eg)
        logger.warning(f"[MCP] test_connection failed for {url}: {err_msg}")
        return {"success": False, "error": err_msg}
    except Exception as e:
        err_msg = _format_mcp_error(e)
        logger.warning(f"[MCP] test_connection failed for {url}: {err_msg}")
        return {"success": False, "error": err_msg}


def _format_mcp_error(exc: BaseException) -> str:
    """Flatten ExceptionGroup / TaskGroup errors from anyio into a readable message.

    The MCP streamable-HTTP client runs inside an anyio TaskGroup; when something
    fails (auth, TLS, DNS, 4xx/5xx, bad JSON-RPC, etc.) the outer exception is a
    BaseExceptionGroup whose default str() is the unhelpful
    "unhandled errors in a TaskGroup (1 sub-exception)". Walk into the group and
    extract the real underlying error(s).
    """
    leaves: List[str] = []

    def _walk(e: BaseException) -> None:
        sub_excs = getattr(e, "exceptions", None)
        if sub_excs:
            for sub in sub_excs:
                _walk(sub)
            return
        msg = str(e).strip()
        if not msg:
            msg = e.__class__.__name__
        else:
            msg = f"{e.__class__.__name__}: {msg}"
        if msg not in leaves:
            leaves.append(msg)

    _walk(exc)
    if not leaves:
        return f"{exc.__class__.__name__}: {exc}"
    if len(leaves) == 1:
        return leaves[0]
    return "; ".join(leaves)


class MCPClientManager:
    """Holds open MCP sessions for the lifetime of an executor.

    Usage:
        mgr = MCPClientManager(configs, logger_)
        await mgr.start()
        # ... mgr.tools is populated ...
        await mgr.close()
    """

    def __init__(
        self,
        server_configs: Optional[List[Dict[str, Any]]],
        logger_: Optional[logging.Logger] = None,
        connect_timeout: float = 30.0,
    ):
        self._logger = logger_ or logger
        self._configs: List[Dict[str, Any]] = [
            c for c in (server_configs or [])
            if isinstance(c, dict) and c.get("enabled", True) and c.get("url")
        ]
        self._connect_timeout = connect_timeout
        self._stack: Optional[AsyncExitStack] = None
        self._sessions: Dict[str, Any] = {}
        self.tools: List[Any] = []  # list[StructuredTool]
        self.connection_errors: List[Dict[str, str]] = []

    async def start(self) -> None:
        if self._stack is not None:
            return
        if not self._configs:
            return

        try:
            from mcp.client.streamable_http import streamablehttp_client
            from mcp.client.session import ClientSession
        except ImportError as e:
            self._logger.error(f"[MCP] mcp client library not available: {e}")
            return

        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            self._logger.error(f"[MCP] langchain_core not available: {e}")
            return

        stack = AsyncExitStack()
        await stack.__aenter__()
        self._stack = stack

        for idx, cfg in enumerate(self._configs):
            server_name = (cfg.get("name") or f"mcp_{idx}").strip() or f"mcp_{idx}"
            url = cfg["url"]
            headers = cfg.get("headers") or {}
            try:
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamablehttp_client(url, headers=headers, timeout=self._connect_timeout)
                )
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                self._sessions[server_name] = session

                tools_result = await session.list_tools()
                discovered = list(getattr(tools_result, "tools", []) or [])
                for mcp_tool in discovered:
                    wrapped = self._wrap_tool(server_name, session, mcp_tool, StructuredTool)
                    if wrapped is not None:
                        self.tools.append(wrapped)

                self._logger.info(
                    f"[MCP] Connected to '{server_name}' ({url}) — {len(discovered)} tools"
                )
            except Exception as e:
                err_msg = str(e)
                self._logger.error(f"[MCP] Failed to connect '{server_name}' ({url}): {err_msg}")
                self.connection_errors.append({"name": server_name, "url": url, "error": err_msg})

    def _wrap_tool(self, server_name: str, session: Any, mcp_tool: Any, StructuredTool):
        try:
            base_name = getattr(mcp_tool, "name", "") or "tool"
            safe_server = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in server_name.lower())
            safe_tool = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base_name.lower())
            tool_name = f"mcp_{safe_server}_{safe_tool}"
            description = (getattr(mcp_tool, "description", None) or
                           f"MCP tool '{base_name}' from server '{server_name}'.")

            input_schema = getattr(mcp_tool, "inputSchema", None)
            args_schema = _build_args_schema(tool_name, input_schema)

            async def _tool_func(**kwargs):
                try:
                    arguments = {k: v for k, v in kwargs.items()
                                 if v is not None and v != "" and v != [] and v != {}}
                    result = await session.call_tool(base_name, arguments)
                    payload = _content_to_text(getattr(result, "content", None))
                    if getattr(result, "isError", False):
                        return json.dumps({"error": payload or "MCP tool returned isError"})
                    return payload or ""
                except Exception as e:
                    return json.dumps({
                        "error": f"MCP tool '{base_name}' on '{server_name}' failed: {e}"
                    })

            return StructuredTool.from_function(
                func=_tool_func,
                coroutine=_tool_func,
                name=tool_name,
                description=description,
                args_schema=args_schema,
            )
        except Exception as e:
            self._logger.warning(f"[MCP] Failed to wrap tool '{getattr(mcp_tool, 'name', '?')}': {e}")
            return None

    async def close(self) -> None:
        stack = self._stack
        self._stack = None
        self._sessions = {}
        if stack is None:
            return
        try:
            await stack.__aexit__(None, None, None)
        except Exception as e:
            # The MCP transport sometimes raises during teardown when the server
            # has already closed the connection. Swallow so cleanup never fails.
            self._logger.warning(f"[MCP] Error while closing sessions: {e}")
