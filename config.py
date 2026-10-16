# backend/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Bootstrap env vars (read before Settings, used for SM bootstrap) ───
_DEPLOY_ENV = os.getenv("DEPLOY_ENV", "local").strip().lower()
_GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "tools-universe-29042025")

# GCP Secret Manager — guarded for envs without the SDK
_gcloud_available = False
try:
    from google.cloud import secretmanager as _secretmanager
    _gcloud_available = True
except Exception:
    _secretmanager = None  # type: ignore[assignment]


def _fetch_secret_value(secret_id: str, project: str = _GCP_PROJECT_ID) -> Optional[str]:
    """Fetch the latest version of a GCP Secret Manager secret by its ID."""
    if not _gcloud_available or _secretmanager is None:
        logger.warning("google-cloud-secret-manager not available, cannot fetch '%s'", secret_id)
        return None

    path = f"projects/{project}/secrets/{secret_id}/versions/latest"
    try:
        client = _secretmanager.SecretManagerServiceClient()
        resp = client.access_secret_version(request={"name": path})
        value = resp.payload.data.decode("utf-8").strip()
        # Strip surrounding quotes if present
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        logger.info("Fetched secret '%s' (length=%d)", secret_id, len(value))
        return value
    except Exception as exc:
        logger.error("Failed to fetch secret '%s': %s: %s", secret_id, type(exc).__name__, exc)
        return None


def _normalize_openai_secret_value(value: str) -> str:
    """Return a usable OpenAI API key from either raw or JSON secret payload."""
    raw = (value or "").strip()
    if not raw:
        return ""

    # Most common format: {"OPENAI_API_KEY": "sk-..."}
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for field in ("OPENAI_API_KEY", "openai_api_key", "api_key", "key"):
                    candidate = str(parsed.get(field, "")).strip()
                    if candidate:
                        return candidate
        except Exception:
            # Keep raw value fallback to preserve backward compatibility.
            pass

    return raw


def _normalize_vertex_secret_value(value: str) -> str:
    """Return a usable Vertex AI API key from either raw or JSON secret payload."""
    raw = (value or "").strip()
    if not raw:
        return ""

    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for field in ("VERTEX_API_KEY", "vertex_api_key", "api_key", "key"):
                    candidate = str(parsed.get(field, "")).strip()
                    if candidate:
                        return candidate
        except Exception:
            pass

    return raw


class Settings(BaseSettings):
    # ---- DB ----
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: str = Field(default="postgres")
    DB_NAME: str = Field(default="appdb")
    DATABASE_URL: str = Field(default="")
    PG_DATABASE_URL: str = Field(default="")

    # -------------------- Interaction Logs Proxy --------------------
    INTERACTION_LOGS_API_URL: str = ""
    INTERACTION_LOGS_SECRET_KEY: str = ""
    INTERACTION_LOGS_EMAIL: str = ""
    INTERACTION_LOGS_ALGORITHM: str = "HS256"

    # ---- JWT Authentication ----
    JWT_SECRET_KEY: str = Field(default="your-secret-key-here")  # Change this in production
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    JWT_SIGNED_URL_SECRET_KEY: str = Field(default="SINARGIOPFSSGAJUNTDFGE2026MARK42")  # Separate secret for signed URL tokens

    MICROSOFT_CLIENT_ID: str = ""
    MICROSOFT_CLIENT_SECRET: str = ""
    MICROSOFT_TENANT_ID: str = "common"  # "common" for multi-tenant, or specific tenant ID
    MICROSOFT_REDIRECT_URI: str = "http://localhost:8000/api/ms/callback"
    FRONTEND_URL: str = "http://localhost:8000"

    # ---- GCS Signed URLs ----
    GCS_SECRET_JSON: str = Field(default="")
    GCP_PROJECT_ID: str = Field(default="")

    # ---- API Keys ----
    OPENAI_API_KEY: str = Field(default="")
    VERTEX_API_KEY: str = Field(default="")

    # ---- Misc ----
    ENABLE_DATADOG: bool = Field(default=False)

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # set True if you prefer strict case
        extra="ignore",
    )


def _make_settings() -> Settings:
    """Build Settings, enriching API keys from Secret Manager."""
    s = Settings()

    print(f"[config] DEPLOY_ENV = '{_DEPLOY_ENV}'")
    print(f"[config] GCP_PROJECT_ID = '{_GCP_PROJECT_ID}'")
    print(f"[config] Secret Manager SDK available: {_gcloud_available}")

    # Fetch OPENAI_API_KEY from Secret Manager (all environments)
    openai_secret = os.getenv("OPENAI_SECRET_NAME", "").strip()
    if openai_secret:
        print(f"[config] Fetching OPENAI_API_KEY from secret '{openai_secret}'")
        val = _fetch_secret_value(openai_secret)
        if val:
            resolved_openai_key = _normalize_openai_secret_value(val)
            if resolved_openai_key != val:
                print("[config] OPENAI secret payload normalized from JSON to raw key")

            s.OPENAI_API_KEY = resolved_openai_key
            print(
                f"[config] ✅ OPENAI_API_KEY loaded from Secret Manager "
                f"(length={len(resolved_openai_key)}, key={resolved_openai_key[:4]}...{resolved_openai_key[-4:]})"
            )
        else:
            print(f"[config] ⚠️ Could not fetch OpenAI key from secret '{openai_secret}', using .env value")
    else:
        print(f"[config] OPENAI_SECRET_NAME not set; using OPENAI_API_KEY from .env (set={bool(s.OPENAI_API_KEY)})")

    if _DEPLOY_ENV == "local":
        # Fetch VERTEX_API_KEY from Secret Manager if secret name is configured
        vertex_secret = os.getenv("LOCAL_VERTEX_AI_SECRET_NAME", "").strip()
        if vertex_secret:
            print(f"[config] Local dev: fetching VERTEX_API_KEY from secret '{vertex_secret}'")
            val = _fetch_secret_value(vertex_secret)
            if val:
                s.VERTEX_API_KEY = _normalize_vertex_secret_value(val)
                print(f"[config] ✅ VERTEX_API_KEY loaded from Secret Manager (length={len(s.VERTEX_API_KEY)})")
            else:
                print(f"[config] ⚠️ Could not fetch Vertex AI key from secret '{vertex_secret}', using .env value")
        else:
            print(f"[config] LOCAL_VERTEX_AI_SECRET_NAME not set; using VERTEX_API_KEY from .env (set={bool(s.VERTEX_API_KEY)})")
            if s.VERTEX_API_KEY:
                s.VERTEX_API_KEY = _normalize_vertex_secret_value(s.VERTEX_API_KEY)

    return s


settings = _make_settings()
