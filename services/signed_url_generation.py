import json
import re
import logging
from typing import Any, Optional
from datetime import timedelta
from urllib.parse import urlparse, unquote

from google.cloud import storage
from config import settings

logger = logging.getLogger(__name__)

# Regex to match GCS URLs
GCS_URL_PATTERN = re.compile(
    r'https?://storage\.(?:googleapis|cloud\.google)\.com/[^\s<>"\'()]+(?:\([^\s<>"\'()]*\))?[^\s<>"\'().,;!?\]]'
)


def get_service_account_from_secret(secret_name: str, project_id: str) -> dict:
    """
    Get service account JSON. Handles multiple cases:
    1. If secret_name is already a JSON string, parse it directly
    2. If it's a secret name, fetch from GCP Secret Manager
    """
    # Check if secret_name is already a JSON string (starts with {)
    if secret_name and secret_name.strip().startswith('{'):
        try:
            logger.info("[get_service_account_from_secret] Parsing GCS_SECRET_JSON as direct JSON string")
            return json.loads(secret_name)
        except json.JSONDecodeError as e:
            logger.error(f"[get_service_account_from_secret] Failed to parse JSON string: {e}")
            raise

    # Otherwise, try to fetch from Secret Manager
    try:
        logger.info(f"[get_service_account_from_secret] Fetching from Secret Manager: {secret_name}")
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(name=secret_path)
        secret_payload = response.payload.data.decode("UTF-8")
        return json.loads(secret_payload)
    except Exception as e:
        logger.warning(f"[get_service_account_from_secret] Failed to fetch from Secret Manager: {e}")
        raise


def _get_storage_client() -> Optional[storage.Client]:
    """Create a GCS storage client using configured credentials."""
    import os

    secret_name = getattr(settings, 'GCS_SECRET_JSON', '') or os.getenv("GCS_SECRET_JSON", "")
    project_id = getattr(settings, 'GCP_PROJECT_ID', '') or os.getenv("GCP_PROJECT_ID", "")

    if not secret_name or not project_id:
        logger.warning("[_get_storage_client] Missing GCS_SECRET_JSON or GCP_PROJECT_ID configuration")
        return None

    try:
        service_account_info = get_service_account_from_secret(
            secret_name=secret_name,
            project_id=project_id
        )
        return storage.Client.from_service_account_info(service_account_info)
    except Exception as e:
        logger.warning(f"[_get_storage_client] Could not create client from config: {e}")
        try:
            return storage.Client()
        except Exception as default_err:
            logger.error(f"[_get_storage_client] Failed with default credentials too: {default_err}")
            return None


def generate_signed_url(bucket_name: str, doc_list: list, expiration_minutes: int = 10) -> dict:
    """
    Generate signed URLs for a list of GCS objects.
    """
    client = _get_storage_client()
    if not client:
        return {}

    bucket = client.bucket(bucket_name)
    signed_url_list = {}
    for doc in doc_list:
        try:
            blob = bucket.blob(doc)
            url = blob.generate_signed_url(
                expiration=timedelta(minutes=expiration_minutes),
                version="v4",
                method="GET"
            )
            signed_url_list[doc] = url
        except Exception as e:
            logger.error(f"[generate_signed_url] Failed for {doc}: {e}")
    return signed_url_list


def convert_gcs_url_to_signed_url(gcs_url: str, expiration_minutes: int = 10) -> Optional[str]:
    """
    Convert a GCS URL (e.g., https://storage.googleapis.com/bucket-name/path/to/file.pdf
    or https://storage.cloud.google.com/bucket-name/path/to/file.pdf)
    to a signed URL.

    Returns signed URL string or None if conversion fails.
    """
    if not gcs_url:
        return None

    try:
        parsed = urlparse(gcs_url)

        if parsed.netloc not in ("storage.googleapis.com", "storage.cloud.google.com"):
            return None

        decoded_path = unquote(parsed.path).rstrip(')').rstrip()
        path_parts = decoded_path.lstrip('/').split('/', 1)

        if len(path_parts) < 1:
            return None

        bucket_name = path_parts[0]
        object_path = path_parts[1] if len(path_parts) > 1 else ""

        client = _get_storage_client()
        if not client:
            return None

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)

        signed_url = blob.generate_signed_url(
            expiration=timedelta(minutes=expiration_minutes),
            version="v4",
            method="GET"
        )
        logger.info(f"[convert_gcs_url_to_signed_url] Generated signed URL for bucket={bucket_name}")
        return signed_url

    except Exception as e:
        logger.error(f"[convert_gcs_url_to_signed_url] Error: {e}")
        return None


def _collect_gcs_urls_from_text(text: str) -> set:
    """Extract all GCS URLs from a text string."""
    if not text:
        return set()
    return set(GCS_URL_PATTERN.findall(text))


def _collect_gcs_urls_from_data(data: Any) -> set:
    """Recursively collect all unique GCS URLs from a data structure."""
    urls = set()
    if isinstance(data, str):
        urls.update(_collect_gcs_urls_from_text(data))
    elif isinstance(data, dict):
        for value in data.values():
            urls.update(_collect_gcs_urls_from_data(value))
    elif isinstance(data, list):
        for item in data:
            urls.update(_collect_gcs_urls_from_data(item))
    return urls


def _build_signed_url_map(gcs_urls: set, expiration_minutes: int = 10) -> dict:
    """Generate signed URLs for a set of unique GCS URLs. Returns {original: signed}."""
    url_map = {}
    for gcs_url in gcs_urls:
        signed = convert_gcs_url_to_signed_url(gcs_url, expiration_minutes)
        if signed:
            url_map[gcs_url] = signed
    return url_map


def _apply_url_map_to_text(text: str, url_map: dict) -> str:
    """Replace all GCS URLs in text using a pre-built mapping."""
    for original, signed in url_map.items():
        text = text.replace(original, signed)
    return text


def _apply_url_map_to_data(data: Any, url_map: dict) -> Any:
    """Recursively replace GCS URLs in a data structure using a pre-built mapping."""
    if isinstance(data, str):
        return _apply_url_map_to_text(data, url_map)
    elif isinstance(data, dict):
        return {key: _apply_url_map_to_data(value, url_map) for key, value in data.items()}
    elif isinstance(data, list):
        return [_apply_url_map_to_data(item, url_map) for item in data]
    return data


def replace_gcs_urls_in_text(text: str, expiration_minutes: int = 1440) -> str:
    """
    Find all unique GCS URLs in a text string, generate one signed URL per
    unique GCS URL, and replace all occurrences.
    """
    if not text:
        return text

    unique_urls = _collect_gcs_urls_from_text(text)
    if not unique_urls:
        return text

    url_map = _build_signed_url_map(unique_urls, expiration_minutes)
    return _apply_url_map_to_text(text, url_map)


def replace_gcs_urls_in_data(data: Any, expiration_minutes: int = 10) -> Any:
    """
    Recursively walk a data structure (str, dict, list), collect all unique
    GCS URLs across the entire structure, generate one signed URL per unique
    GCS URL, then replace all occurrences.
    """
    unique_urls = _collect_gcs_urls_from_data(data)
    if not unique_urls:
        return data

    url_map = _build_signed_url_map(unique_urls, expiration_minutes)
    return _apply_url_map_to_data(data, url_map)
