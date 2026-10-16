import os
import json
from google.cloud import secretmanager


def load_credentials():
    """Load credentials from GCP Secret Manager JSON secret"""
    client = None
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/482146741158/secrets/tools_universe_json_key_gcs/versions/latest"
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        credentials = json.loads(secret_value)
        return credentials.get("redis", {}), credentials.get("database", {})

    except Exception as e:
        return {}, {}
    finally:
        if client is not None:
            client.close()
