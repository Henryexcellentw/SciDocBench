#!/usr/bin/env python3
"""Self-contained OpenAI-compatible GPT client helpers for process_supervision_v1."""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_CHAT_COMPLETIONS_URL = os.environ.get(
    "OPENAI_COMPAT_CHAT_COMPLETIONS_URL",
    "",
)
DEFAULT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "gpt-5.4")

CHAT_COMPLETIONS_URL = DEFAULT_CHAT_COMPLETIONS_URL
OPENAI_SSL_CONTEXT = ssl.create_default_context()
OPENAI_SSL_MODE = "system_default"


def load_api_key(cli_value: str | None, search_root: Path):
    if cli_value:
        return cli_value
    for env_name in ["OPENAI_COMPAT_API_KEY", "OPENROUTER_API_KEY"]:
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
    candidates = [
        search_root / "openrouter.py",
        search_root / "process_supervision_v1" / "openrouter.py",
    ]
    for example_path in candidates:
        if not example_path.exists():
            continue
        content = example_path.read_text(encoding="utf-8")
        for variable_name in ["OPENAI_COMPAT_API_KEY", "OPENROUTER_API_KEY"]:
            match = re.search(rf'{variable_name}\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    return None


def configure_chat_completions_url(api_base_url: str | None = None):
    global CHAT_COMPLETIONS_URL
    CHAT_COMPLETIONS_URL = (api_base_url or DEFAULT_CHAT_COMPLETIONS_URL).strip()
    return CHAT_COMPLETIONS_URL


def discover_ca_bundle():
    candidates = []
    env_path = os.environ.get("SSL_CERT_FILE")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/etc/ssl/certs/ca-certificates.crt"),
            Path("/etc/pki/tls/certs/ca-bundle.crt"),
            Path("/etc/ssl/cert.pem"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    try:
        import certifi  # type: ignore

        certifi_path = Path(certifi.where())
        if certifi_path.exists():
            return certifi_path
    except Exception:
        pass
    return None


def configure_ssl(ca_bundle: Path | None = None, insecure_ssl: bool = False):
    global OPENAI_SSL_CONTEXT, OPENAI_SSL_MODE
    if insecure_ssl:
        OPENAI_SSL_CONTEXT = ssl._create_unverified_context()
        OPENAI_SSL_MODE = "insecure"
        return OPENAI_SSL_MODE

    bundle = ca_bundle or discover_ca_bundle()
    if bundle is not None:
        OPENAI_SSL_CONTEXT = ssl.create_default_context(cafile=str(bundle))
        OPENAI_SSL_MODE = f"ca_bundle:{bundle}"
        return OPENAI_SSL_MODE

    OPENAI_SSL_CONTEXT = ssl.create_default_context()
    OPENAI_SSL_MODE = "system_default"
    return OPENAI_SSL_MODE


def encode_image_data_url(path: Path):
    suffix = path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class ModelContentJSONError(RuntimeError):
    """Raised when the endpoint is reachable but the model content is not JSON."""

    def __init__(self, message: str, content: str):
        super().__init__(message)
        self.content = content


def maybe_extract_json(text: str):
    if not text:
        raise RuntimeError("Expected JSON, got empty text.")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            snippet = text[:1000].replace("\n", "\\n")
            raise ModelContentJSONError(f"Model content is not valid JSON. snippet={snippet}", text) from exc
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as inner_exc:
            snippet = match.group(0)[:1000].replace("\n", "\\n")
            raise ModelContentJSONError(f"Extracted JSON object is invalid. snippet={snippet}", text) from inner_exc


def call_json(api_key: str, model: str, messages, max_tokens: int, temperature: float = 0.2, retries: int = 3):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                CHAT_COMPLETIONS_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            scheme = urllib.parse.urlparse(CHAT_COMPLETIONS_URL).scheme.lower()
            if scheme == "https":
                response_ctx = urllib.request.urlopen(request, timeout=180, context=OPENAI_SSL_CONTEXT)
            else:
                response_ctx = urllib.request.urlopen(request, timeout=180)
            with response_ctx as response:
                raw_response_text = response.read().decode("utf-8", errors="replace")
            try:
                response_payload = json.loads(raw_response_text)
            except json.JSONDecodeError as exc:
                snippet = raw_response_text[:1000].replace("\n", "\\n")
                raise RuntimeError(f"Chat-completions HTTP response is not valid JSON. snippet={snippet}") from exc
            content = ((response_payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not content:
                raise RuntimeError("Chat-completions endpoint returned empty content.")
            return maybe_extract_json(content)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8, 2 ** attempt))
    if last_error is not None and "CERTIFICATE_VERIFY_FAILED" in str(last_error):
        raise RuntimeError(
            "API SSL verification failed. Try --ca-bundle /etc/ssl/certs/ca-certificates.crt "
            "or --insecure-ssl if you trust the network. "
            f"Current ssl_mode={OPENAI_SSL_MODE}. Original error: {last_error}"
        ) from last_error
    raise last_error
