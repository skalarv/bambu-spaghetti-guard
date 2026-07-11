"""Pluggable alerters (brief §5.5).

A notifier failure must never crash the guard, so every transport wraps the
HTTP call in a try/except and logs.
"""

from __future__ import annotations

import logging
import ssl
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# httpx defaults to the certifi CA bundle, which omits enterprise / TLS-inspection
# proxy roots. On such networks (e.g. corporate Windows) that makes every alert
# fail with CERTIFICATE_VERIFY_FAILED. ssl.create_default_context() loads the OS
# trust store instead — the Windows ROOT/CA stores on Windows, the system bundle
# on Linux — which includes those roots. Built once at import.
_SSL_CTX = ssl.create_default_context()


class Notifier(ABC):
    @abstractmethod
    def send(self, title: str, message: str, image_path: Path | None = None) -> bool:
        """Return True on success, False on suppressed failure."""


def _delivered(resp, transport: str) -> bool:
    """A 4xx/5xx means the alert did NOT reach the operator — that must count
    as a failure, not silent success."""
    ok = 200 <= resp.status_code < 300
    if not ok:
        logger.error("%s notify rejected with HTTP %s", transport, resp.status_code)
    return ok


class NoopNotifier(Notifier):
    def send(self, title: str, message: str, image_path: Path | None = None) -> bool:
        logger.info("noop notify: %s | %s", title, message)
        return True


class NtfyNotifier(Notifier):
    def __init__(self, topic_url: str, *, timeout_s: float = 5.0) -> None:
        self._url = topic_url
        self._timeout = timeout_s

    def send(self, title: str, message: str, image_path: Path | None = None) -> bool:
        try:
            headers = {"Title": title}
            resp = httpx.post(self._url, content=message.encode("utf-8"), headers=headers, timeout=self._timeout, verify=_SSL_CTX)
            if not _delivered(resp, "ntfy"):
                return False
            if image_path is not None and image_path.exists():
                # ntfy supports separate attachment publish. The text already
                # reached the operator — a failed attachment is log-only.
                img_resp = httpx.put(
                    self._url,
                    content=image_path.read_bytes(),
                    headers={"Filename": image_path.name, "Title": f"{title} (image)"},
                    timeout=self._timeout,
                    verify=_SSL_CTX,
                )
                _delivered(img_resp, "ntfy attachment")
            return True
        except Exception:
            logger.exception("ntfy notify failed (suppressed)")
            return False


class TelegramNotifier(Notifier):
    def __init__(self, bot_token: str, chat_id: str, *, timeout_s: float = 5.0) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout_s

    def send(self, title: str, message: str, image_path: Path | None = None) -> bool:
        try:
            text = f"*{title}*\n{message}"
            resp = httpx.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=self._timeout,
                verify=_SSL_CTX,
            )
            if not _delivered(resp, "telegram"):
                return False
            if image_path is not None and image_path.exists():
                with image_path.open("rb") as f:
                    img_resp = httpx.post(
                        f"https://api.telegram.org/bot{self._token}/sendPhoto",
                        data={"chat_id": self._chat_id, "caption": title},
                        files={"photo": (image_path.name, f.read(), "image/jpeg")},
                        timeout=self._timeout,
                        verify=_SSL_CTX,
                    )
                # Text already delivered — a failed photo is log-only.
                _delivered(img_resp, "telegram photo")
            return True
        except Exception:
            logger.exception("telegram notify failed (suppressed)")
            return False


class HomeAssistantNotifier(Notifier):
    def __init__(self, webhook_url: str, *, timeout_s: float = 5.0) -> None:
        self._url = webhook_url
        self._timeout = timeout_s

    def send(self, title: str, message: str, image_path: Path | None = None) -> bool:
        try:
            payload = {"title": title, "message": message}
            if image_path is not None:
                payload["image"] = str(image_path)
            resp = httpx.post(self._url, json=payload, timeout=self._timeout, verify=_SSL_CTX)
            return _delivered(resp, "home-assistant")
        except Exception:
            logger.exception("home-assistant notify failed (suppressed)")
            return False


def build_notifier(backend: str, target: str) -> Notifier:
    """Construct the right notifier from a config tuple. `target` is unused for noop."""
    backend = backend.lower()
    if backend == "none":
        return NoopNotifier()
    if backend == "ntfy":
        return NtfyNotifier(target)
    if backend == "telegram":
        # target encodes "<bot_token>:<chat_id>". Bot tokens themselves
        # contain a colon ("123456:AAH..."), so split on the LAST one.
        if ":" not in target:
            raise ValueError("telegram target must be '<bot_token>:<chat_id>'")
        token, _, chat_id = target.rpartition(":")
        return TelegramNotifier(token, chat_id)
    if backend == "homeassistant":
        return HomeAssistantNotifier(target)
    raise ValueError(f"unknown notify backend: {backend}")
