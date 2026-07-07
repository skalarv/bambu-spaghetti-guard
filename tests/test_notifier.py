"""Brief §5.5 — light smoke for the notifier ABC + factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from spaghetti_guard.notifier import (
    HomeAssistantNotifier,
    NoopNotifier,
    NtfyNotifier,
    TelegramNotifier,
    build_notifier,
)


def test_noop_notifier_smoke():
    assert NoopNotifier().send("title", "msg") is True
    assert NoopNotifier().send("title", "msg", image_path=None) is True


def test_build_notifier_none():
    assert isinstance(build_notifier("none", ""), NoopNotifier)


def test_build_notifier_ntfy():
    n = build_notifier("ntfy", "https://ntfy.example/topic")
    assert isinstance(n, NtfyNotifier)


def test_build_notifier_homeassistant():
    n = build_notifier("homeassistant", "https://ha.example/api/webhook/abc")
    assert isinstance(n, HomeAssistantNotifier)


def test_build_notifier_telegram_format():
    n = build_notifier("telegram", "token123:chat456")
    assert isinstance(n, TelegramNotifier)


def test_build_telegram_requires_colon():
    with pytest.raises(ValueError):
        build_notifier("telegram", "no-colon-here")


def test_build_notifier_rejects_unknown():
    with pytest.raises(ValueError):
        build_notifier("nonexistent", "")


def test_ntfy_failure_suppressed(monkeypatch):
    """A network exception in the http call must NOT raise; returns False."""
    import spaghetti_guard.notifier as nm

    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(nm.httpx, "post", boom)
    n = NtfyNotifier("https://ntfy.example/topic")
    assert n.send("t", "m") is False  # exception swallowed
