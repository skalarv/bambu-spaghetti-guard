"""Cover the success paths of ntfy / telegram / home-assistant transports."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import spaghetti_guard.notifier as nm
from spaghetti_guard.notifier import (
    HomeAssistantNotifier,
    NtfyNotifier,
    TelegramNotifier,
    build_notifier,
)


@pytest.fixture
def http(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        return MagicMock(status_code=200)

    def fake_put(url, **kwargs):
        calls.append(("put", url, kwargs))
        return MagicMock(status_code=200)

    monkeypatch.setattr(nm.httpx, "post", fake_post)
    monkeypatch.setattr(nm.httpx, "put", fake_put)
    return calls


# ---- ntfy ----------------------------------------------------------------


def test_ntfy_text_only(http):
    n = NtfyNotifier("https://ntfy.example/topic")
    assert n.send("hi", "body") is True
    assert len(http) == 1
    method, url, kw = http[0]
    assert method == "post" and url == "https://ntfy.example/topic"
    assert kw["headers"]["Title"] == "hi"


def test_ntfy_with_image(http, tmp_path):
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"jpegbytes")
    n = NtfyNotifier("https://ntfy.example/topic")
    assert n.send("hi", "body", image_path=img) is True
    assert len(http) == 2  # text + image attachment
    assert http[1][0] == "put"
    assert http[1][2]["headers"]["Filename"] == "snap.jpg"


def test_ntfy_image_path_missing_skipped(http, tmp_path):
    n = NtfyNotifier("https://ntfy.example/topic")
    n.send("hi", "body", image_path=tmp_path / "nope.jpg")
    assert len(http) == 1  # only text


# ---- telegram -----------------------------------------------------------


def test_telegram_text_only(http):
    t = TelegramNotifier("BOTTOKEN", "12345")
    assert t.send("hi", "body") is True
    assert len(http) == 1
    method, url, kw = http[0]
    assert method == "post" and "api.telegram.org" in url
    assert kw["json"]["chat_id"] == "12345"


def test_telegram_with_image(http, tmp_path):
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"jpegbytes")
    t = TelegramNotifier("BOTTOKEN", "12345")
    assert t.send("hi", "body", image_path=img) is True
    assert len(http) == 2
    photo_call = http[1]
    assert photo_call[0] == "post"
    assert "sendPhoto" in photo_call[1]


def test_telegram_failure_suppressed(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("net")

    monkeypatch.setattr(nm.httpx, "post", boom)
    t = TelegramNotifier("token", "chat")
    assert t.send("h", "b") is False


# ---- telegram target parsing --------------------------------------------
# Real bot tokens contain a colon ("123456:AAH..."), so the builder must
# split on the LAST colon of "<bot_token>:<chat_id>".


def test_telegram_target_token_contains_colon():
    n = build_notifier("telegram", "123456:AAHsome-token:987654321")
    assert isinstance(n, TelegramNotifier)
    assert n._token == "123456:AAHsome-token"
    assert n._chat_id == "987654321"


# ---- HTTP errors are failures, not silent success ------------------------


@pytest.fixture
def http_500(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        return MagicMock(status_code=500)

    def fake_put(url, **kwargs):
        calls.append(("put", url, kwargs))
        return MagicMock(status_code=500)

    monkeypatch.setattr(nm.httpx, "post", fake_post)
    monkeypatch.setattr(nm.httpx, "put", fake_put)
    return calls


def test_ntfy_http_error_returns_false(http_500):
    n = NtfyNotifier("https://ntfy.example/topic")
    assert n.send("hi", "body") is False


def test_telegram_http_error_returns_false(http_500):
    t = TelegramNotifier("BOTTOKEN", "12345")
    assert t.send("hi", "body") is False


def test_homeassistant_http_error_returns_false(http_500):
    ha = HomeAssistantNotifier("https://ha.example/api/webhook/abc")
    assert ha.send("hi", "body") is False


def test_ntfy_image_upload_error_does_not_fail_text_delivery(monkeypatch, tmp_path):
    """The text alert reached the operator; a failed image attachment must not
    report the whole notification as lost."""
    monkeypatch.setattr(nm.httpx, "post", lambda url, **kw: MagicMock(status_code=200))
    monkeypatch.setattr(nm.httpx, "put", lambda url, **kw: MagicMock(status_code=500))
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"jpegbytes")
    n = NtfyNotifier("https://ntfy.example/topic")
    assert n.send("hi", "body", image_path=img) is True


# ---- home assistant ----------------------------------------------------


def test_homeassistant_text(http):
    ha = HomeAssistantNotifier("https://ha.example/api/webhook/abc")
    assert ha.send("hi", "body") is True
    assert len(http) == 1
    assert http[0][2]["json"] == {"title": "hi", "message": "body"}


def test_homeassistant_with_image_path(http, tmp_path):
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"x")
    ha = HomeAssistantNotifier("https://ha.example/api/webhook/abc")
    ha.send("hi", "body", image_path=img)
    assert "image" in http[0][2]["json"]


def test_homeassistant_failure_suppressed(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("net")

    monkeypatch.setattr(nm.httpx, "post", boom)
    ha = HomeAssistantNotifier("https://ha.example/api/webhook/abc")
    assert ha.send("h", "b") is False
