"""Tests for contextflow.lens.middleware — LensMiddleware."""

from __future__ import annotations

import pytest

from contextflow.lens.middleware import LensMiddleware, LensEvent


class TestLensMiddleware:
    @pytest.mark.asyncio
    async def test_observe_records_event(self):
        lens = LensMiddleware()
        messages = [{"role": "user", "content": "Hi"}]

        async def fake_call(msgs):
            return "Hello!"

        response = await lens.observe(messages, fake_call)
        assert response == "Hello!"
        assert len(lens.history) == 1
        assert isinstance(lens.history[0], LensEvent)
        assert lens.history[0].request == messages
        assert lens.history[0].response == "Hello!"

    @pytest.mark.asyncio
    async def test_observe_multiple_calls(self):
        lens = LensMiddleware()

        async def echo(msgs):
            return msgs[0]["content"]

        await lens.observe([{"role": "user", "content": "A"}], echo)
        await lens.observe([{"role": "user", "content": "B"}], echo)

        assert len(lens.history) == 2
        assert lens.history[0].response == "A"
        assert lens.history[1].response == "B"

    @pytest.mark.asyncio
    async def test_observe_passes_messages_to_call(self):
        lens = LensMiddleware()
        received = []

        async def spy_call(msgs):
            received.extend(msgs)
            return "ok"

        messages = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hi"}]
        await lens.observe(messages, spy_call)
        assert len(received) == 2


class TestLensEvent:
    def test_fields(self):
        evt = LensEvent(
            request=[{"role": "user", "content": "Hi"}],
            response="Hello!",
        )
        assert evt.request[0]["role"] == "user"
        assert evt.response == "Hello!"
