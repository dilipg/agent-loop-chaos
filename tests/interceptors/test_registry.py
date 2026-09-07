"""The attach registry: patch what is installed, restore it, and say what happened.

The registry exists because of a specific failure. Pointing the library at a real
service, the `data.*` scenarios armed and never fired: the graph reached its model
through a constructor argument and its data through a repository call, so there was no
`llm` layer and no `tool` layer for a payload fault to attach to. The run was green and
proved nothing.

So silence is the thing under test here. A registry that attaches nothing must say so,
name every strategy it tried, and give the reason -- and a scenario that needs a layer
nobody attached must fail loudly rather than pass vacuously.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.interceptors import Attachment, Registry


class _Host:
    """Something to patch, standing in for `httpx.Client` or `BaseChatModel`."""

    def send(self, payload: str) -> str:
        return f"real:{payload}"

    def other(self, payload: str) -> str:
        return f"other:{payload}"


class _FakeStrategy:
    """Patches `_Host.send`, and records how many calls it saw."""

    name = "fake"

    def __init__(self, *, unavailable: str | None = None, layer: str = "llm") -> None:
        self._unavailable = unavailable
        self._layer = layer
        self._original: Any = None

    def available(self) -> str | None:
        return self._unavailable

    def attach(self, engine: Any, seen: dict[str, int]) -> list[Attachment]:
        self._original = _Host.send

        def patched(host: _Host, payload: str) -> str:
            seen["_Host.send"] = seen.get("_Host.send", 0) + 1
            return self._original(host, payload)

        _Host.send = patched  # type: ignore[method-assign]
        return [Attachment(layer=self._layer, strategy=self.name, target="_Host.send")]

    def detach(self) -> None:
        if self._original is not None:
            _Host.send = self._original  # type: ignore[method-assign]
            self._original = None


class TestItPatchesAndRestores:
    def test_it_patches_on_attach(self) -> None:
        registry = Registry([_FakeStrategy()])
        original = _Host.send
        try:
            registry.attach(engine=None)
            assert _Host.send is not original
            assert _Host().send("x") == "real:x", "the patch must not change behaviour"
        finally:
            registry.detach()
        assert _Host.send is original

    def test_one_broken_strategy_does_not_block_the_others(self) -> None:
        """A strategy that raises on detach leaves its own patch stuck -- nobody else
        can restore an attribute they do not own. What must hold is that it does not
        take the other strategies down with it."""

        class _Broken(_FakeStrategy):
            name = "broken"

            def attach(self, engine: Any, seen: dict[str, int]) -> list[Attachment]:
                _Host.other = lambda host, payload: "patched"  # type: ignore[assignment]
                return [Attachment(layer="tool", strategy=self.name, target="_Host.other")]

            def detach(self) -> None:
                raise RuntimeError("boom")

        registry = Registry([_Broken(), _FakeStrategy()])
        original = _Host.send
        registry.attach(engine=None)
        registry.detach()  # must not raise
        assert _Host.send is original, "the working strategy was not restored"

    def test_detach_is_idempotent(self) -> None:
        registry = Registry([_FakeStrategy()])
        original = _Host.send
        registry.attach(engine=None)
        registry.detach()
        registry.detach()
        assert _Host.send is original

    def test_it_counts_what_each_hook_saw(self) -> None:
        registry = Registry([_FakeStrategy()])
        try:
            report = registry.attach(engine=None)
            _Host().send("a")
            _Host().send("b")
        finally:
            registry.detach()
        assert report.calls("_Host.send") == 2


class TestItReportsWhatHappened:
    def test_an_unavailable_strategy_is_reported_with_its_reason(self) -> None:
        registry = Registry([_FakeStrategy(unavailable="not installed")])
        report = registry.attach(engine=None)
        try:
            assert report.attached == []
            assert report.unavailable == {"fake": "not installed"}
        finally:
            registry.detach()

    def test_it_knows_which_layers_it_attached(self) -> None:
        registry = Registry([_FakeStrategy(layer="tool")])
        try:
            report = registry.attach(engine=None)
            assert report.layers() == {"tool"}
        finally:
            registry.detach()


class TestSilenceIsLoud:
    """The failure this whole module exists to prevent."""

    def test_it_refuses_a_layer_nobody_attached(self) -> None:
        registry = Registry([_FakeStrategy(unavailable="not installed")])
        report = registry.attach(engine=None)
        try:
            with pytest.raises(ConfigError) as caught:
                report.require("llm")
        finally:
            registry.detach()
        message = str(caught.value)
        assert "no llm layer attached" in message
        assert "fake" in message, "the diagnosis must name every strategy it tried"
        assert "not installed" in message, "and why each one could not help"
        assert "seams" in message, "and how to fix it by hand"

    def test_it_says_a_hook_attached_but_saw_nothing(self) -> None:
        """Patched and never called is a different bug from never patched."""
        registry = Registry([_FakeStrategy()])
        report = registry.attach(engine=None)
        registry.detach()
        with pytest.raises(ConfigError, match="saw 0 calls"):
            report.require("llm")

    def test_it_is_satisfied_by_a_hook_that_saw_traffic(self) -> None:
        registry = Registry([_FakeStrategy()])
        try:
            report = registry.attach(engine=None)
            _Host().send("x")
        finally:
            registry.detach()
        report.require("llm")  # must not raise
