"""The vanilla adapter: sync and async parity, inertness, and calling conventions.

Async and sync parity is structural rather than remembered: `build_wrapper` produces
a coroutine wrapper for a coroutine function, and both paths are tested here.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.adapters.vanilla import normalize_messages, resolve_invocation
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults.base import NoopFault
from agent_loop_chaos.targeting import Target


def engine(tmp_path: Path, **kw: Any) -> ChaosEngine:
    """Build an engine writing into a temp directory.

    Args:
        tmp_path: pytest's temp directory.
        **kw: Constructor overrides.

    Returns:
        The engine.
    """
    kw.setdefault("out_dir", tmp_path / ".chaos")
    kw.setdefault("strict_trace", True)
    return ChaosEngine(**kw)


# ------------------------------------------------------------------- decorators


def test_sync_tool_is_intercepted(tmp_path: Path) -> None:
    """The base case."""
    eng = engine(tmp_path)

    @eng.tool
    def fetch(x: int) -> int:
        return x * 2

    result = eng.run(lambda: fetch(21))
    assert result.final_output == 42
    assert result.metrics["tool_calls"] == 1


async def test_async_tool_is_intercepted(tmp_path: Path) -> None:
    """A coroutine function must yield a coroutine wrapper."""
    eng = engine(tmp_path)

    @eng.tool
    async def fetch(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    assert inspect.iscoroutinefunction(fetch)

    async def agent() -> int:
        return await fetch(21)

    result = await eng.arun(agent)
    assert result.final_output == 42
    assert result.metrics["tool_calls"] == 1


def test_sync_llm_is_intercepted(tmp_path: Path) -> None:
    """One LLM call is one step under vanilla (D-05)."""
    eng = engine(tmp_path)

    @eng.llm
    def gen(prompt: str) -> str:
        return f"re: {prompt}"

    result = eng.run(lambda: gen("hi"))
    assert result.final_output == "re: hi"
    assert result.metrics["llm_calls"] == 1
    assert result.metrics["steps"] == 1


async def test_async_llm_is_intercepted(tmp_path: Path) -> None:
    """Parity with the sync path."""
    eng = engine(tmp_path)

    @eng.llm
    async def gen(prompt: str) -> str:
        await asyncio.sleep(0)
        return f"re: {prompt}"

    async def agent() -> str:
        return await gen("hi")

    result = await eng.arun(agent)
    assert result.final_output == "re: hi"
    assert result.metrics["llm_calls"] == 1


def test_the_parameterized_form_sets_the_crossing_name(tmp_path: Path) -> None:
    """`@engine.tool(name=…)` is how a target finds a renamed tool."""
    eng = engine(tmp_path)

    @eng.tool(name="flights")
    def search_flights(origin: str) -> str:
        return origin

    result = eng.run(lambda: search_flights("LHR"))
    assert result.tool_calls[0]["tool"] == "flights"


def test_wrappers_preserve_metadata_and_signature(tmp_path: Path) -> None:
    """A wrapper the user cannot introspect is a wrapper they cannot debug."""
    eng = engine(tmp_path)

    @eng.tool
    def documented(location: str, units: str = "c") -> None:
        """Fetch the weather."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Fetch the weather."
    assert list(inspect.signature(documented).parameters) == ["location", "units"]


# --------------------------------------------------------------------- inertness


def test_a_decorated_tool_is_inert_outside_a_run(tmp_path: Path) -> None:
    """Users leave the decorator on in production, so this must change nothing."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target_tool="fetch")

    @eng.tool
    def fetch(x: int) -> int:
        return x + 1

    assert fetch(1) == 2
    assert eng.is_active() is False


def test_inert_interception_is_cheap(tmp_path: Path) -> None:
    """The unarmed path is one contextvar read; the docs promise ~1 µs."""
    eng = engine(tmp_path)

    @eng.tool
    def noop(x: int) -> int:
        return x

    for _ in range(1000):
        noop(1)

    started = time.perf_counter()
    iterations = 20_000
    for _ in range(iterations):
        noop(1)
    per_call_us = (time.perf_counter() - started) / iterations * 1_000_000
    assert per_call_us < 5.0, f"{per_call_us:.2f} µs per inert call"


# ---------------------------------------------------------------- dict-of-tools


def test_wrap_tools_returns_a_new_mapping(tmp_path: Path) -> None:
    """Never mutate the caller's objects."""
    eng = engine(tmp_path)

    def a() -> str:
        return "a"

    def b() -> str:
        return "b"

    original = {"a": a, "b": b}
    wrapped = eng.wrap_tools(original)
    assert original["a"] is a
    assert wrapped["a"] is not a
    assert set(wrapped) == {"a", "b"}


def test_wrap_tools_is_intercepted_by_name(tmp_path: Path) -> None:
    """The shape a hand-rolled dispatch loop uses (`docs/06` §2.2)."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target=Target(tool="b", phase="post"))
    tools = eng.wrap_tools({"a": lambda: "a", "b": lambda: "b"})

    def agent() -> list[str]:
        return [tools["a"](), tools["b"]()]

    result = eng.run(agent)
    assert result.final_output == ["a", "b"]
    assert result.injected_faults[0]["fire_count"] == 1


# ------------------------------------------------------------- intercept_tools


def test_intercept_tools_restricts_to_named_tools(tmp_path: Path) -> None:
    """Passing names limits interception to them (`docs/06` §2.3)."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target=Target(tool="a", phase="post"))
    eng.register_fault(NoopFault(), target=Target(tool="b", phase="post"))

    @eng.tool
    def a() -> str:
        return "a"

    @eng.tool
    def b() -> str:
        return "b"

    @eng.intercept_tools("a")
    def agent() -> list[str]:
        return [a(), b()]

    result = eng.run(agent)
    assert result.injected_faults[0]["fire_count"] == 1
    assert result.injected_faults[1]["fire_count"] == 0
    assert result.metrics["tool_calls"] == 1


def test_intercept_tools_with_no_names_intercepts_everything(tmp_path: Path) -> None:
    """The bare form is the blueprint-compatible default."""
    eng = engine(tmp_path)
    eng.register_fault(NoopFault(), target=Target(tool="a", phase="post"))

    @eng.tool
    def a() -> str:
        return "a"

    @eng.intercept_tools()
    def agent() -> str:
        return a()

    assert eng.run(agent).injected_faults[0]["fire_count"] == 1


def test_intercept_tools_preserves_the_agent_metadata(tmp_path: Path) -> None:
    """It decorates the entrypoint, so the entrypoint must still look like itself."""
    eng = engine(tmp_path)

    @eng.intercept_tools()
    def weather_agent(user_query: str, state: dict[str, Any]) -> None:
        """Plan a trip."""

    assert weather_agent.__name__ == "weather_agent"
    assert weather_agent.__doc__ == "Plan a trip."


# ------------------------------------------------------------ class-based agents


def test_instrument_object_wraps_named_methods(tmp_path: Path) -> None:
    """`docs/06` §2.5: bare method name for tools, `Class.method` for nodes."""
    eng = engine(tmp_path)

    class Agent:
        def fetch(self, x: int) -> int:
            return x * 2

        def complete(self, prompt: str) -> str:
            return f"re: {prompt}"

        def plan(self) -> str:
            return "planned"

        def go(self) -> tuple[int, str, str]:
            return self.fetch(2), self.complete("hi"), self.plan()

    agent = eng.instrument_object(
        Agent(), tools=["fetch"], llm_methods=["complete"], nodes=["plan"]
    )
    result = eng.run(agent.go)
    assert result.final_output == (4, "re: hi", "planned")
    assert result.tool_calls[0]["tool"] == "fetch"
    assert result.metrics["llm_calls"] == 1


def test_instrument_object_rejects_a_missing_method(tmp_path: Path) -> None:
    """A typo in a method name is a config error, not a silent no-op."""

    class Agent:
        pass

    with pytest.raises(ConfigError, match="no callable attribute"):
        engine(tmp_path).instrument_object(Agent(), tools=["nope"])


# ---------------------------------------------------- message normalization


@pytest.mark.parametrize(
    ("payload", "expected_messages", "unavailable"),
    [
        ("hello", [{"role": "user", "content": "hello"}], False),
        ([{"role": "system", "content": "s"}], [{"role": "system", "content": "s"}], False),
        ([], [], False),
        (42, None, True),
        ({"not": "a list"}, None, True),
        ([{"no_role": 1}], None, True),
    ],
)
def test_message_normalization(payload: Any, expected_messages: Any, unavailable: bool) -> None:
    """A string becomes one user message; anything unrecognizable is reported.

    Saying so in the report beats guessing (`docs/06` §2.1).
    """
    messages, is_unavailable = normalize_messages(payload)
    assert messages == expected_messages
    assert is_unavailable is unavailable


def test_an_unnormalizable_llm_payload_is_recorded_not_guessed(tmp_path: Path) -> None:
    """The agent still runs; the report states the payload could not be normalized."""
    eng = engine(tmp_path)

    @eng.llm
    def gen(payload: object) -> str:
        return "ok"

    result = eng.run(lambda: gen(object()))
    assert result.final_output == "ok"
    assert result.metrics["llm_calls"] == 1


# ------------------------------------------------- calling conventions (D-19)


def test_mapping_inputs_matching_parameter_names_are_spread() -> None:
    """D-19 rule 1."""

    def agent(query: str, depth: int) -> None:
        pass

    assert resolve_invocation(agent, {"query": "q", "depth": 2}, None) == (
        (),
        {"query": "q", "depth": 2},
    )


def test_mapping_inputs_with_no_matching_keys_are_passed_positionally() -> None:
    """D-19 rule 2."""

    def agent(payload: dict[str, Any]) -> None:
        pass

    args, kwargs = resolve_invocation(agent, {"unrelated": 1}, None)
    assert args == ({"unrelated": 1},)
    assert kwargs == {}


def test_scalar_inputs_are_passed_positionally() -> None:
    """D-19 rule 3."""

    def agent(q: str) -> None:
        pass

    assert resolve_invocation(agent, "hi", None) == (("hi",), {})


def test_no_inputs_calls_the_agent_bare() -> None:
    """D-19 rule 3, the `None` branch."""

    def agent() -> None:
        pass

    assert resolve_invocation(agent, None, None) == ((), {})


@pytest.mark.parametrize("param", ["state", "initial_state"])
def test_initial_state_is_passed_only_when_named(param: str) -> None:
    """D-19 rule 4."""
    namespace: dict[str, Any] = {}
    exec(f"def agent(q, {param}): pass", namespace)
    _args, kwargs = resolve_invocation(namespace["agent"], "q", {"k": 1})
    assert kwargs == {param: {"k": 1}}


def test_initial_state_is_withheld_when_not_named() -> None:
    """The engine holds it and exposes it through `state_view` instead (D-19)."""

    def agent(q: str) -> None:
        pass

    _args, kwargs = resolve_invocation(agent, "q", {"k": 1})
    assert kwargs == {}


def test_a_signature_mismatch_is_a_config_error_not_a_type_error() -> None:
    """Never a `TypeError` from inside the agent (D-19)."""

    def agent() -> None:
        pass

    with pytest.raises(ConfigError, match="takes no positional parameter"):
        resolve_invocation(agent, "unexpected", None)


def test_var_keyword_agents_accept_a_mapping() -> None:
    """`**kwargs` means every key is a named parameter."""

    def agent(**kwargs: Any) -> None:
        pass

    assert resolve_invocation(agent, {"a": 1}, None) == ((), {"a": 1})
