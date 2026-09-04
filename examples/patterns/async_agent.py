"""An async agent: awaited tools, awaited model calls, and a concurrent fan-out.

The shape behind an HTTP handler. The model is asked which lookups to run, the
lookups run together under `asyncio.gather`, and the model drafts a quote from what
came back. It is also the async-parity proof: every interception point it crosses --
tool and llm -- is a coroutine, and `ChaosEngine.arun` drives the entrypoint.

The weakness in `build` is the fan-out: `asyncio.gather` with neither
`return_exceptions=True` nor a per-task timeout. `gather` is all-or-nothing, so the
first task to raise discards the siblings that already succeeded -- the price came
back fine and is thrown away with the inventory failure. The retry re-runs the whole
batch, and once the attempt budget is gone the agent drafts a quote from an empty
fact table. The model, asked for a price nobody gave it, invents one; that invention
is the finding, and the lost price is the cause.

`build_fixed` gathers with `return_exceptions=True` under a per-task `wait_for`,
keeps every partial result, retries only the lookup that failed, and when it still
has nothing says which lookup was unavailable instead of asking the model to fill
the hole.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from . import PatternSpec, register

__all__ = ["PATTERN", "build", "build_fixed", "fetch_inventory", "fetch_pricing"]

Tool = Callable[[str], Awaitable[dict[str, Any]]]
Model = Callable[[str], Awaitable[str]]

_DEFAULT_SKU = "SKU-77"
#: Bounds the retry loop: a fault must never be able to hang the suite.
_MAX_ATTEMPTS = 3
#: Generous on purpose. The injected timeout raises immediately, so this never fires
#: in the test run and no wall-clock reading reaches a decision.
_TOOL_TIMEOUT_S = 2.0

_CATALOG: dict[str, dict[str, Any]] = {
    "SKU-77": {"list_price_usd": 249, "on_hand": 18, "warehouse": "EU-2"},
    "SKU-12": {"list_price_usd": 96, "on_hand": 240, "warehouse": "US-1"},
}

_PLAN_TAG = "[PLAN]"
_PLAN_PROMPT = f"""{_PLAN_TAG}
Quote SKU {{sku}} for a customer.
Reply with the lookups to run, comma separated."""

_QUOTE_PROMPT = """[QUOTE]
sku: {sku}
list_price_usd: {price}
on_hand: {on_hand}
Write one sentence for the customer."""

_PLAN_REPLY = "fetch_pricing, fetch_inventory"
#: What the scripted model says when the prompt has no price in it. A real mediocre
#: model does this, and it is the whole reason the grounding assertion exists.
_INVENTED_PRICE = "319"
_INVENTED_STOCK = "24"


async def fetch_pricing(sku: str) -> dict[str, Any]:
    """Look up list price. Read-only.

    Args:
        sku: The catalog id.

    Returns:
        ``{"sku", "list_price_usd", "currency"}``.

    Raises:
        KeyError: When the sku is not in the catalog.
    """
    return {"sku": sku, "list_price_usd": _CATALOG[sku]["list_price_usd"], "currency": "USD"}


async def fetch_inventory(sku: str) -> dict[str, Any]:
    """Look up stock on hand. Read-only.

    Args:
        sku: The catalog id.

    Returns:
        ``{"sku", "on_hand", "warehouse"}``.

    Raises:
        KeyError: When the sku is not in the catalog.
    """
    row = _CATALOG[sku]
    return {"sku": sku, "on_hand": row["on_hand"], "warehouse": row["warehouse"]}


def _field(prompt: str, key: str) -> str:
    """Read one ``key: value`` line out of a prompt.

    Args:
        prompt: The rendered prompt.
        key: The line's key.

    Returns:
        The value, or ``""`` when the line is absent or blank.
    """
    prefix = f"{key}:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _scripted_model() -> Model:
    """Build a deterministic stand-in for a mediocre model.

    Keyed on the call counter (turn one plans) and then on the prompt's fields. No
    network, no clock, no randomness. Asked to report a field the prompt does not
    carry, it fills the gap with a confident number rather than admitting ignorance.

    Returns:
        An async ``model(prompt) -> str``.
    """
    turn = 0

    async def model(prompt: str) -> str:
        """Answer one prompt.

        Args:
            prompt: The rendered prompt.

        Returns:
            The scripted reply.
        """
        nonlocal turn
        turn += 1
        if turn == 1 or prompt.startswith(_PLAN_TAG):
            return _PLAN_REPLY
        sku = _field(prompt, "sku") or _DEFAULT_SKU
        price = _field(prompt, "list_price_usd") or _INVENTED_PRICE
        on_hand = _field(prompt, "on_hand") or _INVENTED_STOCK
        return f"{sku} is listed at {price} USD, with {on_hand} units on hand."

    return model


def _bind(engine: Any) -> tuple[dict[str, Tool], Model]:
    """Instrument the tools and the model, or hand back the raw callables.

    Args:
        engine: A `ChaosEngine`, or `None` to run uninstrumented.

    Returns:
        ``(lookups, model)``.
    """
    raw: dict[str, Tool] = {"fetch_pricing": fetch_pricing, "fetch_inventory": fetch_inventory}
    model = _scripted_model()
    if engine is None:
        return raw, model
    lookups = {name: engine.tool(fn, name=name, side_effecting=False) for name, fn in raw.items()}
    return lookups, engine.llm(model, name="quote")


def _planned(reply: str, lookups: dict[str, Tool]) -> list[str]:
    """Resolve the model's plan to known lookup names.

    Args:
        reply: The planner's reply.
        lookups: The available lookups.

    Returns:
        The lookup names to run, in the order the model asked for them.
    """
    names = [part.strip() for part in reply.split(",")]
    return [name for name in names if name in lookups] or sorted(lookups)


def build(engine: Any = None) -> Callable[..., Awaitable[str]]:
    """Build the naive async agent.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None`.

    Returns:
        A coroutine function ``agent(sku=None) -> str``.
    """
    lookups, model = _bind(engine)

    async def agent(sku: str | None = None) -> str:
        """Quote one sku.

        Args:
            sku: The catalog id. Defaults to `_DEFAULT_SKU`.

        Returns:
            One sentence for the customer.
        """
        target = sku or _DEFAULT_SKU
        names = _planned(await model(_PLAN_PROMPT.format(sku=target)), lookups)
        facts: dict[str, Any] = {}
        for _ in range(_MAX_ATTEMPTS):
            try:
                batch = await asyncio.gather(*(lookups[name](target) for name in names))
            except Exception:
                continue
            for result in batch:
                facts.update(result)
            break
        return await model(
            _QUOTE_PROMPT.format(
                sku=target,
                price=facts.get("list_price_usd", ""),
                on_hand=facts.get("on_hand", ""),
            )
        )

    return agent


async def _settle(
    lookups: dict[str, Tool], names: list[str], sku: str
) -> tuple[dict[str, Any], list[str]]:
    """Run several lookups concurrently, keeping whatever succeeds.

    Args:
        lookups: The available lookups.
        names: Which to run.
        sku: The catalog id.

    Returns:
        ``(facts, failed_names)``.
    """
    batch = await asyncio.gather(
        *(asyncio.wait_for(lookups[name](sku), _TOOL_TIMEOUT_S) for name in names),
        return_exceptions=True,
    )
    facts: dict[str, Any] = {}
    failed: list[str] = []
    for name, result in zip(names, batch, strict=True):
        if isinstance(result, BaseException):
            failed.append(name)
        else:
            facts.update(result)
    return facts, failed


def build_fixed(engine: Any = None) -> Callable[..., Awaitable[str]]:
    """Build the hardened async agent.

    Partial results survive a sibling's failure, only the failed lookup is retried,
    and a lookup that never lands is named in the answer rather than papered over by
    the model.

    Args:
        engine: A `ChaosEngine` to instrument against, or `None`.

    Returns:
        A coroutine function ``agent(sku=None) -> str``.
    """
    lookups, model = _bind(engine)

    async def agent(sku: str | None = None) -> str:
        """Quote one sku, degrading to what is actually known.

        Args:
            sku: The catalog id. Defaults to `_DEFAULT_SKU`.

        Returns:
            One sentence for the customer, naming any lookup that was unavailable.
        """
        target = sku or _DEFAULT_SKU
        names = _planned(await model(_PLAN_PROMPT.format(sku=target)), lookups)
        facts: dict[str, Any] = {}
        pending = names
        attempts = 0
        for _ in range(_MAX_ATTEMPTS):
            attempts += 1
            landed, pending = await _settle(lookups, pending, target)
            facts.update(landed)
            if not pending:
                break
        if engine is not None:
            engine.validated(sorted(facts), name="lookups_landed")
        if not pending:
            return await model(
                _QUOTE_PROMPT.format(
                    sku=target, price=facts["list_price_usd"], on_hand=facts["on_hand"]
                )
            )
        missing = ", ".join(pending)
        if engine is not None:
            engine.note(f"degraded: {missing} failed on {attempts} attempts; quoting partials only")
        price = facts.get("list_price_usd")
        priced = f"{target} is listed at {price} USD." if price is not None else f"{target}:"
        return (
            f"{priced} Stock on hand is unavailable -- {missing} did not respond on "
            f"{attempts} attempts -- so this quote omits availability."
        )

    return agent


PATTERN = register(
    PatternSpec(
        name="async_agent",
        description="An async agent fanning two lookups out concurrently, as a web handler would.",
        build=build,
        build_fixed=build_fixed,
        weakness=(
            "the fan-out is `asyncio.gather` with no `return_exceptions=True` and no "
            "per-task timeout, so one failing lookup discards the sibling result that "
            "already succeeded and the model invents the number that went missing"
        ),
        faults=(
            {
                "type": "ToolTimeoutFault",
                "params": {"after_ms": 1500},
                "target": {"tool": "fetch_inventory"},
                "trigger": {"on_call": [1, 2, 3], "max_fires": 3},
            },
        ),
        inputs=_DEFAULT_SKU,
        tools=("fetch_pricing", "fetch_inventory"),
        llm_names=("quote",),
        is_async=True,
        expected_behavior="graceful_degradation",
    )
)


async def _demo() -> None:
    """Run both trees uninstrumented and check the happy path.

    Raises:
        AssertionError: When either tree stops quoting the catalog's own numbers.
    """
    for builder in (build, build_fixed):
        answer = await builder(None)(_DEFAULT_SKU)
        assert "249 USD" in answer, answer
        assert "18 units" in answer, answer


if __name__ == "__main__":
    asyncio.run(_demo())
    print("ok")
