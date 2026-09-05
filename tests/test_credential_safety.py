"""Credentials, when the agent under test is a real one.

The engine never needs a credential of its own: it wraps the agent's callables, and
the agent authenticates exactly as it does in production. What matters is the other
direction — a credential the agent legitimately holds must not come back out in the
artifacts, because `report.json` and `AGENT_TASK.md` are written to be attached to
tickets and handed to coding agents.

`CLAUDE.md` states the invariant: *every payload passes through `redact.py` before
serialization*. The trace honoured it. `report.tool_calls[]` and
`report.llm_exchanges[]` did not, so a bearer token in a tool's keyword arguments was
redacted in `trace.jsonl` and printed in full in the report beside it (D-131).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine

TOKEN = "sk-live-9f3a2b7c4d1e8f60"
HMAC = "9c1f4ade77b2"


def _leaks(document: Any, secret: str, path: str = "") -> list[str]:
    """Every place a secret appears in a nested document."""
    if isinstance(document, str):
        return [path] if secret in document else []
    if isinstance(document, dict):
        return [p for k, v in document.items() for p in _leaks(v, secret, f"{path}.{k}")]
    if isinstance(document, list):
        return [p for i, v in enumerate(document) for p in _leaks(v, secret, f"{path}[{i}]")]
    return []


def _run(tmp_path: Path, **engine_kw: Any) -> tuple[Any, str]:
    engine = ChaosEngine(
        seed=1337, out_dir=tmp_path, judge="rules", trace_level="verbose", **engine_kw
    )

    @engine.tool
    def fetch_invoice(invoice_id: str, *, authorization: str, x_signature: str) -> dict[str, Any]:
        return {"invoice_id": invoice_id, "amount_usd": 412}

    @engine.llm(name="summarizer")
    def complete(prompt: str) -> str:
        return "The invoice is open."

    @engine.intercept_tools()
    def agent(question: str) -> str:
        invoice = fetch_invoice("INV-77", authorization=f"Bearer {TOKEN}", x_signature=HMAC)
        return complete(f"Answer from: {invoice}")

    result = engine.run(agent, inputs="what is the status of INV-77?")
    trace = next(tmp_path.rglob("trace.jsonl")).read_text(encoding="utf-8")
    return result, trace


class TestTheReportRedactsWhatTheTraceDoes:
    def test_a_bearer_token_never_reaches_the_report(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        found = _leaks(result.to_dict(), TOKEN)
        assert not found, f"the credential is in the report at {found}"

    def test_it_never_reaches_the_trace_either(self, tmp_path: Path) -> None:
        _, trace = _run(tmp_path)
        assert TOKEN not in trace

    def test_the_tool_call_row_still_shows_the_argument_was_there(self, tmp_path: Path) -> None:
        """Redaction, not deletion: a reader must see the call was authenticated."""
        result, _ = _run(tmp_path)
        row = result.to_dict()["tool_calls"][0]
        assert "authorization" in row["kwargs"]
        assert "redacted" in row["kwargs"]["authorization"]

    def test_the_non_secret_arguments_survive(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        row = result.to_dict()["tool_calls"][0]
        assert row["args"] == ["INV-77"]

    def test_the_work_order_carries_no_credential(self, tmp_path: Path) -> None:
        from agent_loop_chaos.bundle import render_agent_task

        result, _ = _run(tmp_path)
        assert TOKEN not in render_agent_task(result)


class TestACustomCredentialNameNeedsDeclaring:
    def test_an_unknown_key_leaks_by_default(self, tmp_path: Path) -> None:
        """Documented, not a defect: the deny-list cannot guess `x_signature`.

        This test exists so the behaviour is stated somewhere executable, and so the
        fix below is demonstrably the fix.
        """
        result, _ = _run(tmp_path)
        assert _leaks(result.to_dict(), HMAC), "the default deny-list matched a custom name"

    def test_redact_keys_closes_it(self, tmp_path: Path) -> None:
        result, trace = _run(tmp_path / "declared", redact_keys=["x_signature"])
        assert not _leaks(result.to_dict(), HMAC)
        assert HMAC not in trace


class TestPrefixedApiKeyShapes:
    @pytest.mark.parametrize(
        "value",
        [
            "sk-live-9f3a2b7c4d1e8f60",
            "sk-proj-abcdefghijklmnopqrst",
            "sk-ant-api03-abcdefghijklmnop",
            "sk-test-51H8xYzAbCdEfGhIjKlMn",
            "rk-live-abcdefghijklmnopqrst",
        ],
    )
    def test_a_segmented_key_is_recognised(self, value: str) -> None:
        """Every current provider issues prefixed keys: `sk-proj-`, `sk-ant-`,
        `sk-live-`. The original pattern required 16 alphanumerics straight after
        `sk-`, so a hyphen four characters in defeated it."""
        from agent_loop_chaos.redact import redaction_reason

        assert redaction_reason(None, value) is not None, f"{value} was not recognised"

    @pytest.mark.parametrize(
        "value",
        ["sk-short", "task-management-system", "risk-assessment-report-2026", "ask-me-anything"],
    )
    def test_ordinary_text_is_not_redacted(self, value: str) -> None:
        """A pattern loose enough to catch prose would redact the evidence."""
        from agent_loop_chaos.redact import redaction_reason

        assert redaction_reason(None, value) is None, f"{value} was wrongly redacted"


def test_the_engine_needs_no_credential_of_its_own(tmp_path: Path) -> None:
    """The point that makes all of this workable: the agent authenticates itself.

    The engine wraps callables. It never holds, reads, or forwards a credential, so
    pointing it at an authenticated agent requires configuring nothing.
    """
    import inspect

    signature = inspect.signature(ChaosEngine.__init__)
    # `redact_keys` and `allow_side_effects` name *patterns* and *tool names*; neither
    # carries a value. Nothing here should ever accept one.
    carriers = {"key", "token", "secret", "credential", "password"}
    for name in signature.parameters:
        if name in {"redact_keys", "allow_side_effects", "allow_remote_judge"}:
            continue
        assert not any(word in name for word in carriers), (
            f"ChaosEngine takes {name!r}; it should never need a credential"
        )
