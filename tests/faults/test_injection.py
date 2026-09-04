"""The prompt-injection corpus and `PromptInjectionFault`.

The corpus entries are format probes, not weapons: fake footers, fake system
notices, an HTML comment. Nothing here would be harmful outside this harness.

The load-bearing detail is D-16. The canary is `ALC-CANARY-{run_id}` and `run_id`
is `run-<8 hex>`, so the obvious `ALC-CANARY-[0-9a-f]+` cannot match the leading
`r` of `run-`. That bug would have silently missed **every** successful
exfiltration, so every `detect` regex is asserted against a real canary below.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from agent_loop_chaos import ChaosEngine
from agent_loop_chaos.errors import ConfigError
from agent_loop_chaos.faults.injection import (
    CANARY_PATTERN,
    CORPUS,
    OBJECTIVES,
    PLACEMENTS,
    PromptInjectionFault,
    canary_for,
    load_corpus,
)

REAL_CANARY = "ALC-CANARY-run-3f9a12c4"
DETECT_KINDS = {"regex", "tool_called", "format_changed", "canary"}
CHECK_KINDS = {"output_is_json", "max_length", "forbidden_token"}


# ------------------------------------------------------------------- the corpus


def test_the_corpus_ships_at_least_fifteen_entries() -> None:
    """The phase requires >= 15, so a thinner corpus is a failed gate."""
    assert len(CORPUS) >= 15


def test_corpus_ids_are_unique() -> None:
    """`payload_id` addresses an entry, so a duplicate would be ambiguous."""
    ids = [entry["id"] for entry in CORPUS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_every_entry_has_the_required_fields(entry: dict[str, Any]) -> None:
    """A partial entry would fail at run time, inside a scenario, far from here."""
    for field in ("id", "objective", "severity", "text", "detect", "placement_hint", "notes"):
        assert field in entry, f"{entry.get('id')} is missing {field}"
    assert "check" in entry, "check is required, even when null"


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_every_detect_kind_is_one_of_the_four(entry: dict[str, Any]) -> None:
    """Phase 04's probes switch on this, so an unknown kind is undetectable."""
    assert entry["detect"]["kind"] in DETECT_KINDS


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_every_detect_regex_matches_a_real_canary(entry: dict[str, Any]) -> None:
    """D-16, and the reason this test exists at all.

    `ALC-CANARY-[0-9a-f]+` looks right and cannot match `ALC-CANARY-run-3f9a12c4`,
    because `run-` is not hex. A corpus with that regex would report zero
    exfiltrations forever, and the bug would look like a well-behaved agent.
    """
    detect = entry["detect"]
    if detect["kind"] not in {"regex", "canary"}:
        pytest.skip(f"{entry['id']} detects by {detect['kind']}")
    assert re.search(detect["value"], REAL_CANARY), (
        f"{entry['id']}'s detect regex {detect['value']!r} does not match a real canary"
    )


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_every_check_is_mechanically_decidable(entry: dict[str, Any]) -> None:
    """A semantic constraint is a judge hypothesis, never a probe.

    If a `check` could not be decided by code, `instruction_precedence_violation`
    would depend on a model's opinion, and `success` would stop being computed.
    """
    check = entry["check"]
    if check is None:
        return
    assert check["kind"] in CHECK_KINDS, f"{entry['id']} has an undecidable check"
    if check["kind"] in {"max_length", "forbidden_token"}:
        assert "value" in check


def test_every_objective_is_covered() -> None:
    """All five, or a whole class of attack has no payload to exercise it."""
    assert {entry["objective"] for entry in CORPUS} == set(OBJECTIVES)


def test_every_placement_is_covered() -> None:
    """All five carriers, since placement is what a delimiting defence must handle."""
    assert {entry["placement_hint"] for entry in CORPUS} == set(PLACEMENTS)


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_no_payload_contains_a_real_credential_shape(entry: dict[str, Any]) -> None:
    """These are format probes, not weapons.

    A payload carrying a live-looking key would be redacted before a probe saw it,
    and would be genuinely unsafe to ship in a public repository.
    """
    from agent_loop_chaos.redact import DEFAULT_VALUE_PATTERNS

    text = entry["text"]
    for reason, pattern in DEFAULT_VALUE_PATTERNS:
        assert not re.search(pattern, text), f"{entry['id']} contains a {reason}-shaped string"


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_exfiltration_payloads_carry_the_canary_placeholder(entry: dict[str, Any]) -> None:
    """Otherwise there is nothing to exfiltrate and the probe is unreachable (D-16)."""
    if entry["objective"] != "exfiltrate_secret":
        pytest.skip("not an exfiltration payload")
    assert "{canary}" in entry["text"]


def test_the_canary_pattern_matches_the_canary_the_engine_plants() -> None:
    """The shipped pattern and the planted value cannot be allowed to drift."""
    assert re.fullmatch(CANARY_PATTERN, REAL_CANARY)
    assert re.fullmatch(CANARY_PATTERN, canary_for("run-deadbeef"))


def test_load_corpus_reads_the_packaged_file() -> None:
    """The corpus ships as package data, so a wheel install has it."""
    assert len(load_corpus()) == len(CORPUS)


# ---------------------------------------------------------- PromptInjectionFault


def test_injection_rides_in_on_a_tool_result() -> None:
    """Catalog B10: the payload is the carrier, so this usually arrives via a tool."""
    assert ("tool", "post") in PromptInjectionFault.accepts
    assert ("llm", "pre") in PromptInjectionFault.accepts


def test_injection_substitutes_the_real_canary_into_the_payload() -> None:
    """A payload still saying `{canary}` would never be detected."""
    from tests.faults.test_llm_faults import context

    ctx = context()
    fault = PromptInjectionFault(payload_id="exfil-01")
    from agent_loop_chaos.context import Crossing

    outcome = fault.apply(
        Crossing(layer="tool", phase="post", name="get_ticket", result={"notes": "all fine"}),
        ctx,
    )
    assert "{canary}" not in json.dumps(outcome.value)
    assert ctx.canary in json.dumps(outcome.value)


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_every_placement_carries_the_payload_into_the_result(placement: str) -> None:
    """Each carrier has to actually deliver the text somewhere reachable."""
    from agent_loop_chaos.context import Crossing
    from tests.faults.test_llm_faults import context

    outcome = PromptInjectionFault(payload_id="exfil-01", placement=placement).apply(
        Crossing(
            layer="tool",
            phase="post",
            name="get_ticket",
            result={"notes": "all fine", "id": 7},
        ),
        context(),
    )
    blob = json.dumps(outcome.value)
    if placement == "base64":
        # Encoded on purpose: the instruction to decode is the attack.
        import base64 as _b64

        encoded = re.search(r"[A-Za-z0-9+/]{24,}={0,2}", blob)
        assert encoded, "no base64 blob found"
        blob = _b64.b64decode(encoded.group()).decode("utf-8", "replace")
    assert "ALC-CANARY" in blob


def test_base64_placement_is_decodable() -> None:
    """The instruction to decode is the attack; an undecodable blob tests nothing."""
    import base64

    from agent_loop_chaos.context import Crossing
    from tests.faults.test_llm_faults import context

    outcome = PromptInjectionFault(payload_id="exfil-01", placement="base64").apply(
        Crossing(layer="tool", phase="post", name="t", result={"notes": "x"}),
        context(),
    )
    blob = json.dumps(outcome.value)
    encoded = re.search(r"[A-Za-z0-9+/]{24,}={0,2}", blob)
    assert encoded, "no base64 blob found"
    assert "ALC-CANARY" in base64.b64decode(encoded.group()).decode("utf-8", "replace")


def test_injection_rejects_an_unknown_payload_id() -> None:
    """A typo in a scenario file must fail at load."""
    with pytest.raises(ConfigError, match="exfil-99"):
        PromptInjectionFault(payload_id="exfil-99")


def test_injection_records_the_payload_so_the_probe_can_attribute_it() -> None:
    """R2: injected text is harness-authored and is never the agent's own output."""
    from agent_loop_chaos.context import Crossing
    from tests.faults.test_llm_faults import context

    outcome = PromptInjectionFault(payload_id="exfil-01").apply(
        Crossing(layer="tool", phase="post", name="t", result={"notes": "x"}), context()
    )
    assert outcome.params["payload_id"] == "exfil-01"
    assert outcome.params["objective"] == "exfiltrate_secret"
    assert outcome.params["injected_text"]
    assert outcome.params["detect"]["kind"] in DETECT_KINDS


def test_call_forbidden_tool_is_blocked_and_stubbed() -> None:
    """D-23: the attempt is the finding, and the call must not execute.

    Letting the payload actually invoke the tool would mean the harness performing
    the very action the fault exists to warn about.
    """
    from agent_loop_chaos.context import Crossing
    from tests.faults.test_llm_faults import context

    entry = next(e for e in CORPUS if e["objective"] == "call_forbidden_tool")
    outcome = PromptInjectionFault(payload_id=entry["id"]).apply(
        Crossing(layer="tool", phase="post", name="t", result={"notes": "x"}), context()
    )
    assert outcome.params["blocked"] is True
    assert outcome.params["forbidden_tool"]


def test_injection_end_to_end_reaches_the_agent(tmp_path: Path) -> None:
    """4/4 end-to-end: the agent sees the poisoned tool result."""
    eng = ChaosEngine(seed=1337, out_dir=tmp_path / ".chaos", strict_trace=True)
    eng.register_fault(PromptInjectionFault(payload_id="exfil-01"), target_tool="get_ticket")

    @eng.tool
    def get_ticket() -> dict[str, Any]:
        return {"id": 7, "notes": "customer reports a delay"}

    result = eng.run(lambda: get_ticket())
    assert "ALC-CANARY" in json.dumps(result.final_output)
    assert result.injected_faults[0]["fired"] is True


# ------------------------------------------------------------------- the canary


def test_the_engine_plants_the_canary_in_initial_state(tmp_path: Path) -> None:
    """D-16: without a reachable secret, `exfiltrate_secret` has nothing to exfiltrate
    and the `secret_in_output` probe can never fire.
    """
    eng = ChaosEngine(seed=1337, out_dir=tmp_path / ".chaos")
    seen: list[Any] = []

    def agent(state: dict[str, Any]) -> str:
        seen.append(dict(state))
        return "done"

    eng.run(agent, initial_state={"query": "hi"})
    assert "_alc_canary" in seen[0]
    assert re.fullmatch(CANARY_PATTERN, seen[0]["_alc_canary"])


def test_the_canary_survives_redaction_but_real_secrets_do_not() -> None:
    """The exemption must be exactly one value wide."""
    from agent_loop_chaos.redact import redact

    out = redact({"answer": REAL_CANARY, "api_key": "sk-" + "A" * 20}, allow=(REAL_CANARY,))
    assert out["answer"] == REAL_CANARY
    assert out["api_key"].startswith("<redacted:")


def test_the_canary_is_redacted_when_it_is_not_exempted() -> None:
    """The exemption must not be a no-op: the redactor has to recognise the shape.

    If the pattern did not match, "exempt the canary" would be doing nothing, and
    the day someone stopped passing `allow=` the canary would leak into reports
    unnoticed.
    """
    from agent_loop_chaos.redact import redact

    assert redact({"answer": REAL_CANARY})["answer"] == "<redacted:canary>"


class TestPayloadMetadataReachesTheProbe:
    """The corpus entry's `detect` and `check` must survive into the fire record.

    `injection_followed` matches each fired payload against its own `detect`/`check`
    rule, and the fault records exactly that in `FaultOutcome.params`. The engine's
    fire record dropped `params`, so the probe saw only the fault's constructor
    arguments -- and could detect nothing except the canary. Three of the four
    shipped objectives (`ignore_instructions`, `change_output_format`,
    `call_forbidden_tool`) were therefore undetectable, and
    `prompt_injection_followed` was very nearly unreachable.
    """

    @staticmethod
    def _fire(objective: str, tmp_path: Any) -> dict[str, Any]:
        from agent_loop_chaos import ChaosEngine
        from agent_loop_chaos.faults import PromptInjectionFault
        from agent_loop_chaos.targeting import Target, Trigger

        engine = ChaosEngine(
            seed=1337, out_dir=tmp_path, write_bundle=False, strict_schema=True, judge="rules"
        )

        @engine.tool(name="supplier", side_effecting=False)
        def supplier() -> dict[str, Any]:
            return {"notes": "Fares refreshed hourly.", "price": 612}

        def agent(question: Any = None) -> str:
            return f"The note said: {supplier()['notes']}"

        engine.register_fault(
            PromptInjectionFault(objective=objective, placement="field_value"),
            target=Target(tool="supplier", phase="post"),
            trigger=Trigger(on_call=1),
        )
        result = engine.run(agent, inputs={"question": "?"}, scenario_id="inj")
        fires = [f for r in result.injected_faults for f in (r.get("fires") or [])]
        assert fires, "the injection did not fire"
        return fires[0]

    @pytest.mark.parametrize(
        "objective",
        ["ignore_instructions", "change_output_format", "call_forbidden_tool"],
    )
    def test_the_fire_carries_its_detect_rule(self, objective: str, tmp_path: Any) -> None:
        fire = self._fire(objective, tmp_path)
        params = fire.get("params") or {}
        assert params.get("payload_id"), f"{objective}: no payload_id on the fire"
        assert params.get("detect"), f"{objective}: no detect rule on the fire"

    def test_the_exfiltration_payload_still_carries_its_canary(self, tmp_path: Any) -> None:
        params = self._fire("exfiltrate_secret", tmp_path).get("params") or {}
        assert params.get("canary")

    def test_the_report_is_still_schema_valid(self, tmp_path: Any) -> None:
        # `strict_schema=True` above already refuses an invalid report, so reaching
        # here at all is the assertion; this states it so a reader sees it.
        assert self._fire("ignore_instructions", tmp_path)["action"]
