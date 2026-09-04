"""Every probe must have a negative control, and this is what enforces it.

Nine library bugs surfaced in phase 08. **Five failed on the *correct* tree, not the
buggy one** -- `retried_then_succeeded` unreachable, the grounding check sourcing a
number from the model's own sentence, `output_non_empty` demanded of a run that was
asked to raise, state targets that could never match, `injection_followed` able to
detect only the canary. Every one was invisible until a correct agent ran, and the
probes had gone four milestones without one.

False negatives are acceptable; false positives are bugs. A probe that fires on
correct behaviour makes the negative control unpassable and the whole suite
untrustworthy. So: a probe with no agent that exercises its code path *without*
tripping it has never been shown not to be a false positive.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent_loop_chaos.probes import PROBES

CODES = sorted(p.code for p in PROBES)

#: Probes whose negative control is a whole agent rather than a unit test, named here
#: so the gap is a deliberate, reviewable list rather than a silent absence. Each entry
#: says which control covers it.
CONTROLLED_ELSEWHERE: dict[str, str] = {
    "assertions_failed": "every `build_fixed` in examples/patterns and trip_planner_fixed",
    "injection_followed": "trip_planner_fixed under adversarial.injection_corpus",
    "secret_in_output": "trip_planner_fixed: the credential never enters a prompt",
    "redacted_value_in_output": "trip_planner_fixed: validated fields only",
    "duplicate_side_effect": "trip_planner_fixed: hold_booking carries an idempotency key",
}


def _control_names(code: str) -> list[str]:
    """Test names that look like a negative control for `code`.

    A control is a test whose name says the probe does *not* fire. The convention is
    checked rather than trusted, because "there is a test somewhere" is how the five
    bugs survived.

    Args:
        code: The probe code.

    Returns:
        Matching test function names across the probe test modules.
    """
    import tests.probes as package

    wanted = (
        f"{code}_does_not_fire",
        f"{code}_is_not_reported",
        f"no_{code}",
        f"{code}_stays_silent",
    )
    found: list[str] = []
    root = Path(package.__file__).parent
    for path in root.glob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        found.extend(name for name in wanted if f"def test_{name}" in text)
    return found


@pytest.mark.parametrize("code", CODES)
def test_every_probe_has_a_negative_control(code: str) -> None:
    """A probe with nothing proving it stays quiet has not been shown to be safe."""
    if code in CONTROLLED_ELSEWHERE:
        pytest.skip(f"controlled by an agent-level negative control: {CONTROLLED_ELSEWHERE[code]}")
    assert _control_names(code), (
        f"probe {code!r} has no negative control. Add a test named "
        f"`test_{code}_does_not_fire` that exercises the probe's code path with a "
        "correct agent and asserts it stays silent -- or, if the control is an agent "
        "rather than a unit test, add it to CONTROLLED_ELSEWHERE with a reason."
    )


def test_the_exemption_list_names_only_real_probes() -> None:
    """A stale exemption silently un-covers a probe."""
    unknown = sorted(set(CONTROLLED_ELSEWHERE) - set(CODES))
    assert unknown == [], f"CONTROLLED_ELSEWHERE names probes that do not exist: {unknown}"


def test_the_exemption_list_stays_small() -> None:
    """Exemptions are a debt, not a design. Keep the pressure on."""
    assert len(CONTROLLED_ELSEWHERE) <= 6, (
        "more than a quarter of the probes are exempt from a unit-level negative "
        "control; that is the state that let five false positives hide"
    )


@pytest.mark.parametrize("code", CODES)
def test_every_probe_documents_what_it_does_not_fire_on(code: str) -> None:
    """The docstring has to say where the probe deliberately stays quiet.

    A probe's false-positive boundary is the part a reader most needs and the part
    most easily left implicit. Writing it down is what makes a review of the
    attribution rules possible at all.
    """
    probe = next(p for p in PROBES if p.code == code)
    doc = inspect.getdoc(type(probe)) or ""
    assert doc.strip(), f"{code} has no docstring"
    markers = ("does not fire", "never fires", "not fire", "excludes", "harness", "attribut")
    assert any(m in doc.lower() for m in markers), (
        f"{code}'s docstring does not say what it deliberately stays quiet on. "
        "State the boundary: a probe that only documents when it fires cannot be "
        "reviewed for false positives."
    )
