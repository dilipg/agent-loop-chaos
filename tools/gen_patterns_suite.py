#!/usr/bin/env python3
"""Regenerate `examples/scenarios/patterns_suite.yaml` from the pattern registry.

`examples/patterns/PATTERNS` is the source of truth for what each shape is and what
should break it. Hand-maintaining a parallel YAML file guarantees the two drift, and
the drift shows up as a scenario that proves nothing. So the YAML is generated, and
`tests/test_patterns.py` asserts the committed file still matches.

    python3 tools/gen_patterns_suite.py           # rewrite the file
    python3 tools/gen_patterns_suite.py --check   # exit 1 if it is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "examples" / "scenarios" / "patterns_suite.yaml"

HEADER = """# The conformance pool, as a runnable suite.
#
# examples/patterns/ is eight agent shapes people actually ship -- a ReAct text loop,
# an OpenAI tool-calling loop, an async agent, a class with state on `self`, a
# supervisor delegating to specialists, a fixed pipeline with no loop, a streaming
# accumulator, and a retrieve-rerank-generate chain. Each ships twice: `build` has a
# planted weakness, `build_fixed` is the hardened twin.
#
# Every `.fixed` scenario is a negative control. A probe that fires on one has a false
# positive and the probe is wrong until proven otherwise (docs/11 section 2). So a
# green run here is not "nothing broke" -- it is the harness proving it can tell a
# real weakness from its own injection, across eight unrelated loop shapes.
#
# GENERATED from examples/patterns/PATTERNS. Do not edit by hand:
#   python3 tools/gen_patterns_suite.py

version: "1.0"

defaults:
  seed: 1337

scenarios:
"""


def render() -> str:
    """Build the suite document.

    Returns:
        The YAML text.
    """
    sys.path.insert(0, str(ROOT))
    from examples.patterns import load_all

    patterns = load_all()
    blocks = [HEADER]
    for name in sorted(patterns):
        spec = patterns[name]
        for suffix, attr in (("", "build"), (".fixed", "build_fixed")):
            rows = [
                f"  - id: pattern.{name}{suffix}",
                f"    entrypoint: examples.patterns.{name}:{attr}",
                f"    description: {json.dumps(spec.description)}",
                f"    inputs: {json.dumps(spec.inputs)}",
            ]
            if spec.initial_state:
                # Flow style: YAML is a superset of JSON, and a nested structure
                # written by hand is one indentation slip from meaning something else.
                rows.append(f"    initial_state: {json.dumps(dict(spec.initial_state))}")
            rows.append(f"    expected_behavior: {spec.expected_behavior}")
            rows.append("    faults:")
            rows += [f"      - {json.dumps(dict(fault))}" for fault in spec.faults]
            blocks.append("\n".join(rows) + "\n")
    return "\n".join(blocks)


def main(argv: list[str]) -> int:
    """Write or check the generated suite.

    Args:
        argv: Command-line arguments.

    Returns:
        A process exit code.
    """
    text = render()
    if "--check" in argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
        if current == text:
            print(f"ok  {TARGET.relative_to(ROOT)} is current")
            return 0
        print(f"STALE  {TARGET.relative_to(ROOT)}; run tools/gen_patterns_suite.py")
        return 1
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
