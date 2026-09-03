"""Consistency check for the agent-loop-chaos handover pack.

Run it any time you edit the docs, schemas, or prompts:

    python3 tools/verify_pack.py

It validates the JSON Schemas and their example instances, and checks that the
names shared across docs/schemas/prompts have not drifted apart.

Dependencies are optional and the script degrades instead of dying: the cross-file
drift checks are pure stdlib and always run. `jsonschema>=4.18` + `referencing`
add instance validation; `pyyaml` adds the three checks that read
`schemas/examples/suite_demo.yaml`. Whatever is missing is listed under SKIPPED.

    python3 tools/verify_pack.py              # run whatever is possible here
    python3 tools/verify_pack.py --strict     # a skip is a failure (use this in CI)

Exit code 1 means something is inconsistent, or --strict was given and something
was skipped.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]  # optional: gates the suite_demo.yaml checks

try:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
except ModuleNotFoundError:
    Draft202012Validator = None  # type: ignore[assignment]  # optional: gates validation
    Registry = Resource = None  # type: ignore[assignment]

HAVE_YAML = yaml is not None
HAVE_JSONSCHEMA = Draft202012Validator is not None
STRICT = "--strict" in sys.argv

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "schemas"
EX = SCHEMA_DIR / "examples"

errors: list[str] = []
notes: list[str] = []
skipped: list[str] = []
unvalidated: list[str] = []

# ---------------------------------------------------------------- load schemas
resources = {}
for p in sorted(SCHEMA_DIR.glob("*.json")):
    try:
        doc = json.loads(p.read_text())
    except Exception as e:
        errors.append(f"{p.name}: invalid JSON: {e}")
        continue
    resources[p.name] = doc

registry = None
if HAVE_JSONSCHEMA:
    registry = Registry()
    for name, doc in resources.items():
        # register under both the $id and the bare filename so relative refs resolve
        res = Resource.from_contents(doc)
        registry = registry.with_resource(uri=name, resource=res)
        if "$id" in doc:
            registry = registry.with_resource(uri=doc["$id"], resource=res)

    for name, doc in resources.items():
        try:
            Draft202012Validator.check_schema(doc)
        except Exception as e:
            errors.append(f"{name}: not a valid Draft 2020-12 schema: {e}")
else:
    skipped.append(
        f"Draft 2020-12 schema check on {len(resources)} schema file(s) "
        "— needs `jsonschema>=4.18` + `referencing`"
    )


# ------------------------------------------------------- validate the examples
def validate(instance, schema_name, label):
    if not HAVE_JSONSCHEMA:
        unvalidated.append(label)
        return
    v = Draft202012Validator(resources[schema_name], registry=registry)
    found = sorted(v.iter_errors(instance), key=lambda e: list(e.absolute_path))
    for err in found:
        ptr = "/".join(str(x) for x in err.absolute_path) or "<root>"
        errors.append(f"{label}: /{ptr}: {err.message[:200]}")
    if not found:
        notes.append(f"{label}: valid against {schema_name}")


report = json.loads((EX / "report_failing.json").read_text())
validate(report, "chaos_report.schema.json", "report_failing.json")

for i, line in enumerate((EX / "trace_excerpt.jsonl").read_text().splitlines(), 1):
    if not line.strip():
        continue
    try:
        ev = json.loads(line)
    except Exception as e:
        errors.append(f"trace_excerpt.jsonl:{i}: invalid JSON: {e}")
        continue
    validate(ev, "trace_event.schema.json", f"trace_excerpt.jsonl:{i}")

if HAVE_YAML:
    suite = yaml.safe_load((EX / "suite_demo.yaml").read_text())
    validate(suite, "scenario.schema.json", "suite_demo.yaml")
else:
    suite = None
    skipped.append(
        "3 checks that read schemas/examples/suite_demo.yaml (schema validity, fault "
        "types vs the catalog, must_not codes vs the probe table) — needs `pyyaml`"
    )

# ------------------------------------------- cross-file consistency of enums
report_schema = resources["chaos_report.schema.json"]
fm_report = set(report_schema["properties"]["failure_mode"]["enum"])
judge_sys = (ROOT / "assets/prompts/judge_system.md").read_text()
m = re.search(r"## `failure_mode` enum\n\n```\n(.*?)```", judge_sys, re.S)
if not m:
    errors.append("judge_system.md: failure_mode enum block not found")
else:
    fm_prompt = {t.strip() for t in m.group(1).replace("\n", " ").split(",") if t.strip()}
    if fm_prompt != fm_report:
        errors.append(
            "failure_mode enum drift between chaos_report.schema.json and judge_system.md: "
            f"only in schema={sorted(fm_report - fm_prompt)} "
            f"only in prompt={sorted(fm_prompt - fm_report)}"
        )
    else:
        notes.append(f"failure_mode enum consistent ({len(fm_report)} values)")

_fix_prop = report_schema["properties"]["suggested_fixes"]["items"]["properties"]["kind"]
fix_kinds = set(_fix_prop["enum"])
for f in ("judge_system.md", "refiner.md"):
    txt = (ROOT / "assets/prompts" / f).read_text()
    missing = [k for k in fix_kinds if k not in txt]
    if missing:
        errors.append(f"{f}: suggested_fixes kinds missing from prompt: {missing}")
    else:
        notes.append(f"{f}: all {len(fix_kinds)} fix kinds present")

# fault kinds named in the catalog vs the API doc vs the demo suite
catalog = (ROOT / "docs/03-FAULT-CATALOG.md").read_text()
api = (ROOT / "docs/02-API.md").read_text()
catalog_faults = set(re.findall(r"### [A-C]\d+\. `(\w+Fault)`", catalog))
api_faults = set(re.findall(r"\b(\w+Fault)\b", api)) - {"Fault"}
missing_in_api = catalog_faults - api_faults
if missing_in_api:
    errors.append(f"faults in catalog but not exported in docs/02-API.md: {sorted(missing_in_api)}")
else:
    notes.append(f"all {len(catalog_faults)} catalog faults appear in docs/02-API.md")

if suite is not None:
    suite_faults = {f["type"] for s in suite["scenarios"] for f in s.get("faults", [])}
    unknown = suite_faults - catalog_faults
    if unknown:
        errors.append(f"suite_demo.yaml uses faults absent from the catalog: {sorted(unknown)}")
    else:
        notes.append(f"suite uses {len(suite_faults)} fault types, all in the catalog")

# presets referenced by the suite exist in the scenario schema enum
preset_enum = set(
    resources["scenario.schema.json"]["$defs"]["scenarioBody"]["properties"]["preset"]["oneOf"][0][
        "enum"
    ]
)
preset_section = catalog.split("## D. Fault-composition recipes")[1]
catalog_presets = set(re.findall(r"^\| `(\w+)` \|", preset_section, re.M))
if not catalog_presets <= preset_enum:
    errors.append(
        f"presets in catalog missing from scenario schema: {sorted(catalog_presets - preset_enum)}"
    )
else:
    notes.append(f"all {len(catalog_presets)} presets in the schema enum")

# probe codes: anchored to the section 3 table ONLY (the fake-agent table matches the
# same row shape, which used to inflate this count)
testing = (ROOT / "docs/07-TESTING.md").read_text()
probe_section = testing.split("## 3. Probe detection rules")[1].split("## 4.")[0]
probe_codes = set(re.findall(r"^\| `([a-z_]+)` \| ", probe_section, re.M))
if len(probe_codes) != 20:
    errors.append(f"docs/07 section 3 defines {len(probe_codes)} probes, expected 20")
else:
    notes.append("docs/07 section 3 defines exactly 20 probes")

# probes removed in the revision must not reappear
outcomes = (ROOT / "docs/11-OUTCOMES-AND-ASSERTIONS.md").read_text()
for gone in ("fabricated_value", "latency_budget_exceeded", "state_key_lost"):
    if re.search(rf"^\| `{gone}` \| ", probe_section, re.M):
        errors.append(f"removed probe {gone} is back in the docs/07 section 3 table")
notes.append(
    "removed probes (fabricated_value, latency_budget_exceeded, state_key_lost) stay removed"
)

# PROBE_PRECEDENCE in docs/11 section 8 must list exactly the same codes
m_pp = re.search(r"PROBE_PRECEDENCE`? is an explicit list.*?```\n(.*?)```", outcomes, re.S)
if not m_pp:
    errors.append("docs/11: PROBE_PRECEDENCE block not found")
else:
    pp = {t.strip() for t in m_pp.group(1).replace("\n", " ").split(",") if t.strip()}
    if pp != probe_codes:
        errors.append(
            "PROBE_PRECEDENCE vs docs/07 section 3 drift: "
            f"only in precedence={sorted(pp - probe_codes)} "
            f"only in table={sorted(probe_codes - pp)}"
        )
    else:
        notes.append(f"PROBE_PRECEDENCE matches the probe table ({len(pp)} codes)")

# the judge may only author the judge_output fields, and the prompt must ask for exactly those
jo = resources["judge_verdict.schema.json"]["$defs"]["judge_output"]["properties"]
m_out = re.search(r"## Output shape.*?```json\n(.*?)```", judge_sys, re.S)
if not m_out:
    errors.append("judge_system.md: output shape block not found")
else:
    asked = set(re.findall(r'^  "(\w+)":', m_out.group(1), re.M))
    if asked != set(jo):
        errors.append(
            "judge_system.md output block vs judge_output subschema drift: "
            f"only in prompt={sorted(asked - set(jo))} "
            f"only in schema={sorted(set(jo) - asked)}"
        )
    else:
        notes.append(f"judge prompt asks for exactly the {len(jo)} judge_output fields")
for banned in ("passed", "expected_behavior"):
    if m_out and re.search(rf'"{banned}"\s*:', m_out.group(1)):
        errors.append(f"judge_system.md asks the model for `{banned}`, which the library owns")

# the canary regex in the corpus spec must actually match the canary format
p03 = (ROOT / "prompts/03-llm-faults.md").read_text()
canary_example = "ALC-CANARY-run-3f9a12c4"
CANONICAL = r"ALC-CANARY-run-[0-9a-f]{8}"
if CANONICAL not in p03:
    errors.append(f"prompts/03: the canonical canary regex {CANONICAL!r} is not documented")
elif not re.search(CANONICAL, canary_example):
    errors.append(f"canonical canary regex does not match {canary_example!r}")
else:
    notes.append("canonical canary regex matches ALC-CANARY-run-<8 hex>")
# any regex offered as a corpus `detect` value must match too
for rx in re.findall(r'"detect":\s*\{[^}]*"value":\s*"([^"]+)"', p03):
    try:
        if not re.search(rx, canary_example):
            errors.append(f"corpus detect regex {rx!r} cannot match the canary {canary_example!r}")
    except re.error as e:
        errors.append(f"corpus detect regex {rx!r} does not compile: {e}")

# untrusted spans in the judge prompts must be fenced
ju = (ROOT / "assets/prompts/judge_user.md").read_text()
for var in (
    "{{injected}}",
    "{{final_output}}",
    "{{last_exchanges}}",
    "{{code_context}}",
    "{{tool_summary}}",
):
    idx = ju.find(var)
    if idx == -1 or "UNTRUSTED_DATA" not in ju[max(0, idx - 120) : idx]:
        errors.append(f"judge_user.md: {var} is not inside an UNTRUSTED_DATA fence (D-21)")
if all("UNTRUSTED_DATA fence" not in e for e in errors):
    notes.append("every untrusted span in judge_user.md is fenced")

# decisions log: ids unique and contiguous
dec = (ROOT / "docs/DECISIONS.md").read_text()
ids = re.findall(r"^### (D-\d+) ", dec, re.M)
nums = [int(i.split("-")[1]) for i in ids]
if len(set(ids)) != len(ids):
    errors.append("docs/DECISIONS.md has duplicate D- ids")
elif nums != list(range(1, len(nums) + 1)):
    errors.append(f"docs/DECISIONS.md ids are not contiguous from D-01: {ids[:3]}...{ids[-3:]}")
else:
    notes.append(f"docs/DECISIONS.md: {len(ids)} contiguous decisions")

# report example may only use real probe codes, and its assertions only real checks
_scenario_body = resources["scenario.schema.json"]["$defs"]["scenarioBody"]["properties"]
expect_props = set(_scenario_body["expect"]["properties"])
for a in report.get("assertions", []):
    if a["check"] not in expect_props:
        errors.append(f"report example uses assertion check not in the expect schema: {a['check']}")
if report.get("assertions"):
    notes.append(f"report example: {len(report['assertions'])} assertions, all valid checks")
if suite is not None:
    suite_must_not = {c for sc in suite["scenarios"] for c in sc.get("must_not", [])}
    suite_must_not |= set(suite.get("defaults", {}).get("must_not", []))
    bad = suite_must_not - probe_codes
    if bad:
        errors.append(f"suite must_not uses codes that are not probe codes: {sorted(bad)}")
    else:
        notes.append(f"all {len(suite_must_not)} must_not codes are real probe codes")

for sy in report["symptoms"]:
    if sy["code"] not in probe_codes:
        errors.append(f"report example uses probe code not in docs/07 table: {sy['code']}")

# ------------------------------------------- prompts reference existing files
# Only the pack itself. Without the prune, a local .venv/ or .git/ lands in here,
# which makes the scan slow and the reported file count meaningless.
_PRUNE = {
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
}
pack_files = {
    str(p.relative_to(ROOT))
    for p in ROOT.rglob("*")
    if p.is_file() and not (_PRUNE & set(p.relative_to(ROOT).parts))
}
# Paths that legitimately do not exist in the pack: created by the build itself,
# or section shorthands like `docs/04`.
BUILT_BY_AGENT = {
    "docs/FAQ.md",
    "schemas/suite.schema.json",  # written in phase 07 (D-25)
    "prompts/plan.md",
    "prompts/summarize.md",
    "prompts/respond.md",
}


def unresolvable(ref: str) -> bool:
    if ref in pack_files or ref in BUILT_BY_AGENT:
        return False
    if ref.endswith("/") or "*" in ref:
        return False
    # section shorthand, e.g. `docs/04` §3
    return not re.fullmatch(r"docs/\d\d", ref)


for prompt in sorted((ROOT / "prompts").glob("*.md")):
    txt = prompt.read_text()
    for ref in set(re.findall(r"`((?:docs|schemas|assets|prompts)/[\w./*-]+)`", txt)):
        if unresolvable(ref):
            errors.append(f"{prompt.name}: references missing pack file `{ref}`")
    # every phase prompt must have the required sections
    for section in ("## Scope", "## Acceptance checklist", "## Verify"):
        if section not in txt and prompt.name != "PROMPTING-GUIDE.md":
            errors.append(f"{prompt.name}: missing section {section}")

_scanned_docs = [
    *sorted((ROOT / "docs").glob("*.md")),
    ROOT / "README.md",
    ROOT / "PACK.md",
    ROOT / "CLAUDE.md",
]
for doc in _scanned_docs:
    txt = doc.read_text()
    _doc_ref = r"`((?:docs|schemas|assets|prompts)/[\w./-]+\.(?:md|json|yaml|jsonl))`"
    for ref in set(re.findall(_doc_ref, txt)):
        if unresolvable(ref):
            errors.append(f"{doc.name}: references missing pack file `{ref}`")

# the build pack's prompt table must list every phase prompt (D-49: moved off README.md,
# which is now the library's own readme and rendered by PyPI)
pack = (ROOT / "PACK.md").read_text()
for p in sorted((ROOT / "prompts").glob("[0-9]*.md")):
    if f"prompts/{p.name}" not in pack:
        errors.append(f"PACK.md: phase prompt not listed: {p.name}")

# ------------------------------------------------------------------- report
if unvalidated:
    skipped.append(
        f"instance validation of {len(unvalidated)} example(s) against the JSON Schemas "
        "— needs `jsonschema>=4.18` + `referencing`"
    )

print("=" * 72)
for n in notes:
    print("  ok   ", n)
if skipped:
    print("=" * 72)
    for sk in skipped:
        print("  SKIP ", sk)
    print("  ->    pip install 'jsonschema>=4.18' referencing pyyaml   # to run everything")
print("=" * 72)
if errors:
    print(f"{len(errors)} PROBLEM(S):")
    for e in errors:
        print("  FAIL ", e)
    sys.exit(1)
if skipped and STRICT:
    print(f"{len(skipped)} CHECK GROUP(S) SKIPPED and --strict was given.")
    sys.exit(1)
print("ALL AVAILABLE CHECKS PASSED" if skipped else "ALL CHECKS PASSED")
print(f"files: {len(pack_files)}")
