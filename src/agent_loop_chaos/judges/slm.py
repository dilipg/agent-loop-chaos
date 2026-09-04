"""`SLMJudge` — the model path (`docs/05` §§3-5).

A small local model is good at naming a behaviour and writing a sentence someone can
act on, and bad at being the source of truth. So everything here is advisory: the
verdict's `passed`, `expected_behavior` and `observed_behavior` are carried from the
evidence, and the model is only ever asked for the `judge_output` subschema (D-20).

The other half of this module is defensive. It must never raise into the run it is
observing, whatever the endpoint returns -- a hang, an HTTP 500, prose instead of
JSON, or a schema-valid object asserting the opposite of what the probes found. Every
one of those ends in a rules verdict, not an exception.
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from ..errors import ConfigError
from ..redact import redact
from .base import JudgeEvidence, JudgeMeta, Verdict, build_prompt_values, render
from .rules import RuleJudge

__all__ = ["SLMJudge", "judge_output_schema", "normalize_base_url", "prompt_text"]

log = logging.getLogger("agent_loop_chaos")

Transport = Literal["openai", "ollama", "anthropic"]

_PROMPT_DIR = Path(__file__).parent / "prompts"
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}
_REACHABILITY_TIMEOUT_S = 1.5
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def prompt_text(name: str, prompt_dir: Path | None = None) -> str:
    """Read one prompt asset.

    Args:
        name: File name, e.g. ``"judge_system.md"``.
        prompt_dir: Override directory, from `SLMJudge(prompt_dir=…)`.

    Returns:
        The asset's text.

    Raises:
        ConfigError: When the asset is missing.
    """
    path = (prompt_dir or _PROMPT_DIR) / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"judge prompt asset not readable: {path}") from exc


def normalize_base_url(base_url: str, transport: Transport) -> str:
    """Normalize an endpoint for its transport (D-34).

    The `ollama` transport posts to `{base_url}/api/chat`, so a `base_url` ending in
    `/v1` would produce `/v1/api/chat` -- a 404 that looks like an unreachable model.
    The `openai` transport needs the `/v1` it is named for.

    Args:
        base_url: The endpoint as the caller wrote it.
        transport: Which wire protocol will be used.

    Returns:
        The normalized endpoint, without a trailing slash.
    """
    url = base_url.rstrip("/")
    if transport == "ollama":
        return url[: -len("/v1")].rstrip("/") if url.endswith("/v1") else url
    if transport == "openai" and not url.endswith("/v1"):
        return url + "/v1"
    return url


def _is_loopback(base_url: str) -> bool:
    """Report whether an endpoint stays on this machine.

    Args:
        base_url: The endpoint.

    Returns:
        True when the host is a loopback address.
    """
    host = urlparse(base_url).hostname or ""
    return host in _LOOPBACK or host.startswith("127.")


def _inline_refs(node: Any, root: Mapping[str, Any]) -> Any:
    """Resolve every `$ref` into a self-contained schema.

    A remote endpoint cannot resolve `chaos_report.schema.json#/properties/…` against
    our local registry, so the schema on the wire has to carry its own definitions.

    Args:
        node: The subschema being walked.
        root: The document `#/…` pointers resolve against.

    Returns:
        A copy with no `$ref` remaining.
    """
    if isinstance(node, list):
        return [_inline_refs(item, root) for item in node]
    if not isinstance(node, Mapping):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        from ..schema import registry

        document, _, pointer = ref.partition("#")
        target: Any = root
        if document:
            target = registry().get_or_retrieve(document).value.contents
        for part in pointer.strip("/").split("/"):
            if part:
                target = target[part]
        merged = {k: v for k, v in node.items() if k != "$ref"}
        return {**_inline_refs(target, root), **merged}
    return {k: _inline_refs(v, root) for k, v in node.items()}


@cache
def judge_output_schema() -> dict[str, Any]:
    """The exact object a model may author, with `$ref`s inlined (D-20).

    Handing the model the full verdict schema was a latent dead path: it marks
    `expected_behavior` required, the model is never asked for it, so validation
    would always fail, always repair, and always fall back to rules while looking
    implemented.

    Returns:
        A self-contained Draft 2020-12 schema for `judge_output`.
    """
    # Imported here, not at module scope: `jsonschema` is the library's one required
    # dependency and `import agent_loop_chaos` must not pull it in (docs/01 §7).
    from ..schema import load_schema

    root = load_schema("verdict")
    return cast("dict[str, Any]", _inline_refs(root["$defs"]["judge_output"], root))


@cache
def _judge_output_validator() -> Any:
    """Compile a validator for the subschema, once.

    Returns:
        A jsonschema validator over `judge_output_schema()`.
    """
    from jsonschema import Draft202012Validator

    return Draft202012Validator(judge_output_schema())


def _extract_json(text: str, pattern: re.Pattern[str]) -> Any:
    """Pull a JSON value out of whatever a model actually returned.

    Handles the three shapes small models produce in practice: bare JSON, JSON in a
    markdown fence, and JSON buried in a sentence of preamble.

    Args:
        text: The raw assistant message.
        pattern: `_JSON_OBJECT` or `_JSON_ARRAY`.

    Returns:
        The parsed value, or `None` when nothing parses.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    for candidate in (stripped, *(m.group(0) for m in [pattern.search(stripped)] if m)):
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


class SLMJudge:
    """Small-language-model judge, over an OpenAI-compatible or Ollama transport.

    Proves the weakness: nothing -- it narrates. What it must never do is decide
    `passed`, invent a file it was not shown, or crash the run it is observing.
    """

    name = "slm"

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        transport: Transport = "openai",
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout_s: float = 60.0,
        retries: int = 2,
        prompt_dir: str | Path | None = None,
        offline_fallback: bool = True,
        allow_remote_judge: bool = False,
    ) -> None:
        """Initialise.

        Args:
            model: Model identifier.
            base_url: Endpoint, normalized per transport (D-34).
            api_key: Credential, if the endpoint needs one. Never recorded.
            transport: Wire protocol.
            temperature: Sampling temperature. Anything but 0 costs reproducibility.
            max_tokens: Response cap.
            timeout_s: Per-request timeout.
            retries: Transport-error retries, beyond the first attempt.
            prompt_dir: Override the packaged prompt assets.
            offline_fallback: Fall back to `RuleJudge` rather than raising when the
                endpoint cannot be reached.
            allow_remote_judge: Consent to a non-loopback endpoint receiving the
                run's code context and payloads (D-22).

        Raises:
            ConfigError: When the endpoint is not loopback and consent was not given.
        """
        self.transport: Transport = transport
        self.base_url = normalize_base_url(base_url, transport)
        if not allow_remote_judge and (transport == "anthropic" or not _is_loopback(self.base_url)):
            raise ConfigError(
                f"judge endpoint {self.base_url!r} is not on this machine. It would receive "
                "this run's prompts, tool payloads, final output and excerpts of your source "
                "code (redacted, but redaction is best-effort). Pass "
                "allow_remote_judge=True (or --allow-remote-judge) to consent. See "
                "docs/DECISIONS.md D-22."
            )
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retries = retries
        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.offline_fallback = offline_fallback
        self.allow_remote_judge = allow_remote_judge
        self.last_call: dict[str, Any] = {}
        self._reachable: bool | None = None

    # -- transport -------------------------------------------------------------

    @property
    def _chat_url(self) -> str:
        """The chat endpoint for this transport.

        Returns:
            A full URL.
        """
        if self.transport == "ollama":
            return f"{self.base_url}/api/chat"
        if self.transport == "anthropic":
            return f"{self.base_url}/v1/messages"
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        """Build request headers.

        Returns:
            Headers including the credential when one is configured.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if self.transport == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, url: str, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        """POST JSON and return the parsed response.

        `httpx` when the `[slm]` extra is installed, `urllib` otherwise, so
        `pip install agent-loop-chaos` alone can still judge (`docs/05` §4).

        Args:
            url: The endpoint.
            payload: The request body.
            timeout_s: Per-request timeout.

        Returns:
            The parsed response body.

        Raises:
            OSError: On any transport failure, including a non-200 status. Callers
                treat every transport failure identically, so they are unified here
                rather than leaking two exception hierarchies upward.
        """
        body = json.dumps(payload).encode("utf-8")
        try:
            import httpx
        except ImportError:
            return self._post_stdlib(url, body, timeout_s)
        response_ = httpx.post(url, content=body, headers=self._headers(), timeout=timeout_s)
        if response_.status_code != 200:
            raise OSError(f"HTTP {response_.status_code} from {url}")
        try:
            return cast("dict[str, Any]", response_.json())
        except ValueError as exc:
            raise OSError(f"unparseable response from {url}: {exc}") from exc

    def _post_stdlib(self, url: str, body: bytes, timeout_s: float) -> dict[str, Any]:
        """POST without `httpx`, closing the connection whatever happens.

        `urllib.request.urlopen` abandons its connection when the read times out,
        leaking a socket per call -- a suite judged against a hung endpoint would leak
        one per scenario. `http.client` gives a handle to close in a `finally`.

        Args:
            url: The endpoint.
            body: The encoded request body.
            timeout_s: Per-request timeout.

        Returns:
            The parsed response body.

        Raises:
            OSError: On any transport failure, including a non-200 status.
        """
        parts = urlparse(url)
        factory = (
            http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        )
        connection = factory(parts.netloc, timeout=timeout_s)
        try:
            connection.request("POST", parts.path or "/", body=body, headers=self._headers())
            response = connection.getresponse()
            raw = response.read()
            if response.status != 200:
                raise OSError(f"HTTP {response.status} from {url}")
            try:
                return cast("dict[str, Any]", json.loads(raw or b"{}"))
            except ValueError as exc:
                raise OSError(f"unparseable response from {url}: {exc}") from exc
        finally:
            connection.close()

    def reachable(self) -> bool:
        """Probe the endpoint once per instance, with a short timeout.

        Drives the automatic rules-vs-ensemble choice, so it must be fast and must
        never raise.

        Returns:
            True when the endpoint answered.
        """
        if self._reachable is not None:
            return self._reachable
        path = "/api/tags" if self.transport == "ollama" else "/models"
        parts = urlparse(f"{self.base_url}{path}")
        connection: http.client.HTTPConnection | None = None
        try:
            factory = (
                http.client.HTTPSConnection
                if parts.scheme == "https"
                else http.client.HTTPConnection
            )
            connection = factory(parts.netloc, timeout=_REACHABILITY_TIMEOUT_S)
            connection.request("GET", parts.path or "/", headers=self._headers())
            connection.getresponse().read()
            self._reachable = True
        except Exception:
            self._reachable = False
        finally:
            if connection is not None:
                connection.close()
        return self._reachable

    def _envelope(self, messages: Sequence[Mapping[str, str]], schema: Any) -> dict[str, Any]:
        """Build the transport-specific request body.

        Args:
            messages: The chat messages.
            schema: A JSON schema for native structured output, or `None`.

        Returns:
            The request payload.
        """
        if self.transport == "ollama":
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": list(messages),
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            }
            if schema is not None:
                payload["format"] = schema
            return payload
        if self.transport == "anthropic":
            system = " ".join(m["content"] for m in messages if m["role"] == "system")
            return {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                    if m["role"] != "system"
                ],
            }
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "judge_output", "strict": True, "schema": schema},
            }
        return payload

    @staticmethod
    def _content(response: Mapping[str, Any]) -> str:
        """Pull the assistant text out of any of the three envelopes.

        Args:
            response: The parsed response body.

        Returns:
            The message content, or the empty string.
        """
        message = response.get("message")
        if isinstance(message, Mapping):
            return str(message.get("content") or "")
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            if isinstance(first, Mapping) and isinstance(first.get("message"), Mapping):
                return str(first["message"].get("content") or "")
        content = response.get("content")
        if isinstance(content, Sequence) and not isinstance(content, str):
            return "".join(str(b.get("text") or "") for b in content if isinstance(b, Mapping))
        return ""

    @staticmethod
    def _usage(response: Mapping[str, Any]) -> tuple[int | None, int | None]:
        """Extract token counts from any of the three envelopes.

        Args:
            response: The parsed response body.

        Returns:
            ``(tokens_in, tokens_out)``, either possibly `None`.
        """
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            return (
                usage.get("prompt_tokens") or usage.get("input_tokens"),
                usage.get("completion_tokens") or usage.get("output_tokens"),
            )
        return response.get("prompt_eval_count"), response.get("eval_count")

    def _call(
        self, messages: Sequence[Mapping[str, str]], *, schema: Any = None
    ) -> tuple[str, dict[str, Any]]:
        """One chat call, with transport-level retries.

        Args:
            messages: The chat messages.
            schema: A schema for native structured output, or `None`.

        Returns:
            ``(content, parsed_response)``.

        Raises:
            OSError: When every attempt failed.
        """
        payload = self._envelope(messages, schema)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._post(self._chat_url, payload, self.timeout_s)
                return self._content(response), response
            except Exception as exc:
                last = exc
                log.debug("judge transport attempt %d failed: %s", attempt + 1, exc)
        raise OSError(str(last))

    # -- judging ---------------------------------------------------------------

    def _messages(self, ev: JudgeEvidence) -> tuple[list[dict[str, str]], str]:
        """Render the system and user prompts for a verdict.

        Args:
            ev: The evidence, already fitted to the byte budget.

        Returns:
            ``(messages, prompt_hash)``.
        """
        system = prompt_text("judge_system.md", self.prompt_dir)
        user = render(prompt_text("judge_user.md", self.prompt_dir), ev.to_prompt_dict())
        digest = hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}], digest

    def judge(self, ev: JudgeEvidence) -> Verdict:
        """Ask the model to classify and narrate, and never let it decide `passed`.

        Args:
            ev: The evidence assembled by the engine.

        Returns:
            A verdict. On any failure this is `RuleJudge`'s verdict with
            `judge_meta.fell_back_to_rules` set -- the run is never interrupted.
        """
        fitted, dropped = ev.fit()
        messages, prompt_hash = self._messages(fitted)
        started = time.monotonic()
        attempts = 0
        raw = ""
        response: Mapping[str, Any] = {}
        errors: list[str] = []

        try:
            for attempt in range(2):  # one call, then at most one repair
                attempts += 1
                if attempt == 1:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw[:2000]},
                        {
                            "role": "user",
                            "content": (
                                "That response was rejected: "
                                + "; ".join(errors)[:600]
                                + ". Return only the corrected JSON object, no prose, no fences."
                            ),
                        },
                    ]
                raw, response = self._call(messages, schema=judge_output_schema())
                parsed = _extract_json(raw, _JSON_OBJECT)
                errors = (
                    ["response did not contain a JSON object"]
                    if not isinstance(parsed, Mapping)
                    else [e.message for e in _judge_output_validator().iter_errors(parsed)]
                )
                if not errors:
                    return self._verdict(
                        ev,
                        cast("Mapping[str, Any]", parsed),
                        prompt_hash=prompt_hash,
                        attempts=attempts,
                        started=started,
                        response=response,
                        dropped=dropped,
                        raw=raw,
                    )
        except Exception as exc:  # transport, prompt, or anything unforeseen
            log.info("judge call failed, falling back to rules: %s", exc)
            errors = [str(exc)]

        return self._fallback(
            ev,
            prompt_hash=prompt_hash,
            attempts=attempts,
            started=started,
            dropped=dropped,
            raw=raw,
            errors=errors,
        )

    def _meta(
        self,
        *,
        prompt_hash: str,
        attempts: int,
        started: float,
        response: Mapping[str, Any],
        dropped: Sequence[str],
        fell_back: bool,
    ) -> JudgeMeta:
        """Assemble `judge_meta` for either outcome.

        Args:
            prompt_hash: Hash of the rendered prompt.
            attempts: Structured-output attempts, including repairs.
            started: `time.monotonic()` at the start of the call.
            response: The last parsed response, for token counts.
            dropped: Evidence sections dropped to fit the budget.
            fell_back: Whether the rules verdict is being returned instead.

        Returns:
            The metadata block. `latency_ms` is a timing field, never read by
            decision logic.
        """
        tokens_in, tokens_out = self._usage(response)
        return JudgeMeta(
            kind="slm",
            model=self.model,
            endpoint=self.base_url,
            transport=self.transport,
            temperature=self.temperature,
            prompt_hash=prompt_hash,
            latency_ms=int((time.monotonic() - started) * 1000),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            attempts=attempts,
            fell_back_to_rules=fell_back,
            evidence_dropped=tuple(dropped),
        )

    def _verdict(
        self,
        ev: JudgeEvidence,
        parsed: Mapping[str, Any],
        *,
        prompt_hash: str,
        attempts: int,
        started: float,
        response: Mapping[str, Any],
        dropped: Sequence[str],
        raw: str,
    ) -> Verdict:
        """Build a verdict from a validated model response.

        `passed`, `expected_behavior` and `observed_behavior` come from the evidence,
        not from `parsed` -- the model does not own them, and the subschema does not
        even let it return the first two.

        Args:
            ev: The evidence.
            parsed: The validated model output.
            prompt_hash: Hash of the rendered prompt.
            attempts: How many calls it took.
            started: Start time.
            response: The raw parsed response.
            dropped: Evidence sections dropped.
            raw: The raw assistant text, for `judge.json`.

        Returns:
            The verdict.
        """
        self._record(raw, response)
        return Verdict(
            passed=ev.passed,
            expected_behavior=ev.expected_behavior,
            observed_behavior=ev.observed_behavior,
            failure_mode=parsed["failure_mode"],
            severity=parsed["severity"],
            confidence=float(parsed.get("confidence", 0.0)),
            narrative=str(parsed.get("narrative") or ""),
            root_cause_hypothesis=parsed.get("root_cause_hypothesis"),
            refinement_hint=parsed.get("refinement_hint"),
            suggested_fixes=list(parsed.get("suggested_fixes") or []),
            judge_meta=self._meta(
                prompt_hash=prompt_hash,
                attempts=attempts,
                started=started,
                response=response,
                dropped=dropped,
                fell_back=False,
            ),
        )

    def _fallback(
        self,
        ev: JudgeEvidence,
        *,
        prompt_hash: str,
        attempts: int,
        started: float,
        dropped: Sequence[str],
        raw: str,
        errors: Sequence[str],
    ) -> Verdict:
        """Return the rules verdict, keeping the raw text for `judge.json`.

        Args:
            ev: The evidence.
            prompt_hash: Hash of the rendered prompt.
            attempts: How many calls were made.
            started: Start time.
            dropped: Evidence sections dropped.
            raw: Whatever the model last said.
            errors: Why it was rejected.

        Returns:
            `RuleJudge`'s verdict with `fell_back_to_rules` set.
        """
        self._record(raw, {}, errors=list(errors))
        verdict = RuleJudge().judge(ev)
        verdict.judge_meta = self._meta(
            prompt_hash=prompt_hash,
            attempts=attempts,
            started=started,
            response={},
            dropped=dropped,
            fell_back=True,
        )
        return verdict

    def _record(self, raw: str, response: Mapping[str, Any], *, errors: Sequence[str] = ()) -> None:
        """Store the redacted request and response for `judge.json`.

        Args:
            raw: The assistant text.
            response: The parsed response envelope.
            errors: Why the response was rejected, if it was.
        """
        self.last_call = cast(
            "dict[str, Any]",
            redact(
                {
                    "request": {
                        "model": self.model,
                        "endpoint": self.base_url,
                        "transport": self.transport,
                        "temperature": self.temperature,
                    },
                    "response": {"content": raw[:8000], "envelope": copy.deepcopy(dict(response))},
                    "errors": list(errors),
                }
            ),
        )

    # -- optional side calls ---------------------------------------------------

    def narrate(self, ev: JudgeEvidence) -> str | None:
        """Ask for prose only, using the cheaper `narrator.md` prompt.

        Args:
            ev: The evidence.

        Returns:
            The narrative, or `None` when the call failed. Individually
            fallback-safe: a failure here never affects the verdict.
        """
        fitted, _dropped = ev.fit()
        try:
            content, _ = self._call(
                [
                    {
                        "role": "user",
                        "content": render(
                            prompt_text("narrator.md", self.prompt_dir), fitted.to_prompt_dict()
                        ),
                    }
                ]
            )
        except Exception as exc:
            log.info("narration failed: %s", exc)
            return None
        return content.strip() or None

    def suggest_fixes(
        self, ev: JudgeEvidence, verdict: Verdict | None = None
    ) -> list[dict[str, Any]]:
        """Ask for a ranked fix list, using `refiner.md`.

        Args:
            ev: The evidence.
            verdict: The verdict so far, for the failure classification it carries.

        Returns:
            The fixes, or an empty list on any failure.
        """
        fitted, _dropped = ev.fit()
        values = build_prompt_values(fitted, verdict)
        try:
            content, _ = self._call(
                [
                    {
                        "role": "user",
                        "content": render(prompt_text("refiner.md", self.prompt_dir), values),
                    }
                ]
            )
        except Exception as exc:
            log.info("fix suggestion failed: %s", exc)
            return []
        parsed = _extract_json(content, _JSON_ARRAY)
        if not isinstance(parsed, list):
            return []
        return [dict(f) for f in parsed if isinstance(f, Mapping) and f.get("kind")]
