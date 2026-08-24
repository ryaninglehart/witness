"""Model access, with the offline path as the default rather than the fallback.

Three providers, one interface: Anthropic, OpenAI, and a local model over
Ollama. Plus a cassette layer that replays recorded responses, which is what
makes the test suite hermetic and the CI run free.

The ordering matters and is a design position, not a convenience. **Ollama is
tried first.** For a clinical tool the local path is the one with the strongest
privacy story -- encounter text never leaves the machine, so no business
associate agreement is needed with anyone -- and a harness whose default
requires shipping patient conversations to a third party has chosen the wrong
default no matter how good the BAA is.

DeepScribe's own published architecture reasons the same way: they hold BAAs
with OpenAI and Anthropic *and* fine-tune 7B models in-house for the hot path
(SpecialtyScribe, ACL 2025). A legal control and an architectural control are
not substitutes.

Nothing here requires an API key to run. If no provider is reachable the
harness says so and continues with the lexical grader rather than failing --
the same reasoning that keeps the corpus loader working without audio.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CASSETTE_DIR = Path(__file__).resolve().parents[2] / "cassettes"


class ProviderError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    #: True when this came from a recorded cassette rather than a live call.
    #: Surfaced all the way into the report, because a number produced from a
    #: replay and a number produced from a live model are different claims.
    replayed: bool = False


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion: ...

    def available(self) -> bool: ...

    def request_fingerprint(self) -> str:
        """Request settings, other than the prompt, that change the response."""
        ...


def _cassette_key(
    provider: str, model: str, system: str, user: str, fingerprint: str = ""
) -> str:
    """Identify a recorded response by the whole request, not just the prompt.

    ``fingerprint`` carries the request parameters that change the answer --
    for Ollama, whether output was constrained to JSON. Without it the key was
    the prompt alone, so turning constrained decoding *on* would have replayed
    the unconstrained responses recorded before it and reported that the
    change made no difference. A cache that ignores an input is not a cache,
    it is a way to measure the wrong thing quickly. See WIT-020.
    """
    parts = f"{provider}\x00{model}\x00{system}\x00{user}\x00{fingerprint}"
    return hashlib.sha256(parts.encode()).hexdigest()[:32]


class Cassette:
    """Record and replay model responses.

    An evaluation harness that cannot reproduce its own numbers without a
    network and a credit card is not much of a harness. Recording every call
    means the published results can be re-derived exactly by anyone who clones
    the repository, and it makes the test suite deterministic.
    """

    def __init__(self, directory: Path | None = None, record: bool = True):
        self.dir = Path(directory or CASSETTE_DIR)
        self.record = record

    def get(self, key: str) -> Completion | None:
        path = self.dir / f"{key}.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return Completion(d["text"], d["provider"], d["model"], replayed=True)

    def put(self, key: str, c: Completion) -> None:
        if not self.record:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{key}.json").write_text(
            json.dumps({"text": c.text, "provider": c.provider, "model": c.model}, indent=2)
        )


class OllamaProvider:
    """A local model. No key, no network egress, no BAA."""

    name = "ollama"

    #: Any instruct-tuned model Ollama can serve. Overridable with
    #: ``OLLAMA_MODEL`` so a reviewer can point the harness at whatever they
    #: already have pulled instead of downloading another several gigabytes.
    DEFAULT_MODEL = "qwen2.5:7b-instruct"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        constrain_json: bool | None = None,
        schema: dict | None = None,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL") or self.DEFAULT_MODEL
        #: Ask the server to constrain decoding to valid JSON. Left on by
        #: default because the alternative is measured: unconstrained, this
        #: model abandoned the requested format on 29% of batches and wrote
        #: prose instead, and every claim in those batches was refused.
        #: Set WITNESS_CONSTRAIN_JSON=0 to reproduce the unconstrained runs.
        if constrain_json is None:
            constrain_json = os.environ.get("WITNESS_CONSTRAIN_JSON", "1") != "0"
        self.constrain_json = constrain_json
        #: A JSON schema to constrain the *shape* of the output, not just its
        #: syntax. Passing ``"json"`` alone was measured and made things worse:
        #: the model stopped writing prose and started writing valid JSON that
        #: contained no answer. A schema removes that option too. See WIT-021.
        self.schema = schema
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    def available(self) -> bool:
        """Reachable *and* serving the requested model.

        Checking only that the daemon answers reports a provider that then
        fails on every call, which the semantic grader turns into a note full
        of UNSUPPORTED verdicts -- a plausible-looking result produced by a
        missing model rather than by a bad note.
        """
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                if r.status != 200:
                    return False
                tags = json.loads(r.read()).get("models", [])
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
            return False
        names = {m.get("name", "") for m in tags}
        return any(n == self.model or n.split(":")[0] == self.model.split(":")[0]
                   for n in names)

    def installed_models(self) -> list[str]:
        """What this daemon can actually serve. Printed when the match fails."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                return sorted(m.get("name", "") for m in json.loads(r.read()).get("models", []))
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
            return []

    def request_fingerprint(self) -> str:
        if self.schema is not None:
            # Distinct from plain "json": a run under a schema answers a
            # different question than one merely constrained to valid JSON,
            # and must not replay its recordings.
            digest = hashlib.sha256(
                json.dumps(self.schema, sort_keys=True).encode()
            ).hexdigest()[:12]
            return f"schema:{digest}"
        return "json" if self.constrain_json else "free"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        body: dict[str, object] = {
                "model": self.model,
                "system": system,
                "prompt": user,
                "stream": False,
                # Grading is a classification task. Sampling temperature above
                # zero makes the same note score differently on reruns, which
                # would make a release gate non-deterministic.
                "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        if self.schema is not None:
            body["format"] = self.schema
        elif self.constrain_json:
            # Server-side grammar-constrained decoding. The model can only
            # emit syntactically valid JSON, which removes the failure that
            # cost 29% of batches: abandoning the format and narrating.
            # It constrains syntax, not shape -- the tolerant parser and the
            # retry below still carry the rest.
            body["format"] = "json"
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return Completion(json.loads(r.read())["response"], self.name, self.model)
        except (urllib.error.URLError, OSError, TimeoutError, KeyError) as exc:
            raise ProviderError(f"ollama: {exc}") from exc


class AnthropicProvider:
    name = "anthropic"

    def request_fingerprint(self) -> str:
        return "free"

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        if not self.api_key:
            raise ProviderError("anthropic: ANTHROPIC_API_KEY is not set")
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            return Completion(d["content"][0]["text"], self.name, self.model)
        except (urllib.error.URLError, OSError, TimeoutError, KeyError, IndexError) as exc:
            raise ProviderError(f"anthropic: {exc}") from exc


class OpenAIProvider:
    name = "openai"

    def request_fingerprint(self) -> str:
        return "free"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        if not self.api_key:
            raise ProviderError("openai: OPENAI_API_KEY is not set")
        body = json.dumps(
            {
                "model": self.model,
                "max_completion_tokens": max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            return Completion(d["choices"][0]["message"]["content"], self.name, self.model)
        except (urllib.error.URLError, OSError, TimeoutError, KeyError, IndexError) as exc:
            raise ProviderError(f"openai: {exc}") from exc


class CachedProvider:
    """Wraps a provider with cassette replay."""

    def __init__(self, inner: Provider, cassette: Cassette | None = None):
        self.inner = inner
        self.cassette = cassette or Cassette()
        self.name = inner.name
        self.model = inner.model

    def available(self) -> bool:
        return True  # replay works with no live provider at all

    def request_fingerprint(self) -> str:
        return self.inner.request_fingerprint()

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        key = _cassette_key(
            self.inner.name,
            self.inner.model,
            system,
            user,
            self.inner.request_fingerprint(),
        )
        hit = self.cassette.get(key)
        if hit is not None:
            return hit
        if not self.inner.available():
            raise ProviderError(
                f"{self.inner.name} is unavailable and no cassette exists for this "
                "prompt. Run once with the provider reachable to record it."
            )
        out = self.inner.complete(system, user, max_tokens)
        self.cassette.put(key, out)
        return out


def default_provider(prefer: str | None = None) -> Provider | None:
    """Pick a provider, local first.

    Returns ``None`` rather than raising when nothing is reachable, so callers
    can fall back to the lexical grader and say so, instead of the whole run
    dying because a laptop is offline.
    """
    from witness.graders.semantic import VERDICT_SCHEMA

    candidates: list[Provider] = [
        OllamaProvider(schema=VERDICT_SCHEMA),
        AnthropicProvider(),
        OpenAIProvider(),
    ]
    if prefer:
        candidates.sort(key=lambda p: p.name != prefer)
    for p in candidates:
        if p.available():
            return CachedProvider(p)
    # Replay-only: if cassettes exist the harness still reproduces its numbers.
    if CASSETTE_DIR.exists() and any(CASSETTE_DIR.glob("*.json")):
        return CachedProvider(OllamaProvider(schema=VERDICT_SCHEMA))
    return None
