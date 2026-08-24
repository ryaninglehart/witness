"""The semantic grader, exercised against a scripted provider.

This grader had never been run when these tests were written. It could not be:
the machine it was built on had no model server, no API key and no route to
pretrained weights. Shipping it unmeasured and describing it as the fix would
have been the kind of claim this project exists to object to.

A scripted provider does not measure accuracy and nothing here pretends to.
What it does measure is every path where the grader must *refuse* a model's
answer -- a cited utterance that does not exist, a SUPPORTED verdict with no
citation, unparseable output, a dead provider. Those are the paths that decide
whether a wrong answer from the model becomes a wrong answer from the harness,
and they are exactly the paths a live run exercises least, because a working
model mostly behaves.
"""

from __future__ import annotations

import json

import pytest

from witness.corpus import Utterance
from witness.graders.semantic import SemanticGrader
from witness.grounding import Assertion, Slot, Verdict
from witness.providers import Completion, ProviderError

UTTERANCES = [
    Utterance(speaker="doctor", text="What brings you in?", start_s=0.0, end_s=2.0),
    Utterance(
        speaker="patient",
        text="Loose watery stool about six times a day for three days",
        start_s=2.0,
        end_s=8.0,
    ),
    Utterance(speaker="doctor", text="Any blood in it?", start_s=8.0, end_s=10.0),
    Utterance(speaker="patient", text="No, no blood at all", start_s=10.0, end_s=12.0),
]


class ScriptedProvider:
    """Returns whatever it is told to, and records what it was asked."""

    name = "scripted"
    model = "test"

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        self.calls.append(user)
        text = self.responses.pop(0) if self.responses else "{}"
        if isinstance(text, Exception):  # pragma: no cover - defensive
            raise text
        return Completion(text, self.name, self.model)


class FailingProvider:
    name = "failing"
    model = "test"

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        raise ProviderError("connection refused")


def _verdicts(*items: dict) -> str:
    return json.dumps({"verdicts": list(items)})


def _assertion(i: int, text: str, negated: bool = False) -> Assertion:
    return Assertion(index=i, text=text, slot=Slot.OTHER, negated=negated)


def _ground(provider, assertions):
    return SemanticGrader(provider).ground_note("c1", "m1", assertions, UTTERANCES)


class TestHappyPath:
    def test_a_cited_verdict_carries_its_span(self):
        g = _ground(
            ScriptedProvider(_verdicts({"i": 0, "v": "SUPPORTED", "u": 1, "why": "stated"})),
            [_assertion(0, "diarrhoea for three days")],
        )
        a = g.assertions[0]
        assert a.verdict is Verdict.SUPPORTED
        assert [s.utterance_index for s in a.spans] == [1]
        assert a.spans[0].text.startswith("Loose watery stool")

    def test_inferred_is_preserved_rather_than_flattened(self):
        # The whole reason this is not an entailment model. Collapsing INFERRED
        # into SUPPORTED or UNSUPPORTED loses the distinction the grader exists
        # to make.
        g = _ground(
            ScriptedProvider(
                _verdicts({"i": 0, "v": "INFERRED", "u": 1, "why": "six loose stools -> diarrhoea"})
            ),
            [_assertion(0, "diarrhoea")],
        )
        assert g.assertions[0].verdict is Verdict.INFERRED
        assert g.assertions[0].spans

    def test_transcript_is_numbered_for_the_model(self):
        p = ScriptedProvider(_verdicts({"i": 0, "v": "UNSUPPORTED", "u": None}))
        _ground(p, [_assertion(0, "x")])
        assert "[1] patient: Loose watery stool" in p.calls[0]


class TestRefusals:
    """Every path where the grader must not take the model's word for it."""

    def test_a_fabricated_utterance_number_is_refused(self):
        # Worse than no citation, because it looks like evidence.
        g = _ground(
            ScriptedProvider(_verdicts({"i": 0, "v": "SUPPORTED", "u": 99, "why": "said so"})),
            [_assertion(0, "chest pain")],
        )
        a = g.assertions[0]
        assert a.verdict is Verdict.UNSUPPORTED
        assert "does not exist" in a.rationale
        assert not a.spans

    def test_negative_index_is_refused(self):
        g = _ground(
            ScriptedProvider(_verdicts({"i": 0, "v": "SUPPORTED", "u": -1})),
            [_assertion(0, "chest pain")],
        )
        assert g.assertions[0].verdict is Verdict.UNSUPPORTED

    def test_supported_without_a_citation_is_refused(self):
        g = _ground(
            ScriptedProvider(_verdicts({"i": 0, "v": "SUPPORTED", "u": None, "why": "obvious"})),
            [_assertion(0, "chest pain")],
        )
        a = g.assertions[0]
        assert a.verdict is Verdict.UNSUPPORTED
        assert "without citing" in a.rationale

    def test_unrecognised_verdict_is_refused(self):
        g = _ground(
            ScriptedProvider(_verdicts({"i": 0, "v": "PROBABLY_FINE", "u": 1})),
            [_assertion(0, "chest pain")],
        )
        assert g.assertions[0].verdict is Verdict.UNSUPPORTED
        assert "unrecognised" in g.assertions[0].rationale

    def test_unparseable_output_does_not_become_a_clean_bill_of_health(self):
        g = _ground(
            ScriptedProvider("I'm sorry, I can't help with clinical documents."),
            [_assertion(0, "chest pain"), _assertion(1, "nkda")],
        )
        assert all(a.verdict is Verdict.UNSUPPORTED for a in g.assertions)

    def test_a_dead_provider_blocks_rather_than_passes(self):
        # The dangerous failure: a grader that cannot reach its model returning
        # nothing to report, and a gate reading that as a clean note.
        g = _ground(FailingProvider(), [_assertion(0, "chest pain")])
        a = g.assertions[0]
        assert a.verdict is Verdict.UNSUPPORTED
        assert "grader unavailable" in a.rationale
        assert a.is_failure

    def test_a_claim_the_model_skipped_is_not_silently_dropped(self):
        # Two claims in, one verdict back. The missing one must appear in the
        # output as unresolved, or the note gets graded on a subset of itself.
        g = _ground(
            ScriptedProvider(_verdicts({"i": 0, "v": "SUPPORTED", "u": 1})),
            [_assertion(0, "diarrhoea"), _assertion(1, "no blood")],
        )
        assert len(g.assertions) == 2
        assert g.assertions[1].verdict is Verdict.UNSUPPORTED
        assert "no verdict" in g.assertions[1].rationale


class TestBatching:
    def test_every_claim_is_graded_across_batches(self):
        claims = [_assertion(i, f"claim {i}") for i in range(30)]
        responses = [
            _verdicts(*[{"i": i, "v": "SUPPORTED", "u": 1} for i in range(s, min(s + 12, 30))])
            for s in range(0, 30, 12)
        ]
        p = ScriptedProvider(*responses)
        g = _ground(p, claims)
        assert len(p.calls) == 3
        assert len(g.assertions) == 30
        assert [a.assertion.index for a in g.assertions] == list(range(30))
        assert all(a.verdict is Verdict.SUPPORTED for a in g.assertions)

    def test_one_failed_batch_does_not_discard_the_others(self):
        claims = [_assertion(i, f"claim {i}") for i in range(24)]
        p = ScriptedProvider(
            _verdicts(*[{"i": i, "v": "SUPPORTED", "u": 1} for i in range(12)]),
            "garbage",
        )
        g = _ground(p, claims)
        assert len(g.assertions) == 24
        assert all(a.verdict is Verdict.SUPPORTED for a in g.assertions[:12])
        assert all(a.verdict is Verdict.UNSUPPORTED for a in g.assertions[12:])

    def test_an_empty_note_makes_no_calls(self):
        p = ScriptedProvider()
        g = _ground(p, [])
        assert p.calls == []
        assert g.assertions == []


class TestReplayIsDisclosed:
    def test_a_replayed_verdict_says_so(self):
        # A number produced from a cassette and a number produced from a live
        # model are different claims, and the report must not conflate them.
        class Replaying(ScriptedProvider):
            def complete(self, system, user, max_tokens=1024):
                self.calls.append(user)
                return Completion(
                    _verdicts({"i": 0, "v": "SUPPORTED", "u": 1, "why": "stated"}),
                    self.name,
                    self.model,
                    replayed=True,
                )

        g = _ground(Replaying(), [_assertion(0, "diarrhoea")])
        assert "(replayed)" in g.assertions[0].rationale


class TestOllamaAvailability:
    def test_a_missing_model_reports_unavailable(self, monkeypatch):
        # Reporting "available" when the daemon is up but the model is absent
        # produces a note full of UNSUPPORTED verdicts that looks like a
        # measurement and is a missing download.
        import witness.providers as pv

        class FakeResponse:
            status = 200

            def read(self):
                return json.dumps({"models": [{"name": "llama3.1:8b"}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(pv.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
        assert not pv.OllamaProvider(model="qwen2.5:7b-instruct").available()
        assert pv.OllamaProvider(model="llama3.1:8b").available()
        # A bare family name matches the tag the daemon actually serves.
        assert pv.OllamaProvider(model="llama3.1").available()


@pytest.mark.parametrize(
    "payload",
    [
        '{"verdicts": []}',
        "{}",
        "",
        "null",
        # This one crashed the grader the first time these tests were run:
        # .get("verdicts", []) returns None when the key is present and null.
        '{"verdicts": null}',
        '{"verdicts": "SUPPORTED"}',
        '{"verdicts": [null, 3, "x"]}',
        'Sure! Here is the JSON: {"verdicts": [{"i": 0, "v": "SUPP',
        '```json\n{"verdicts": []}\n```',
    ],
)
def test_degenerate_payloads_do_not_raise(payload):
    g = _ground(ScriptedProvider(payload), [_assertion(0, "chest pain")])
    assert g.assertions[0].verdict is Verdict.UNSUPPORTED


# -- shapes a real model actually emits ------------------------------------
#
# WIT-016. Everything above was written against a scripted provider, and the
# scripts were written by the same person as the parser, so they all used the
# envelope the prompt asks for. Against a live qwen2.5:7b that envelope arrived
# 5 times in 19. These are the other fourteen.

LIVE_SHAPES = {
    "bare array": '[{"i":12,"v":"SUPPORTED","u":1,"why":"a"},{"i":13,"v":"UNSUPPORTED","u":null}]',
    "prose then array": (
        "Based on the dialogue, the claims can be summarized as follows:\n"
        '[{"i":0,"v":"INFERRED","u":1,"why":"b"}]'
    ),
    "concatenated objects": '{"i":0,"v":"SUPPORTED","u":1},\n{"i":1,"v":"UNSUPPORTED","u":null}',
    "single object": '{"i":0,"v":"CONTRADICTED","u":1,"why":"c"}',
    "code fence": '```json\n[{"i":0,"v":"SUPPORTED","u":1}]\n```',
    "prose then envelope": (
        'Here you go!\n{"verdicts":[{"i":0,"v":"SUPPORTED","u":1}]}\nHope that helps.'
    ),
    "nested under another key": '{"result":{"verdicts":[{"i":0,"v":"SUPPORTED","u":1}]}}',
    "truncated after one": '[{"i":0,"v":"SUPPORTED","u":1},{"i":1,"v":"SUPP',
}


@pytest.mark.parametrize("shape", sorted(LIVE_SHAPES), ids=lambda s: s.replace(" ", "-"))
def test_shapes_a_real_model_emits_are_read(shape):
    verdicts = SemanticGrader._parse(LIVE_SHAPES[shape])
    assert verdicts, f"{shape!r} yielded nothing; every claim in it would be refused"
    assert all("i" in v and "v" in v for v in verdicts)


def test_a_verdict_is_not_just_any_dict():
    # The opposite failure. A parser loose enough to accept the model's own
    # commentary would reintroduce WIT-016 pointing the other way.
    assert SemanticGrader._parse('{"note":"these all look fine to me"}') == []
    assert SemanticGrader._parse('{"a":{"b":{"c":[1,2,3]}}}') == []
    assert SemanticGrader._parse('{"verdicts": [null, 3, "x"]}') == []


def test_a_restated_answer_votes_once():
    twice = (
        '{"verdicts":[{"i":0,"v":"SUPPORTED","u":1}]}\n'
        'In summary: [{"i":0,"v":"UNSUPPORTED","u":null}]'
    )
    assert [v["v"] for v in SemanticGrader._parse(twice)] == ["SUPPORTED"]


def test_unreadable_output_is_counted_not_just_absorbed():
    # The heart of WIT-016 was not the parse failure, it was that nobody could
    # see it. Blocking quietly is indistinguishable from a note that deserved
    # to be blocked.
    provider = ScriptedProvider(
        "no json here at all", _verdicts({"i": 1, "v": "SUPPORTED", "u": 1})
    )
    g = SemanticGrader(provider)
    g.ground_note("c1", "m", [_assertion(0, "x")], UTTERANCES)
    g.ground_note("c1", "m", [_assertion(1, "y")], UTTERANCES)
    assert g.batches == 2
    assert g.unparsed_batches == 1
    assert g.unparsed_rate == 0.5


# -- constrained decoding and retries (WIT-020) ----------------------------


class _Recording(ScriptedProvider):
    name = "ollama"
    model = "qwen2.5:7b-instruct"

    def request_fingerprint(self) -> str:
        return "json"


def test_a_retry_rescues_an_unreadable_first_attempt():
    provider = _Recording("Based on the dialogue, the claims are as follows...",
                          _verdicts({"i": 0, "v": "SUPPORTED", "u": 1}))
    g = SemanticGrader(provider)
    out = g.ground_note("c1", "m", [_assertion(0, "cough")], UTTERANCES)
    assert out.assertions[0].verdict is Verdict.SUPPORTED
    assert (g.retried_batches, g.rescued_batches, g.unparsed_batches) == (1, 1, 0)


def test_the_retry_says_something_different():
    # Restating the same format spec politely is not a retry strategy; the
    # first attempt already carried it and the model ignored it.
    provider = _Recording("prose", _verdicts({"i": 0, "v": "SUPPORTED", "u": 1}))
    SemanticGrader(provider).ground_note("c1", "m", [_assertion(0, "x")], UTTERANCES)
    assert len(provider.calls) == 2
    assert provider.calls[1] != provider.calls[0]
    assert "Begin with [" in provider.calls[1]


def test_a_hopeless_batch_is_still_counted_as_unreadable():
    # A retry that never succeeds is a slower failure, and must not be
    # smoothed out of the reported rate.
    provider = _Recording("prose", "more prose")
    g = SemanticGrader(provider)
    g.ground_note("c1", "m", [_assertion(0, "x")], UTTERANCES)
    assert g.unparsed_batches == 1
    assert g.retry_rate == 1.0
    assert g.rescued_batches == 0


def test_retries_can_be_disabled():
    provider = _Recording("prose", _verdicts({"i": 0, "v": "SUPPORTED", "u": 1}))
    g = SemanticGrader(provider, retries=0)
    g.ground_note("c1", "m", [_assertion(0, "x")], UTTERANCES)
    assert len(provider.calls) == 1
    assert g.unparsed_batches == 1


def test_the_cassette_key_covers_the_request_not_just_the_prompt():
    """WIT-020. Turning constrained decoding on must not replay the
    unconstrained recordings and report that nothing changed."""
    from witness.providers import OllamaProvider, _cassette_key

    on, off = OllamaProvider(), OllamaProvider(constrain_json=False)
    assert on.request_fingerprint() != off.request_fingerprint()
    same_prompt = ("ollama", "m", "sys", "user")
    assert _cassette_key(*same_prompt, on.request_fingerprint()) != _cassette_key(
        *same_prompt, off.request_fingerprint()
    )


def test_constrained_decoding_is_on_by_default_and_can_be_turned_off():
    from witness.providers import OllamaProvider

    assert OllamaProvider().constrain_json is True
    assert OllamaProvider(constrain_json=False).constrain_json is False


def test_the_schema_pins_the_shape_not_just_the_syntax():
    """WIT-021. Constraining to valid JSON alone made things worse: the model
    stopped writing prose and started writing valid documents containing no
    answer. The schema removes that option."""
    from witness.graders.semantic import VERDICT_SCHEMA

    props = VERDICT_SCHEMA["properties"]["verdicts"]["items"]["properties"]
    assert set(VERDICT_SCHEMA["required"]) == {"verdicts"}
    assert set(VERDICT_SCHEMA["properties"]["verdicts"]["items"]["required"]) == {"i", "v"}
    assert set(props["v"]["enum"]) == {
        "SUPPORTED", "INFERRED", "UNSUPPORTED", "CONTRADICTED",
    }
    # The shapes the model actually produced under syntax-only constraint must
    # not satisfy this schema.
    assert "claims" not in VERDICT_SCHEMA["properties"]


def test_a_schema_run_does_not_replay_a_syntax_only_run():
    from witness.graders.semantic import VERDICT_SCHEMA
    from witness.providers import OllamaProvider, _cassette_key

    plain, schema = OllamaProvider(), OllamaProvider(schema=VERDICT_SCHEMA)
    prompt = ("ollama", "m", "sys", "user")
    assert _cassette_key(*prompt, plain.request_fingerprint()) != _cassette_key(
        *prompt, schema.request_fingerprint()
    )


def test_the_shapes_that_defeated_syntax_only_constraint_still_yield_nothing():
    # Recorded verbatim from the live constrained run.
    for junk in ('{"claims":[0,1,2,3,4,5,6,7,8,9,10,11]}', "{  }", '{"claims":[12,13]}'):
        assert SemanticGrader._parse(junk) == [], junk
