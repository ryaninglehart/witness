"""A grader that learns the shorthand instead of being told it.

The lexical grader carries a hand-written synonym table: ``abdo`` means
``tummy``, ``diarrhoea`` means ``loose stools``. Every entry in it was added
because a real case failed, which is a slow and biased way to build a lexicon
and it will never be finished. It is also the reason that grader cannot ground
60% of a clinician's note -- the table stops where its author's patience did.

The mapping does not have to be written by hand, because the corpus contains
it. PriMock57 pairs 57 transcripts with the notes the treating clinicians wrote
from them, so a note term and the transcript language it was derived from
co-occur far more often than chance. ``inhaler`` in a note sits beside
*"my blue one, the salbutamol"* in the transcript. That is a parallel corpus,
and the shorthand is learnable from it.

The method is deliberately old and transparent: positive pointwise mutual
information over a term co-occurrence matrix, reduced with a truncated SVD. It
is latent semantic analysis, it predates every model this project is nominally
about, and it needs no pretrained weights, no network, no GPU and no API key.

Three corrections separate this from the naive version, and each was made
because the naive version produced a specific wrong answer (see WIT-010).

**Two views, not one.** Cross-view counts pair note terms with transcript terms
-- the parallel-corpus signal, and the only place ``inhaler``/``salbutamol``
comes from. Local counts pair terms inside a sliding window of utterances,
which supplies the thousands of observations that 57 documents cannot.

**A document-frequency ceiling.** A term appearing in most consultations
discriminates nothing. Left in, ``yeah`` and ``uh`` acquire high similarity to
everything, because in a 57-document matrix they genuinely do co-occur with
everything, and the grader then supports any claim that shares a filler word.

**Context distribution smoothing.** Raw PPMI is biased toward rare terms: a
term seen twice gets an enormous PMI with whatever it happened to appear
beside. Raising the context marginal to the 0.75 power is the standard
correction (Levy, Goldberg & Dagan, TACL 2015) and it is what stops ``hx``
from being a near-synonym of ``bit``.

**It is also the module most at risk of lying to itself.** If the associations
are learned from the same consultations they are then evaluated on, the grader
has seen the answers. Every number this module produces comes from
:mod:`witness.crossval`, which fits on training folds and scores held-out ones.
Reporting an in-sample figure here would be indistinguishable from a real
result and much easier to get.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from witness.corpus import Consultation, Utterance
from witness.graders.lexical import content_terms
from witness.grounding import (
    Assertion,
    GroundedAssertion,
    NoteGrounding,
    Span,
    Verdict,
)
from witness.segment import is_negated

GRADER_NAME = "statistical/1.0"


@dataclass
class AssociationModel:
    """Learned term associations. Fit on transcripts paired with their notes."""

    vocab: dict[str, int] = field(default_factory=dict)
    vectors: np.ndarray | None = None
    #: Consultations the model was fitted on. Carried so that an evaluation can
    #: assert it is not scoring a consultation it trained on -- the check is
    #: cheap and the mistake is invisible.
    fitted_on: tuple[str, ...] = ()
    #: Cosine floor used when the neighbour table was built. Stored on the
    #: model so a grader cannot silently ask for a looser floor than the table
    #: it is reading was built for.
    floor: float = 0.45
    _neighbours: dict[str, dict[str, float]] | None = field(
        default=None, repr=False, compare=False
    )

    def similarity(self, a: str, b: str) -> float:
        if self.vectors is None:
            return 0.0
        ia, ib = self.vocab.get(a), self.vocab.get(b)
        if ia is None or ib is None:
            return 0.0
        va, vb = self.vectors[ia], self.vectors[ib]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def neighbours(self) -> dict[str, dict[str, float]]:
        """Every above-floor association, as a lookup table.

        Built once. Scoring 1,140 notes term-by-term against the vector matrix
        is tens of millions of dot products; the table is one dense matrix
        multiply and turns the grader into set intersections. The table is also
        the artifact a clinician can actually review -- a learned lexicon
        nobody can read is a black box with better marketing.
        """
        if self._neighbours is not None:
            return self._neighbours
        table: dict[str, dict[str, float]] = {}
        if self.vectors is not None and len(self.vocab):
            v = self.vectors
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            unit = v / norms
            sims = unit @ unit.T
            np.fill_diagonal(sims, 0.0)
            inv = [t for t, _ in sorted(self.vocab.items(), key=lambda kv: kv[1])]
            rows, cols = np.nonzero(sims >= self.floor)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                table.setdefault(inv[r], {})[inv[c]] = float(sims[r, c])
        self._neighbours = table
        return table

    def nearest(self, term: str, k: int = 5) -> list[tuple[str, float]]:
        """Inspection helper."""
        ns = self.neighbours().get(term, {})
        return sorted(ns.items(), key=lambda kv: -kv[1])[:k]


def _windows(terms: list[set[str]], width: int) -> list[set[str]]:
    """Sliding union of adjacent utterance term sets.

    A finding and the words describing it are rarely in one utterance -- the
    doctor asks, the patient answers, the doctor summarises back. A window of
    utterances is the unit over which that exchange is one topic.
    """
    out: list[set[str]] = []
    for i in range(len(terms)):
        w: set[str] = set()
        for j in range(i, min(i + width, len(terms))):
            w |= terms[j]
        if len(w) > 1:
            out.append(w)
    return out


def fit(
    consultations: list[Consultation],
    dims: int = 64,
    min_count: int = 2,
    max_doc_frac: float = 0.5,
    window: int = 3,
    smoothing: float = 0.75,
    floor: float = 0.45,
) -> AssociationModel:
    """Learn note-shorthand to spoken-language associations.

    Args:
        dims: SVD rank. Below the vocabulary size or the reduction does nothing.
        min_count: A term must appear in this many consultations to be learned.
        max_doc_frac: A term appearing in more than this share of consultations
            is dropped. It is filler by definition -- it discriminates nothing,
            and left in it becomes a near-synonym of every other term.
        window: Width, in utterances, of the local co-occurrence window.
        smoothing: Context-marginal exponent. 1.0 is textbook PMI and is biased
            toward rare terms; 0.75 is the standard correction.
        floor: Cosine below which two terms are recorded as unrelated.
    """
    note_terms: list[set[str]] = []
    utt_terms: list[list[set[str]]] = []
    doc_freq: dict[str, int] = defaultdict(int)
    for c in consultations:
        nt = content_terms(c.clinician_note)
        ut = [content_terms(u.text) for u in c.utterances]
        note_terms.append(nt)
        utt_terms.append(ut)
        seen = set(nt)
        for s in ut:
            seen |= s
        for t in seen:
            doc_freq[t] += 1

    n_docs = max(1, len(consultations))
    ceiling = max_doc_frac * n_docs
    keep = sorted(t for t, n in doc_freq.items() if min_count <= n <= ceiling)
    fitted = tuple(c.consultation_id for c in consultations)
    if len(keep) < dims + 1:
        # Not enough vocabulary to reduce into. An empty model degrades the
        # grader to exact matching rather than producing noise shaped like
        # signal, which is the failure mode that matters here.
        return AssociationModel(fitted_on=fitted, floor=floor)

    vocab = {t: i for i, t in enumerate(keep)}
    n = len(vocab)
    co = np.zeros((n, n), dtype=np.float32)

    def add(a: set[str], b: set[str], weight: float) -> None:
        ia = [vocab[t] for t in a if t in vocab]
        ib = [vocab[t] for t in b if t in vocab]
        for i in ia:
            for j in ib:
                if i != j:
                    co[i, j] += weight
                    co[j, i] += weight

    for nt, ut in zip(note_terms, utt_terms, strict=True):
        # Cross-view: the parallel-corpus signal. Weighted above local counts
        # because it is the only evidence that a note term and a spoken phrase
        # describe the same fact rather than merely the same consultation.
        spoken: set[str] = set()
        for s in ut:
            spoken |= s
        add(nt, spoken, 2.0)
        for w in _windows(ut, window):
            add(w, w, 0.5)

    total = co.sum()
    if total == 0:
        return AssociationModel(fitted_on=fitted, floor=floor)

    row = co.sum(axis=1, keepdims=True)
    col = np.power(co.sum(axis=0, keepdims=True), smoothing)
    col_total = col.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co / total) / ((row / total) * (col / col_total)))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)

    u, sv, _ = np.linalg.svd(ppmi, full_matrices=False)
    vectors = u[:, :dims] * sv[:dims]
    return AssociationModel(
        vocab=vocab,
        vectors=vectors.astype(np.float32),
        fitted_on=fitted,
        floor=floor,
    )


class _AssocIndex:
    """Inverted index over utterance terms, widened by learned associations.

    A claim term reaches an utterance either because the utterance contains it
    or because it contains an associated term. Postings are built once per
    consultation so scoring is over candidates rather than over every turn --
    the same reason the lexical grader has an index, and the difference between
    a CI gate and a nightly job.
    """

    def __init__(self, utterances: list[Utterance], neighbours: dict[str, dict[str, float]]):
        self.utterances = utterances
        self.terms: list[set[str]] = [content_terms(u.text) for u in utterances]
        self.neighbours = neighbours
        self.postings: dict[str, set[int]] = defaultdict(set)
        for i, ts in enumerate(self.terms):
            for t in ts:
                self.postings[t].add(i)

    def candidates(self, claim_terms: set[str]) -> set[int]:
        out: set[int] = set()
        for t in claim_terms:
            out |= self.postings.get(t, set())
            for other in self.neighbours.get(t, ()):
                out |= self.postings.get(other, set())
        return out


class StatisticalGrader:
    """Ground assertions using learned term associations plus negation polarity.

    Same structure as the lexical grader -- polarity is still checked first and
    still outranks similarity -- but "does this utterance support this claim"
    is answered with soft term matching rather than exact overlap.

    High-frequency terms are absent from the association model by construction
    (see :func:`fit`), so for those this grader degrades to exact matching and
    behaves as the lexical one does. That is the intended floor: the learned
    layer adds reach for mid-frequency clinical vocabulary and is silent where
    it would only add noise.
    """

    def __init__(
        self,
        model: AssociationModel,
        support_threshold: float = 0.5,
        weak_threshold: float = 0.34,
        association_credit: float = 0.7,
    ):
        self.model = model
        self.support_threshold = support_threshold
        self.weak_threshold = weak_threshold
        #: Weight given to an associated term relative to the term itself. An
        #: association is weaker evidence than a quotation, and scoring it as
        #: equal lets a loose neighbourhood of related words masquerade as one.
        self.association_credit = association_credit
        self._nb = model.neighbours()

    @property
    def name(self) -> str:
        dims = 0 if self.model.vectors is None else self.model.vectors.shape[1]
        return f"{GRADER_NAME}(d={dims},floor={self.model.floor:g})"

    def _coverage(self, claim_terms: set[str], utt_terms: set[str]) -> float:
        """Fraction of claim terms present in the utterance, exactly or by association."""
        if not claim_terms:
            return 0.0
        hit = 0.0
        for t in claim_terms:
            if t in utt_terms:
                hit += 1.0
                continue
            ns = self._nb.get(t)
            if not ns:
                continue
            best = 0.0
            # Iterate the smaller side. Claim neighbourhoods are usually a
            # handful of terms and an utterance a couple of dozen.
            if len(ns) <= len(utt_terms):
                for other, sim in ns.items():
                    if other in utt_terms and sim > best:
                        best = sim
            else:
                for other in utt_terms:
                    sim = ns.get(other, 0.0)
                    if sim > best:
                        best = sim
            hit += self.association_credit * best
        return hit / len(claim_terms)

    def ground_note(
        self,
        consultation_id: str,
        model_id: str,
        assertions: list[Assertion],
        utterances: list[Utterance],
        index: _AssocIndex | None = None,
    ) -> NoteGrounding:
        if consultation_id in self.model.fitted_on:
            raise ValueError(
                f"{consultation_id} was in the training set for this association "
                "model. Scoring it would be in-sample; use witness.crossval."
            )
        idx = index or _AssocIndex(utterances, self._nb)
        graded = [self._ground_one(a, idx) for a in assertions]
        return NoteGrounding(consultation_id, model_id, self.name, graded)

    def index_for(self, utterances: list[Utterance]) -> _AssocIndex:
        """Build a reusable index. Scoring many notes against one consultation
        -- which is exactly what the 1,140-annotation evaluation does -- should
        not rebuild it 20 times."""
        return _AssocIndex(utterances, self._nb)

    def _ground_one(self, assertion: Assertion, index: _AssocIndex) -> GroundedAssertion:
        terms = content_terms(assertion.text)
        scored = sorted(
            ((self._coverage(terms, index.terms[i]), i) for i in index.candidates(terms)),
            key=lambda x: (-x[0], x[1]),
        )[:3]
        if not scored or scored[0][0] == 0.0:
            return GroundedAssertion(
                assertion, Verdict.UNSUPPORTED, [],
                "no utterance shares or associates with any clinical term",
                0.6, self.name,
            )

        top_score, top_i = scored[0]
        top = index.utterances[top_i]

        # Polarity outranks similarity, exactly as in the lexical grader. A
        # claim that matches an utterance closely but disagrees about whether
        # the finding is present is the dangerous case.
        if top_score >= self.weak_threshold and is_negated(top.text) != assertion.negated:
            for score, i in scored[1:]:
                if score >= self.weak_threshold and is_negated(
                    index.utterances[i].text
                ) == assertion.negated:
                    return GroundedAssertion(
                        assertion, Verdict.SUPPORTED, [_span(index.utterances[i], i)],
                        "polarity matches a later utterance on the same finding",
                        round(score, 3), self.name,
                    )
            return GroundedAssertion(
                assertion, Verdict.CONTRADICTED, [_span(top, top_i)],
                f"note asserts {'absence' if assertion.negated else 'presence'}; "
                "encounter asserts the opposite",
                round(top_score, 3), self.name,
            )

        if top_score >= self.support_threshold:
            return GroundedAssertion(
                assertion, Verdict.SUPPORTED, [_span(top, top_i)],
                f"{top_score:.0%} term coverage including learned associations",
                round(top_score, 3), self.name,
            )
        if top_score >= self.weak_threshold:
            return GroundedAssertion(
                assertion, Verdict.INFERRED, [_span(top, top_i)],
                f"partial coverage ({top_score:.0%}), consistent polarity",
                round(top_score, 3), self.name,
            )
        return GroundedAssertion(
            assertion, Verdict.UNSUPPORTED, [],
            f"best coverage {top_score:.0%}, below {self.weak_threshold:.0%}",
            round(1.0 - top_score, 3), self.name,
        )


def _span(u: Utterance, i: int) -> Span:
    return Span(i, u.speaker, u.start_s, u.end_s, u.text)
