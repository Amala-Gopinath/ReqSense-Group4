"""F3: conflict detector - finds pairs of requirements that contradict each other.

Techniques (Table I of the proposal):
  * heuristic : negation / opposite-word rules                        (baseline)
  * nli       : NLI cross-encoder on ALL pairs                         (baseline)
  * sim+nli   : similar pairs first (Sentence-BERT), then NLI on those (proposed, tiered)
"""

import re

import numpy as np

from doorstop.reqsense.model import Finding

NLI_MODEL = "cross-encoder/nli-distilroberta-base"  # RoBERTa-based, like the NLI model in [6]
CANDIDATE_THRESHOLD = 0.5  # sim+nli: only pairs at least this similar are sent to NLI
NLI_THRESHOLD = 0.5  # report a conflict if the contradiction probability >= this
LAST_PAIRS_CHECKED = 0  # how many pairs the last run sent to the checker (for the evaluation)

# ---------------------------------------------------------------------------
# 1. Baseline: simple rules
# ---------------------------------------------------------------------------

STOP_WORDS = set(
    "a an the of to in on for by with and or is are be been shall must will should may can "
    "could would system it its this that these those as at from all any each which when "
    "than then also only after before within".split()
)
NEGATION = re.compile(
    r"\b(not|no|never|none|cannot|can't|without|prohibited|forbidden|unable|nobody|nothing)\b",
    re.IGNORECASE,
)
OPPOSITES = [
    ("allow", "deny"), ("allow", "prevent"), ("allow", "prohibit"), ("permit", "prohibit"),
    ("enable", "disable"), ("accept", "reject"), ("include", "exclude"), ("lock", "unlock"),
    ("show", "hide"), ("display", "hide"), ("mandatory", "optional"), ("required", "optional"),
    ("always", "never"), ("increase", "decrease"), ("online", "offline"), ("public", "private"),
    ("manual", "automatic"), ("retain", "delete"), ("keep", "delete"), ("store", "delete"),
    ("minimum", "maximum"), ("encrypted", "unencrypted"), ("start", "stop"), ("grant", "revoke"),
]


def content_words(text):
    """Important words of a sentence (lower case, no stop words, no negations, no plural 's')."""
    found = re.findall(r"[a-z][a-z'-]+", text.lower())
    return {w.rstrip("s") for w in found if w not in STOP_WORDS and not NEGATION.fullmatch(w)}


def has_word(words, target):
    return any(w.startswith(target) for w in words)


def heuristic(a, b, min_overlap=0.3):
    """Return (score, reason). Score 0 means 'no conflict found'."""
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0, ""
    overlap = len(wa & wb) / len(wa | wb)  # how much the two requirements talk about the same thing
    if overlap < min_overlap:
        return 0.0, ""
    # rule 1: one is negative, the other is not
    # (count negations: "not ... without" = 2 negations, which cancel out like in maths)
    if len(NEGATION.findall(a)) % 2 != len(NEGATION.findall(b)) % 2:
        return 0.5 + overlap / 2, f"one requirement is negated, the other is not (word overlap {overlap:.2f})"
    # rule 2: opposite words, e.g. allow / deny
    for x, y in OPPOSITES:
        if (has_word(wa, x) and has_word(wb, y)) or (has_word(wa, y) and has_word(wb, x)):
            return 0.45 + overlap / 2, f"opposite words '{x}' / '{y}' on the same subject (word overlap {overlap:.2f})"
    return 0.0, ""


# ---------------------------------------------------------------------------
# 2. NLI model (AI): for each pair, probability that they CONTRADICT
# ---------------------------------------------------------------------------

_nli_model = None


def load_nli():
    """Load the NLI model once (downloads ~330 MB the first time)."""
    global _nli_model  # pylint: disable=global-statement
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder(NLI_MODEL)
    return _nli_model


def contradiction_index(model):
    """Find which output column means 'contradiction' (models differ in label order)."""
    for config in (getattr(model, "config", None), getattr(getattr(model, "model", None), "config", None)):
        labels = getattr(config, "id2label", None)
        if labels:
            for idx, name in labels.items():
                if "contradiction" in str(name).lower():
                    return int(idx)
    return 0  # cross-encoder/nli-* models: 0 = contradiction, 1 = entailment, 2 = neutral


def nli_contradiction(pairs):
    """Return one contradiction probability (0..1) per (text_a, text_b) pair."""
    if not pairs:
        return np.zeros(0)
    model = load_nli()
    col = contradiction_index(model)

    def probs(batch):
        scores = np.asarray(model.predict(batch), dtype=float)
        if scores.min() < 0 or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
            scores = np.exp(scores - scores.max(axis=1, keepdims=True))  # softmax: raw scores -> probabilities
            scores = scores / scores.sum(axis=1, keepdims=True)
        return scores[:, col]

    forward = probs(pairs)
    backward = probs([(b, a) for a, b in pairs])  # check both directions
    return np.maximum(forward, backward)


# ---------------------------------------------------------------------------
# 3. The detector
# ---------------------------------------------------------------------------


def detect_conflicts(requirements, method="heuristic", embedder="sbert"):
    """Return a list of Findings for pairs that look contradictory."""
    from doorstop.reqsense.redundancy import similarity_matrix

    uids = [u for u, _ in requirements]
    texts = [t for _, t in requirements]
    n = len(requirements)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Stage 1 (only for sim+nli): keep only pairs that talk about the same thing
    if method == "sim+nli":
        sim = similarity_matrix(texts, embedder)
        pairs = [(i, j) for i, j in all_pairs if sim[i][j] >= CANDIDATE_THRESHOLD]
    else:
        pairs = all_pairs
    global LAST_PAIRS_CHECKED  # pylint: disable=global-statement
    LAST_PAIRS_CHECKED = len(pairs)
    print(f"ReqSense conflict ({method}): checking {len(pairs)} of {len(all_pairs)} pairs")

    findings = []
    if method == "heuristic":
        for i, j in pairs:
            score, reason = heuristic(texts[i], texts[j])
            if score > 0:
                findings.append(_finding(uids[i], uids[j], score, "heuristic", reason))
    elif method in ("nli", "sim+nli"):
        probs = nli_contradiction([(texts[i], texts[j]) for i, j in pairs])
        for (i, j), p in zip(pairs, probs):
            if p >= NLI_THRESHOLD:
                findings.append(_finding(uids[i], uids[j], float(p), method, f"NLI contradiction probability {p:.2f}"))
    else:
        raise ValueError(f"unknown conflict method: {method}")
    findings.sort(key=lambda f: -f.score)
    return findings


def _finding(uid_a, uid_b, score, stage, reason):
    return Finding(
        check="conflict",
        uids=[uid_a, uid_b],
        defect="contradiction",
        rationale=reason,
        score=round(score, 3),
        suggestion=f"Ask stakeholders which of {uid_a}/{uid_b} is correct, or add the condition under which each applies.",
        stage=stage,
    )