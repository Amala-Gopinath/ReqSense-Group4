"""F1: ambiguity / specification-defect detector (rule-based baseline).

The 6 defect classes come from the labelled dataset of Fazelnia et al. (2024):
ambiguous, non_measurable, optional, uncertain, non_atomic, directive.
"""

import re

import numpy as np

from doorstop.reqsense.model import Finding

# ---------------------------------------------------------------------------
# 1. Word lists ("lexicon"): words that usually signal each defect class.
#    Each entry is (regular expression, score). The score is how sure we are.
# ---------------------------------------------------------------------------


def words(word_list, score):
    """Turn a list of words/phrases into (regex, score) pairs that match whole words only."""
    return [(r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"\b", score) for w in word_list]


RULES = {
    # leaves the decision to the developer
    "optional": words(
        ["may", "might", "could", "optionally", "if necessary", "if needed", "if possible",
         "if appropriate", "where applicable", "as required", "as appropriate", "as needed",
         "at the discretion of", "preferably", "ideally", "whenever possible"],
        0.85,
    )
    + words(["should", "can"], 0.6),  # weaker signal
    # vague quantity / time that cannot be measured or tested
    "non_measurable": words(
        ["large", "small", "long", "short", "high", "low", "many", "few", "several", "numerous",
         "minimal", "minimum", "maximum", "as soon as possible", "asap", "in a timely manner",
         "timely", "quickly", "promptly", "rapidly", "immediately", "frequently", "rarely",
         "periodically", "regularly", "sufficient", "adequate", "acceptable", "reasonable",
         "significant", "significantly", "approximately", "enough"],
        0.8,
    ),
    # undefined quality, or a standard/regulation that is not named
    "uncertain": words(
        ["good", "bad", "fast", "slow", "efficient", "efficiently", "effective", "user-friendly",
         "user friendly", "easy", "easily", "simple", "intuitive", "robust", "flexible",
         "seamless", "seamlessly", "secure", "reliable", "scalable", "optimal", "normal",
         "typical", "proper", "properly", "state of the art", "best practices", "tbd",
         "to be determined", "etc"],
        0.75,
    )
    + [(r"\b(applicable|relevant|appropriate)\s+([\w-]+\s+){0,2}(standards?|regulations?|laws?|guidelines?|policies)\b", 0.8)],
    # can be read in more than one way
    "ambiguous": words(["and/or", "any", "various", "relevant", "appropriate", "similar",
                        "such as", "some", "certain", "related"], 0.65)
    + words(["it", "they", "them", "its", "their"], 0.55)  # pronoun: what does "it" refer to?
    + [(r"\b(better|faster|easier|more|less|greater|higher|lower)\b(?!\s+than)", 0.6)],  # comparative
    # describes more than one action ("... and email it ...")
    "non_atomic": [
        (r"\b(and|or|as well as|and also|and then)\s+(shall\s+|must\s+|will\s+|also\s+)?"
         r"(accept|allow|calculate|check|create|delete|display|email|export|generate|import|"
         r"inform|log|notify|print|process|provide|record|reject|remove|report|save|send|show|"
         r"store|update|upload|validate|verify|write)(s|es|ed)?\b", 0.75),
        (r";\s*\w+", 0.6),
    ],
    # points the reader to another place instead of stating the requirement
    "directive": [
        (r"\b(see|refer to)\s+(the\s+)?(figure|fig\.?|table|section|appendix|annex|diagram|document)\b", 0.85),
        (r"\b(as\s+)?(shown|described|specified|defined|listed)\s+in\s+(the\s+)?(figure|fig\.?|table|section|appendix|annex|diagram)\b", 0.85),
        (r"\b(e\.g\.|i\.e\.|for example)", 0.6),
    ],
}

EXPLANATION = {
    "optional": "optional wording leaves the obligation open",
    "non_measurable": "vague quantity or time with no measurable value",
    "uncertain": "undefined quality or unnamed standard",
    "ambiguous": "vague wording that can be read in more than one way",
    "non_atomic": "describes more than one action",
    "directive": "points to another artefact instead of stating the requirement",
}

SUGGESTION = {
    "optional": "Use 'shall' if it is mandatory, or make it a separate optional requirement.",
    "non_measurable": "Replace '{term}' with a measurable value, e.g. 'within 2 seconds for 95% of requests'.",
    "uncertain": "Define '{term}' with an acceptance criterion, or cite the exact standard (name, version).",
    "ambiguous": "Replace '{term}' with the exact item(s) meant.",
    "non_atomic": "Split into one requirement per action.",
    "directive": "Copy the referenced content into the requirement text.",
}

# A number close to the vague word makes it measurable: "minimum of 10 users", "at least 12".
NUMBER_AFTER = re.compile(r"^\W*(\w+\W+){0,3}?\d")
NUMBER_BEFORE = re.compile(r"\d\W*(\w+\W+){0,2}?\W*$")
COMPILED = {cls: [(re.compile(p, re.IGNORECASE), s) for p, s in rules] for cls, rules in RULES.items()}


# ---------------------------------------------------------------------------
# 2. The detector
# ---------------------------------------------------------------------------


def find_problems(text):
    """Return {defect_class: (score, [matched words])} for one requirement text."""
    hits = {}
    for cls, rules in COMPILED.items():
        for regex, score in rules:
            for match in regex.finditer(text):
                # skip vague words that have a number right next to them (they ARE measurable)
                if cls in ("non_measurable", "ambiguous") and (
                    NUMBER_AFTER.match(text[match.end():]) or NUMBER_BEFORE.search(text[: match.start()])
                ):
                    continue
                term = match.group(0)
                old_score, terms = hits.get(cls, (0.0, []))
                if term.lower() not in [t.lower() for t in terms]:
                    terms.append(term)
                hits[cls] = (max(old_score, score), terms)
    return hits


def detect_ambiguity(requirements, method="rules", min_score=0.5):
    """F1: return a list of Findings.

    method="rules"     : every rule that fires is reported (baseline)
    method="rules+nli" : rules propose, the NLI model keeps only the most likely class per requirement
    """
    keep = None
    if method == "rules+nli" and requirements:
        chosen = classify([t for _, t in requirements], "rules+nli")
        keep = {uid: cls for (uid, _), cls in zip(requirements, chosen)}
    findings = []
    for uid, text in requirements:
        for cls, (score, terms) in find_problems(text).items():
            if score < min_score:
                continue
            if keep is not None and keep[uid] != cls:
                continue  # NLI says another class fits better: drop this finding
            quoted = ", ".join(f"'{t}'" for t in terms)
            findings.append(
                Finding(
                    check="ambiguity",
                    uids=[uid],
                    defect=cls,
                    rationale=f"{EXPLANATION[cls]}: {quoted}",
                    score=score,
                    suggestion=SUGGESTION[cls].format(term=terms[0]),
                    stage=method,
                    evidence=terms,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# 3. AI layer: zero-shot NLI (like Fazelnia et al. 2024)
#    For each class we ask the NLI model: does the requirement ENTAIL this sentence?
#    We use 3 sentences per class and average them, so one wording does not decide everything.
# ---------------------------------------------------------------------------

# A stronger NLI model, trained on several NLI datasets (MNLI, FEVER, ANLI) and widely used for
# zero-shot classification. (The conflict detector keeps the smaller, faster model.)
AMBIGUITY_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

HYPOTHESES = {
    "ambiguous": [
        "This requirement is ambiguous and can be interpreted in more than one way.",
        "The meaning of this requirement is unclear.",
        "Different readers could understand this requirement differently.",
    ],
    "non_measurable": [
        "This requirement uses a quantity that cannot be measured or tested.",
        "This requirement does not give a measurable value.",
        "This requirement cannot be verified by a test because no value is given.",
    ],
    "optional": [
        "This requirement is optional and leaves the choice to the developer.",
        "This requirement does not say whether something is mandatory.",
        "The developer can decide whether to implement this requirement.",
    ],
    "uncertain": [
        "This requirement uses vague or subjective words such as good or fast.",
        "This requirement describes a quality that is not clearly defined.",
        "This requirement relies on a subjective judgement.",
    ],
    "non_atomic": [
        "This requirement contains more than one requirement.",
        "This requirement describes several different actions.",
        "This requirement should be split into separate requirements.",
    ],
    "directive": [
        "This requirement refers to another document, figure or source for its details.",
        "This requirement points to external information instead of stating it.",
        "The details of this requirement are found somewhere else.",
    ],
}
CLASSES = list(HYPOTHESES)

_ambiguity_nli = None


def load_ambiguity_nli():
    """Load the zero-shot NLI model once (downloads ~740 MB the first time)."""
    global _ambiguity_nli  # pylint: disable=global-statement
    if _ambiguity_nli is None:
        from sentence_transformers import CrossEncoder

        _ambiguity_nli = CrossEncoder(AMBIGUITY_NLI_MODEL)
        # The model is stored in float16, which is very slow on a normal CPU: convert to float32.
        inner = getattr(_ambiguity_nli, "model", None)
        if inner is not None and hasattr(inner, "float"):
            inner.float()
    return _ambiguity_nli


_probs_cache = {}  # the same texts are scored only once (nli and rules+nli share the result)


def nli_class_probs(texts):
    """Return a table (one row per text, one column per class in CLASSES) of entailment probabilities."""
    key = tuple(texts)
    if key not in _probs_cache:
        _probs_cache[key] = _compute_class_probs(texts)
    return _probs_cache[key]


def _compute_class_probs(texts):
    model = load_ambiguity_nli()
    entail_col = 0  # this model's label order: entailment, neutral, contradiction
    for config in (getattr(model, "config", None), getattr(getattr(model, "model", None), "config", None)):
        for idx, name in (getattr(config, "id2label", None) or {}).items():
            if "entail" in str(name).lower():
                entail_col = int(idx)
    table = np.zeros((len(texts), len(CLASSES)))
    for k, cls in enumerate(CLASSES):
        for hypothesis in HYPOTHESES[cls]:
            scores = np.asarray(model.predict([(t, hypothesis) for t in texts]), dtype=float)
            if scores.min() < 0 or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
                scores = np.exp(scores - scores.max(axis=1, keepdims=True))  # softmax
                scores = scores / scores.sum(axis=1, keepdims=True)
            table[:, k] += scores[:, entail_col] / len(HYPOTHESES[cls])  # average of the 3 sentences
    return table


def classify(texts, method="rules"):
    """Pick ONE defect class per requirement (used for evaluation). Returns a list of class names (or None)."""
    rule_hits = [find_problems(t) for t in texts]
    if method == "rules":
        return [max(h, key=lambda c: h[c][0]) if h else None for h in rule_hits]
    probs = nli_class_probs(texts)
    if len(texts) >= 20:
        # Calibration: remove each class's average score ("label bias"), so a class only wins
        # when this requirement scores HIGHER than usual for it. Uses no labels.
        probs = probs - probs.mean(axis=0)
    result = []
    for h, row in zip(rule_hits, probs):
        if method == "rules+nli" and h:
            candidates = [CLASSES.index(c) for c in h]  # rules propose, NLI decides
        else:
            candidates = range(len(CLASSES))  # "nli", or no rule fired
        best = max(candidates, key=lambda k: row[k])
        result.append(CLASSES[best])
    return result