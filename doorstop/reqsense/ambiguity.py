"""F1: ambiguity / specification-defect detector (rule-based baseline).

The 6 defect classes come from the labelled dataset of Fazelnia et al. (2024):
ambiguous, non_measurable, optional, uncertain, non_atomic, directive.
"""

import re

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


def detect_ambiguity(requirements, min_score=0.5):
    """F1 baseline: run the rules on every (uid, text) and return a list of Findings."""
    findings = []
    for uid, text in requirements:
        for cls, (score, terms) in find_problems(text).items():
            if score < min_score:
                continue
            quoted = ", ".join(f"'{t}'" for t in terms)
            findings.append(
                Finding(
                    check="ambiguity",
                    uids=[uid],
                    defect=cls,
                    rationale=f"{EXPLANATION[cls]}: {quoted}",
                    score=score,
                    suggestion=SUGGESTION[cls].format(term=terms[0]),
                    stage="rules",
                    evidence=terms,
                )
            )
    return findings