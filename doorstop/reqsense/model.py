"""Shared data structure for all three detectors."""

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class Finding:
    """One problem found by a detector."""

    check: str  # "ambiguity", "redundancy" or "conflict"
    uids: List[str]  # requirement ID(s), e.g. ["REQ002"] or ["REQ012", "REQ013"]
    defect: str  # e.g. "non_measurable", "duplicate", "contradiction"
    rationale: str  # short human explanation of WHY it was flagged
    score: float = 1.0  # confidence between 0 and 1
    suggestion: str = ""  # how to fix it
    stage: str = "rules"  # which technique found it (rules, tfidf, sbert, nli, llm...)
    evidence: List[str] = field(default_factory=list)  # the words that triggered it

    def to_dict(self):
        """Convert to a plain dictionary (used for JSON output)."""
        return asdict(self)