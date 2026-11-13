"""F2: redundancy detector - finds pairs of requirements that say the same thing.

Two techniques (Table I of the proposal):
  * baseline : TF-IDF vectors + cosine similarity   (--embedder tfidf)
  * proposed : Sentence-BERT embeddings + cosine     (--embedder sbert)
"""

import numpy as np

from doorstop.reqsense.model import Finding

# Pairs at or above this similarity are reported. Tune these on your dev data later.
DEFAULT_THRESHOLD = {"tfidf": 0.40, "sbert": 0.80}
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_sbert_model = None  # loaded once, then reused (loading takes several seconds)


def embed(texts, method="tfidf"):
    """Turn each requirement text into a vector of numbers (one row per requirement).

    Rows are normalised to length 1, so the dot product of two rows = cosine similarity.
    """
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        return vectorizer.fit_transform(texts).toarray()
    if method == "sbert":
        global _sbert_model  # pylint: disable=global-statement
        if _sbert_model is None:
            from sentence_transformers import SentenceTransformer

            _sbert_model = SentenceTransformer(SBERT_MODEL)
        return _sbert_model.encode(texts, normalize_embeddings=True)
    raise ValueError(f"unknown method: {method}")


def similarity_matrix(texts, method="tfidf"):
    """Return an N x N table where cell [i][j] = how similar requirement i and j are (0..1)."""
    vectors = np.asarray(embed(texts, method), dtype=float)
    return vectors @ vectors.T


def detect_redundancy(requirements, method="tfidf", threshold=None):
    """Compare every pair of requirements and report those that are too similar."""
    if threshold is None:
        threshold = DEFAULT_THRESHOLD[method]
    if len(requirements) < 2:
        return []

    uids = [uid for uid, _ in requirements]
    texts = [text for _, text in requirements]
    sim = similarity_matrix(texts, method)

    findings = []
    n = len(requirements)
    for i in range(n):
        for j in range(i + 1, n):  # j > i, so each pair is checked only once
            score = float(sim[i][j])
            if score >= threshold:
                findings.append(
                    Finding(
                        check="redundancy",
                        uids=[uids[i], uids[j]],
                        defect="duplicate" if score >= 0.95 else "near-duplicate",
                        rationale=f"{method} cosine similarity {score:.2f} >= {threshold:.2f}",
                        score=round(score, 3),
                        suggestion=f"Merge {uids[j]} into {uids[i]} (or link them) and delete the duplicate.",
                        stage=method,
                    )
                )
    findings.sort(key=lambda f: -f.score)  # most similar first
    return findings