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
        return vectorizer.fit_transform(texts)  # sparse matrix: stays small even for 10,000 texts
    if method == "sbert":
        global _sbert_model  # pylint: disable=global-statement
        if _sbert_model is None:
            from sentence_transformers import SentenceTransformer

            _sbert_model = SentenceTransformer(SBERT_MODEL)
        return _sbert_model.encode(texts, normalize_embeddings=True)
    raise ValueError(f"unknown method: {method}")


def similarity_matrix(texts, method="tfidf"):
    """Return an N x N table where cell [i][j] = how similar requirement i and j are (0..1).

    Only for small sets (evaluation). For large trees use similar_pairs(), which needs far less memory.
    """
    vectors = embed(texts, method)
    sim = vectors @ vectors.T
    return sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim, dtype=float)


def similar_pairs(texts, method="tfidf", threshold=0.5, top_k=None, chunk=1000):
    """Return [(i, j, similarity)] with i < j and similarity >= threshold.

    Works block by block (1,000 rows at a time) so memory stays small, and uses fast
    matrix maths instead of a Python loop over every pair. With top_k, each requirement
    keeps only its top_k most similar partners (bounds the work on very large trees).
    """
    vectors = embed(texts, method)
    n = vectors.shape[0]
    found = {}
    for start in range(0, n, chunk):
        block = vectors[start:start + chunk] @ vectors.T  # similarities of these rows to ALL rows
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        for r in range(block.shape[0]):
            i = start + r
            row = block[r]
            row[i] = -1.0  # ignore similarity with itself
            hits = np.flatnonzero(row >= threshold)
            if top_k and len(hits) > top_k:
                hits = hits[np.argsort(-row[hits])[:top_k]]
            for j in hits:
                a, b = (i, int(j)) if i < j else (int(j), i)
                found[(a, b)] = float(row[j])
    return [(a, b, score) for (a, b), score in sorted(found.items())]


def detect_redundancy(requirements, method="tfidf", threshold=None):
    """Compare every pair of requirements and report those that are too similar."""
    if threshold is None:
        threshold = DEFAULT_THRESHOLD[method]
    if len(requirements) < 2:
        return []

    uids = [uid for uid, _ in requirements]
    texts = [text for _, text in requirements]

    findings = []
    for i, j, score in similar_pairs(texts, method, threshold):  # only pairs above the threshold
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