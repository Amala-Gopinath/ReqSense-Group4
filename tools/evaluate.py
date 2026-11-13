"""ReqSense evaluation: produces the numbers for the report (RQ1, RQ2).

Usage (run from C:\\projects\\doorstop):
  python tools/evaluate.py ambiguity  --methods rules nli rules+nli tfidf+lr sbert+lr
  python tools/evaluate.py conflict   --methods heuristic nli sim+nli
  python tools/evaluate.py redundancy --methods tfidf sbert

Results are printed as Markdown tables and saved in the results/ folder.
"""

import argparse
import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from doorstop.reqsense import ambiguity, conflict, redundancy

RESULTS = Path("results")
BOOTSTRAP = 1000  # number of resamples for the 95% confidence intervals
SEED = 42


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def prf(tp, fp, fn):
    """Precision, recall, F1."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def binary_f1(gold, pred):
    gold, pred = np.asarray(gold, bool), np.asarray(pred, bool)
    return prf(int((gold & pred).sum()), int((~gold & pred).sum()), int((gold & ~pred).sum()))


def macro_f1(gold, pred, classes):
    return float(np.mean([binary_f1([g == c for g in gold], [p == c for p in pred])[2] for c in classes]))


def bootstrap_ci(metric, n_items):
    """95% confidence interval: recompute `metric(indices)` on random resamples of the items."""
    rng = np.random.default_rng(SEED)
    values = [metric(rng.integers(0, n_items, n_items)) for _ in range(BOOTSTRAP)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"[{lo:.2f}, {hi:.2f}]"


def paired_test(metric_a, metric_b, n_items):
    """RQ2: how often (out of 1000 resamples) is method A NOT better than baseline B? (= one-sided p-value)"""
    rng = np.random.default_rng(SEED)
    worse = 0
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n_items, n_items)
        worse += metric_a(idx) <= metric_b(idx)
    return worse / BOOTSTRAP


def save(name, lines):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}_{datetime.now():%Y%m%d-%H%M}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {path}")


def dev_test_split(labels):
    """Stratified 80% dev / 20% test split (same proportion of positives in both parts)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(SEED)
    dev = np.zeros(len(labels), bool)
    for value in (True, False):
        idx = np.flatnonzero(labels == value)
        dev[rng.choice(idx, int(round(0.8 * len(idx))), replace=False)] = True
    return dev, ~dev


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# F1 ambiguity: classify each of the 131 requirements into one of 6 classes
# ---------------------------------------------------------------------------


def trained_predictions(texts, gold, method, folds=4):
    """Supervised classifier (logistic regression) evaluated with stratified k-fold cross-validation.

    Each requirement is predicted by a model that was trained on the OTHER folds only,
    so no requirement is ever tested by a model that saw it during training.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    gold = np.asarray(gold)
    if method == "sbert+lr":
        features = np.asarray(redundancy.embed(texts, "sbert"))  # meaning vectors, computed once
    pred = np.empty(len(texts), dtype=object)
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(texts, gold):
        if method == "tfidf+lr":
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)  # fitted on the training fold only
            x_train = vec.fit_transform([texts[i] for i in train])
            x_test = vec.transform([texts[i] for i in test])
        else:
            x_train, x_test = features[train], features[test]
        model = LogisticRegression(max_iter=3000, class_weight="balanced", C=10)
        model.fit(x_train, gold[train])
        pred[test] = model.predict(x_test)
    return list(pred)


def eval_ambiguity(args):
    rows = read_csv(args.data or "data/defects.csv")
    texts = [r["text"] for r in rows]
    gold = [r["label"] for r in rows]
    classes = ambiguity.CLASSES
    n = len(rows)

    preds, seconds = {}, {}
    for method in args.methods:
        start = time.perf_counter()
        if method in ("tfidf+lr", "sbert+lr"):
            preds[method] = trained_predictions(texts, gold, method)
        else:
            preds[method] = ambiguity.classify(texts, method)
        seconds[method] = time.perf_counter() - start

    lines = [f"# F1 ambiguity / defect classification ({n} requirements, 6 classes)", "",
             "| Method | Accuracy [95% CI] | Macro-F1 [95% CI] | Not classified | Time (s) |", "|---|---|---|---|---|"]
    for m, pred in preds.items():
        acc = lambda idx, p=pred: float(np.mean([gold[i] == p[i] for i in idx]))  # noqa: E731
        mf1 = lambda idx, p=pred: macro_f1([gold[i] for i in idx], [p[i] for i in idx], classes)  # noqa: E731
        everything = range(n)
        lines.append(f"| {m} | {acc(everything):.2f} {bootstrap_ci(acc, n)} | {mf1(everything):.2f} {bootstrap_ci(mf1, n)} "
                     f"| {sum(p is None for p in pred)} | {seconds[m]:.1f} |")

    lines += ["", "Per-class F1:", "", "| Method | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for m, pred in preds.items():
        cells = [f"{binary_f1([g == c for g in gold], [p == c for p in pred])[2]:.2f}" for c in classes]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "Support (number of requirements per class): "
              + ", ".join(f"{c}={gold.count(c)}" for c in classes)]

    if "rules" in preds and len(preds) > 1:
        lines += ["", "Paired bootstrap vs rules baseline (macro-F1): p = probability the method is NOT better", ""]
        base = preds["rules"]
        for m, pred in preds.items():
            if m == "rules":
                continue
            fa = lambda idx, p=pred: macro_f1([gold[i] for i in idx], [p[i] for i in idx], classes)  # noqa: E731
            fb = lambda idx: macro_f1([gold[i] for i in idx], [base[i] for i in idx], classes)  # noqa: E731
            lines.append(f"- {m} vs rules: p = {paired_test(fa, fb, n):.3f}")
    save("ambiguity", lines)


# ---------------------------------------------------------------------------
# F3 conflict: within each dataset, check ALL pairs of requirements
# ---------------------------------------------------------------------------


def eval_conflict(args):
    """Thresholds of the NLI methods are tuned on an 80% dev split of the pairs; scores are reported
    on the remaining 20% test split (which the threshold never saw), as planned in the proposal."""
    reqs = read_csv(args.data or "data/conflict_reqs.csv")
    gold_rows = read_csv(args.pairs)
    by_dataset = defaultdict(list)
    for r in reqs:
        by_dataset[r["dataset"]].append((r["id"], r["text"]))
    gold_pairs = {frozenset((r["id1"], r["id2"])) for r in gold_rows}

    conflict.NLI_THRESHOLD = 0.0  # keep every scored pair; the threshold is chosen below
    labels, dataset_of = [], []
    scores = defaultdict(list)  # method -> one score per pair (0 = not flagged / not checked)
    cost = defaultdict(lambda: [0, 0, 0.0])  # method -> [pairs checked, all pairs, seconds]
    for name, items in by_dataset.items():
        uids = [u for u, _ in items]
        n = len(items)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        labels += [frozenset((uids[i], uids[j])) in gold_pairs for i, j in all_pairs]
        dataset_of += [name] * len(all_pairs)
        for method in args.methods:
            start = time.perf_counter()
            found = conflict.detect_conflicts(items, method, "sbert")
            cost[method][2] += time.perf_counter() - start
            cost[method][0] += conflict.LAST_PAIRS_CHECKED
            cost[method][1] += len(all_pairs)
            score_of = {frozenset(f.uids): f.score for f in found}
            scores[method] += [score_of.get(frozenset((uids[i], uids[j])), 0.0) for i, j in all_pairs]

    labels = np.asarray(labels)
    dataset_of = np.asarray(dataset_of)
    dev, test = dev_test_split(labels)  # stratified 80/20 split of the pairs

    grid = np.round(np.arange(0.50, 1.0, 0.01), 2)
    lines = ["# F3 conflict detection (all pairs within each of the 7 datasets)", "",
             f"{len(labels)} pairs, {int(labels.sum())} conflicts. NLI thresholds tuned on the 80% dev split "
             f"({int(dev.sum())} pairs); scores below are on the 20% test split ({int(test.sum())} pairs).", "",
             "| Method | Threshold | P | R | F1 [95% CI] | Pairs checked by the model | Time (s) |",
             "|---|---|---|---|---|---|---|"]
    preds = {}
    for m in args.methods:
        sc = np.asarray(scores[m])
        if m == "heuristic":
            thr = 0.0
            pred = sc > 0
        else:
            dev_f1 = [binary_f1(labels[dev], sc[dev] >= t)[2] for t in grid]
            thr = float(grid[int(np.argmax(dev_f1))])
            pred = sc >= thr
        preds[m] = pred
        g, pr = labels[test], pred[test]
        f1 = lambda idx, g=g, pr=pr: binary_f1(g[idx], pr[idx])[2]  # noqa: E731
        p, r, f = binary_f1(g, pr)
        checked, total, secs = cost[m]
        lines.append(f"| {m} | {thr:.2f} | {p:.2f} | {r:.2f} | {f:.2f} {bootstrap_ci(f1, len(g))} "
                     f"| {checked}/{total} ({100 * checked / total:.0f}%) | {secs:.1f} |")

    if "heuristic" in preds and len(preds) > 1:
        lines += ["", "Paired bootstrap on the test split vs heuristic baseline (F1): "
                  "p = probability the method is NOT better", ""]
        g, base = labels[test], preds["heuristic"][test]
        for m, pred in preds.items():
            if m != "heuristic":
                pr = pred[test]
                fa = lambda idx, pr=pr: binary_f1(g[idx], pr[idx])[2]  # noqa: E731
                fb = lambda idx: binary_f1(g[idx], base[idx])[2]  # noqa: E731
                lines.append(f"- {m} vs heuristic: p = {paired_test(fa, fb, len(g)):.3f}")

    lines += ["", "Per dataset (all pairs, using the tuned thresholds above):", "",
              "| Dataset | " + " | ".join(f"{m} F1" for m in args.methods) + " |", "|---|" + "---|" * len(args.methods)]
    for name in by_dataset:
        sel = dataset_of == name
        lines.append(f"| {name} | " + " | ".join(f"{binary_f1(labels[sel], preds[m][sel])[2]:.2f}" for m in args.methods) + " |")

    lines += ["", "Threshold sweep on ALL pairs (P / R / F1), for discussion only:", ""]
    for m in args.methods:
        if m != "heuristic":
            sc = np.asarray(scores[m])
            cells = [f"{t:.2f}: " + "/".join(f"{v:.2f}" for v in binary_f1(labels, sc >= t)) for t in (0.5, 0.7, 0.9, 0.95, 0.99)]
            lines.append(f"- {m}: " + "; ".join(cells))
    save("conflict", lines)


# ---------------------------------------------------------------------------
# F2 redundancy: seeded duplicates (id, text, dup_of)
# ---------------------------------------------------------------------------


def eval_redundancy(args):
    """Similarity threshold tuned on an 80% dev split of the pairs, reported on the 20% test split."""
    rows = read_csv(args.data or "data/seeded_duplicates.csv")
    reqs = [(r["id"], r["text"]) for r in rows]
    uids = [u for u, _ in reqs]
    gold_pairs = {frozenset((r["id"], r["dup_of"])) for r in rows if r.get("dup_of")}
    n = len(reqs)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    labels = np.asarray([frozenset((uids[i], uids[j])) in gold_pairs for i, j in all_pairs])
    dev, test = dev_test_split(labels)
    grid = np.round(np.arange(0.05, 1.0, 0.01), 2)

    lines = [f"# F2 redundancy ({n} requirements, {len(gold_pairs)} duplicate pairs, {len(all_pairs)} pairs)", "",
             f"Threshold tuned on the 80% dev split ({int(dev.sum())} pairs); scores on the 20% test split "
             f"({int(test.sum())} pairs, {int(labels[test].sum())} duplicates).", "",
             "| Method | Threshold | P | R | F1 [95% CI] | Time (s) |", "|---|---|---|---|---|---|"]
    sweep = ["", "Threshold sweep on ALL pairs (P / R / F1), for discussion only:", ""]
    preds = {}
    for method in args.methods:
        start = time.perf_counter()
        sim = redundancy.similarity_matrix([t for _, t in reqs], method)
        seconds = time.perf_counter() - start
        scores = np.asarray([sim[i][j] for i, j in all_pairs])
        dev_f1 = [binary_f1(labels[dev], scores[dev] >= t)[2] for t in grid]
        thr = float(grid[int(np.argmax(dev_f1))])
        preds[method] = scores >= thr
        g, pr = labels[test], preds[method][test]
        f1 = lambda idx, g=g, pr=pr: binary_f1(g[idx], pr[idx])[2]  # noqa: E731
        p, r, f = binary_f1(g, pr)
        lines.append(f"| {method} | {thr:.2f} | {p:.2f} | {r:.2f} | {f:.2f} {bootstrap_ci(f1, len(g))} | {seconds:.1f} |")
        cells = [f"{t:.1f}: " + "/".join(f"{v:.2f}" for v in binary_f1(labels, scores >= t)) for t in np.arange(0.3, 1.0, 0.1)]
        sweep.append(f"- {method}: " + "; ".join(cells))

    if "tfidf" in preds and len(preds) > 1:
        lines += ["", "Paired bootstrap on the test split vs TF-IDF baseline (F1): p = probability the method is NOT better", ""]
        g, base = labels[test], preds["tfidf"][test]
        for m, pred in preds.items():
            if m != "tfidf":
                pr = pred[test]
                fa = lambda idx, pr=pr: binary_f1(g[idx], pr[idx])[2]  # noqa: E731
                fb = lambda idx: binary_f1(g[idx], base[idx])[2]  # noqa: E731
                lines.append(f"- {m} vs tfidf: p = {paired_test(fa, fb, len(g)):.3f}")
    save("redundancy", lines + sweep)


def main():
    parser = argparse.ArgumentParser(description="ReqSense evaluation")
    parser.add_argument("task", choices=["ambiguity", "conflict", "redundancy"])
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--data", help="CSV with requirements (default: data/...)")
    parser.add_argument("--pairs", default="data/conflict_pairs.csv", help="conflict: CSV with gold pairs")
    args = parser.parse_args()
    {"ambiguity": eval_ambiguity, "conflict": eval_conflict, "redundancy": eval_redundancy}[args.task](args)


if __name__ == "__main__":
    main()