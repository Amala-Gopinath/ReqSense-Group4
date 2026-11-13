"""Tests for ReqSense (the `doorstop analyze` extension).

Run with:  pytest doorstop/reqsense --cov=doorstop/reqsense --cov-report=term-missing

The AI models are replaced by small FAKE models, so the tests are fast and need no downloads.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from doorstop.reqsense import ambiguity, cli, conflict, redundancy, report
from doorstop.reqsense.model import Finding

DEMO = [
    ("REQ001", "The system shall lock a user account after 5 consecutive failed login attempts."),
    ("REQ002", "The system shall respond to search queries quickly."),
    ("REQ003", "The application may send a confirmation email to the customer if necessary."),
    ("REQ004", "The user interface shall be user-friendly and intuitive."),
    ("REQ005", "The system shall generate the monthly sales report and email it to the store manager."),
    ("REQ006", "The export format shall be as shown in Figure 3."),
    ("REQ007", "The system shall store relevant customer data and/or order data."),
    ("REQ008", "The system shall encrypt all stored passwords using bcrypt."),
    ("REQ009", "All stored passwords shall be encrypted by the system with bcrypt."),
    ("REQ012", "The system shall allow guest users to place orders without creating an account."),
    ("REQ013", "Guest users shall not be able to place orders without creating an account."),
    ("REQ014", "The system shall retain audit logs for 7 years."),
    ("REQ015", "The system shall delete audit logs after 90 days."),
]


# ---------------------------------------------------------------------------
# fake AI models
# ---------------------------------------------------------------------------


class FakeNLI:
    """Pretends to be an NLI cross-encoder. Label order: contradiction, entailment, neutral."""

    def __init__(self):
        self.config = type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()

    def predict(self, pairs):
        rows = []
        for a, b in pairs:
            if b.startswith("This requirement") or b.startswith("The "):  # ambiguity hypothesis
                ent = 0.9 if ("quick" in a and "measur" in b) or ("may" in a and "optional" in b) else 0.1
                rows.append([0.05, ent, 0.95 - ent])
            else:  # conflict pair: contradiction if exactly one side says "not"/"delete"
                c = 0.95 if (" not " in a or "delete" in a) != (" not " in b or "delete" in b) else 0.05
                rows.append([c, 0.02, 0.98 - c])
        return np.asarray(rows)


class FakeSBERT:
    """Pretends to be Sentence-BERT: bag-of-words vectors (normalised)."""

    def encode(self, texts, normalize_embeddings=True):
        import re

        vocab = sorted({w for t in texts for w in re.findall(r"[a-z]+", t.lower())})
        m = np.zeros((len(texts), len(vocab)))
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                m[r, vocab.index(w)] += 1
        return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(conflict, "_nli_model", FakeNLI())
    monkeypatch.setattr(ambiguity, "_ambiguity_nli", FakeNLI())
    monkeypatch.setattr(redundancy, "_sbert_model", FakeSBERT())
    monkeypatch.setattr(ambiguity, "_probs_cache", {})


# ---------------------------------------------------------------------------
# F1 ambiguity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The system shall respond quickly.", "non_measurable"),
        ("The app may send an email if necessary.", "optional"),
        ("The UI shall be user-friendly.", "uncertain"),
        ("The system shall comply with applicable data-protection regulations.", "uncertain"),
        ("The system shall generate the report and email it to the manager.", "non_atomic"),
        ("The format shall be as shown in Figure 3.", "directive"),
        ("The system shall store customer and/or order data.", "ambiguous"),
    ],
)
def test_rules_find_each_defect_class(text, expected):
    assert expected in ambiguity.find_problems(text)


@pytest.mark.parametrize(
    "text",
    [
        "The system shall lock the account after 5 failed attempts.",
        "The pool shall hold a minimum of 10 connections.",
    ],
)
def test_clean_or_measurable_text_is_not_flagged(text):
    assert ambiguity.find_problems(text) == {}


def test_detect_ambiguity_on_demo():
    found = {(f.uids[0], f.defect) for f in ambiguity.detect_ambiguity(DEMO)}
    assert ("REQ002", "non_measurable") in found
    assert ("REQ003", "optional") in found
    assert ("REQ006", "directive") in found
    assert not any(uid == "REQ001" for uid, _ in found)  # clean requirement


def test_finding_has_rationale_and_suggestion():
    f = ambiguity.detect_ambiguity([("R1", "The system shall respond quickly.")])[0]
    assert "quickly" in f.rationale and "quickly" in f.suggestion
    assert f.to_dict()["check"] == "ambiguity"


def test_classify_rules():
    assert ambiguity.classify(["Reports shall be printed in colour."], "rules") == [None]
    assert ambiguity.classify(["The system may log errors."], "rules") == ["optional"]


def test_classify_nli_and_hybrid(fake_models):
    texts = ["The system shall respond quickly.", "The system may log errors."] * 12  # >= 20 -> calibration
    nli = ambiguity.classify(texts, "nli")
    hybrid = ambiguity.classify(texts, "rules+nli")
    assert len(nli) == len(texts) and all(c in ambiguity.CLASSES for c in nli)
    assert hybrid[0] == "non_measurable" and hybrid[1] == "optional"


def test_detect_ambiguity_rules_plus_nli_keeps_one_class(fake_models):
    found = ambiguity.detect_ambiguity([("R1", "The system may respond quickly.")], method="rules+nli")
    assert len(found) == 1 and found[0].stage == "rules+nli"


# ---------------------------------------------------------------------------
# F2 redundancy
# ---------------------------------------------------------------------------


def test_redundancy_tfidf_finds_duplicate():
    reqs = [
        ("A", "The system shall export the monthly report as CSV."),
        ("B", "The monthly report shall be exported by the system as CSV."),
        ("C", "Users shall reset passwords by email."),
    ]
    found = redundancy.detect_redundancy(reqs, "tfidf", threshold=0.3)
    assert [f.uids for f in found] == [["A", "B"]]


def test_redundancy_sbert_and_edge_cases(fake_models):
    found = redundancy.detect_redundancy(DEMO, "sbert", threshold=0.7)
    assert ["REQ008", "REQ009"] in [f.uids for f in found]
    assert redundancy.detect_redundancy(DEMO[:1]) == []
    with pytest.raises(ValueError):
        redundancy.embed(["x"], "unknown")


def test_similarity_matrix_is_symmetric():
    sim = redundancy.similarity_matrix([t for _, t in DEMO], "tfidf")
    assert np.allclose(sim, sim.T) and np.allclose(np.diag(sim), 1.0)


# ---------------------------------------------------------------------------
# F3 conflict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,is_conflict",
    [
        ("The system shall allow guests to order.", "The system shall not allow guests to order.", True),
        ("The system shall enable the camera at login.", "The system shall disable the camera at login.", True),
        ("Reports are printed in colour.", "The camera is enabled at login.", False),
        ("Guests shall order without an account.", "Guests shall not order without an account.", True),
    ],
)
def test_conflict_heuristic(a, b, is_conflict):
    score, reason = conflict.heuristic(a, b)
    assert (score > 0) == is_conflict
    assert bool(reason) == is_conflict


def test_detect_conflicts_heuristic_on_demo():
    found = [set(f.uids) for f in conflict.detect_conflicts(DEMO, "heuristic")]
    assert {"REQ012", "REQ013"} in found and {"REQ014", "REQ015"} in found
    assert conflict.LAST_PAIRS_CHECKED == len(DEMO) * (len(DEMO) - 1) // 2


def test_tiered_sim_nli_checks_fewer_pairs(fake_models):
    found = [set(f.uids) for f in conflict.detect_conflicts(DEMO, "sim+nli", "sbert")]
    assert {"REQ012", "REQ013"} in found
    assert conflict.LAST_PAIRS_CHECKED < len(DEMO) * (len(DEMO) - 1) // 2  # the cost saving


def test_nli_all_pairs_and_errors(fake_models):
    found = conflict.detect_conflicts(DEMO[9:11], "nli")
    assert found and found[0].stage == "nli"
    assert len(conflict.nli_contradiction([])) == 0
    with pytest.raises(ValueError):
        conflict.detect_conflicts(DEMO, "unknown")


# ---------------------------------------------------------------------------
# report + CLI helpers
# ---------------------------------------------------------------------------

FINDINGS = [
    Finding("ambiguity", ["REQ002"], "non_measurable", "vague | word", 0.8, "fix it"),
    Finding("conflict", ["REQ012", "REQ013"], "contradiction", "negation", 0.9, "ask"),
]


def test_report_formats():
    console = report.render(FINDINGS, DEMO, "console")
    assert "REQ012 <-> REQ013" in console and "2 issue(s)" in console
    md = report.render(FINDINGS, DEMO, "md", {"seconds": 1.0})
    assert "# ReqSense analysis report" in md and "vague \\| word" in md
    data = json.loads(report.render(FINDINGS, DEMO, "json"))
    assert data["summary"] == {"ambiguity": 1, "redundancy": 0, "conflict": 1}


@pytest.mark.parametrize(
    "fail_on,expected",
    [("none", False), ("any", True), ("conflict", True), ("redundancy", False), ("ambiguity,redundancy", True)],
)
def test_should_fail(fail_on, expected):
    assert cli.should_fail(fail_on, FINDINGS) == expected


# ---------------------------------------------------------------------------
# integration: real Doorstop tree + the real `doorstop analyze` command
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_tree(tmp_path_factory):
    """Create a small Doorstop project with the demo requirements."""
    from doorstop.core.builder import build

    root = tmp_path_factory.mktemp("demo")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        tree = build(cwd=str(root), root=str(root))
        doc = tree.create_document(str(root / "reqs"), "REQ")
        for _, text in DEMO:
            item = doc.add_item()
            item.text = text
            item.save()
    finally:
        os.chdir(cwd)
    return root


def run_doorstop(args, cwd):
    return subprocess.run([sys.executable, "-m", "doorstop.cli.main", *args], cwd=cwd, capture_output=True, text=True)


def test_cli_analyze_console(demo_tree):
    result = run_doorstop(["analyze"], demo_tree)
    assert result.returncode == 0, result.stderr
    assert "[AMBIGUITY]" in result.stdout and "[CONFLICT]" in result.stdout


def test_cli_fail_on_gives_exit_code_1(demo_tree):
    assert run_doorstop(["analyze", "--checks", "ambiguity", "--fail-on", "any"], demo_tree).returncode == 1
    assert run_doorstop(["analyze", "--checks", "ambiguity", "--fail-on", "conflict"], demo_tree).returncode == 0


def test_cli_json_report_to_file(demo_tree):
    result = run_doorstop(["analyze", "REQ", "--format", "json", "-o", "report.json"], demo_tree)
    assert result.returncode == 0, result.stderr
    data = json.loads((demo_tree / "report.json").read_text(encoding="utf-8"))
    assert data["requirements"] == len(DEMO) and data["summary"]["ambiguity"] >= 5


def test_regular_doorstop_validation_still_works(demo_tree):
    """Regression: plain `doorstop` (validate the tree) must still pass with ReqSense installed."""
    assert run_doorstop([], demo_tree).returncode == 0


def run_in_process(args, cwd, monkeypatch):
    """Run `doorstop <args>` inside this Python process (so coverage can measure cli.py)."""
    from doorstop.cli import main as doorstop_main

    monkeypatch.chdir(cwd)
    try:
        doorstop_main.main(args)
        return 0
    except SystemExit as exc:
        return exc.code


def test_cli_in_process_all_formats(demo_tree, monkeypatch, capsys):
    assert run_in_process(["analyze"], demo_tree, monkeypatch) == 0
    assert "[CONFLICT]" in capsys.readouterr().out
    assert run_in_process(["analyze", "REQ", "--format", "md", "-o", "r.md"], demo_tree, monkeypatch) == 0
    assert (demo_tree / "r.md").read_text(encoding="utf-8").startswith("# ReqSense")
    assert run_in_process(["analyze", "--checks", "ambiguity", "--fail-on", "any"], demo_tree, monkeypatch) == 1


def test_cli_in_process_with_ai_options(demo_tree, monkeypatch, capsys, fake_models):
    code = run_in_process(["analyze", "--embedder", "sbert", "--conflict-method", "sim+nli"], demo_tree, monkeypatch)
    assert code == 0
    out = capsys.readouterr().out
    assert "sim+nli" in out and "[CONFLICT]" in out


def test_cli_on_an_empty_project(tmp_path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert run_in_process(["analyze"], tmp_path, monkeypatch) == 0
    assert "0 issue(s) in 0 requirements" in capsys.readouterr().out