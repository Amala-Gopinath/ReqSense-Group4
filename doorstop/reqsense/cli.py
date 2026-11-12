"""The `doorstop analyze` command (ReqSense)."""

import os
import time

from doorstop import common
from doorstop.reqsense.ambiguity import detect_ambiguity
from doorstop.reqsense.conflict import detect_conflicts
from doorstop.reqsense.redundancy import detect_redundancy
from doorstop.reqsense.report import render


def add_parser(subs, shared):
    """Tell Doorstop that a new command called 'analyze' exists, and what options it takes."""
    sub = subs.add_parser(
        "analyze",
        help="check requirement text for ambiguity, redundancy and conflicts",
        **shared,
    )
    sub.add_argument(
        "prefix",
        nargs="?",
        default="all",
        help="document to analyze, e.g. REQ (default: all documents)",
    )
    sub.add_argument(
        "--checks",
        default="ambiguity,redundancy,conflict",
        help="which checks to run, comma separated: ambiguity,redundancy,conflict",
    )
    sub.add_argument(
        "--embedder",
        choices=["tfidf", "sbert"],
        default="tfidf",
        help="similarity technique: tfidf (baseline) or sbert (AI model)",
    )
    sub.add_argument(
        "--redundancy-threshold",
        type=float,
        default=None,
        help="similarity (0-1) above which two requirements count as duplicates",
    )
    sub.add_argument(
        "--conflict-method",
        choices=["heuristic", "nli", "sim+nli"],
        default="heuristic",
        help="conflict technique: heuristic (baseline), nli (all pairs), sim+nli (tiered)",
    )
    sub.add_argument(
        "--format",
        dest="report_format",
        choices=["console", "md", "json"],
        default="console",
        help="report format",
    )
    sub.add_argument("-o", "--output", help="save the report to this file instead of printing it")
    sub.add_argument(
        "--fail-on",
        default="none",
        help="for CI: exit with an error if issues are found. none (default), any, "
        "or a list of checks, e.g. conflict or ambiguity,conflict",
    )
    return sub


def get_requirements(tree, prefix):
    """Read requirements from Doorstop and return a list of (uid, text) pairs."""
    if prefix == "all":
        items = [item for document in tree for item in document]
    else:
        items = list(tree.find_document(prefix))

    requirements = []
    for item in items:
        # skip deleted (inactive) items, headings and empty items
        if not item.active or not item.normative or item.heading:
            continue
        text = " ".join(str(item.text).split())  # tidy up spaces and line breaks
        if text:
            requirements.append((str(item.uid), text))
    return requirements


def run_analyze(args, cwd, error, catch=True):
    """Called by Doorstop when the user types `doorstop analyze`."""
    from doorstop.core.builder import build

    try:
        tree = build(cwd=cwd, root=args.project)  # find the Doorstop project
        tree.load()  # load all documents and items
    except common.DoorstopError as exc:
        error(str(exc))
        return False
    return run(args, cwd, error, tree)


def should_fail(fail_on, findings):
    """Decide if the command should return an error code (used by CI pipelines)."""
    if fail_on == "none":
        return False
    if fail_on == "any":
        return len(findings) > 0
    wanted = [c.strip() for c in fail_on.split(",")]
    return any(f.check in wanted for f in findings)


def run(args, cwd, error, tree):
    """Run the analysis. Return True if OK, False to make the command fail."""
    start = time.perf_counter()
    requirements = get_requirements(tree, args.prefix)
    checks = [c.strip() for c in args.checks.split(",")]

    findings = []
    if "ambiguity" in checks:
        findings += detect_ambiguity(requirements)
    if "redundancy" in checks:
        findings += detect_redundancy(requirements, args.embedder, args.redundancy_threshold)
    if "conflict" in checks:
        embedder = "sbert" if args.conflict_method == "sim+nli" else args.embedder
        findings += detect_conflicts(requirements, args.conflict_method, embedder)

    # A pair that CONTRADICTS each other is not a duplicate, even if the words are similar.
    conflict_pairs = {frozenset(f.uids) for f in findings if f.check == "conflict"}
    findings = [f for f in findings if not (f.check == "redundancy" and frozenset(f.uids) in conflict_pairs)]

    stats = {
        "seconds": round(time.perf_counter() - start, 2),
        "embedder": args.embedder,
        "conflict method": args.conflict_method,
    }
    report = render(findings, requirements, args.report_format, stats)

    if args.output:
        path = os.path.join(cwd, args.output)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"ReqSense: report with {len(findings)} issue(s) saved to {path}")
    else:
        print(report, end="")

    return not should_fail(args.fail_on, findings)  # False -> Doorstop exits with code 1