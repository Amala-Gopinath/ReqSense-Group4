"""The `doorstop analyze` command (ReqSense)."""

from doorstop import common
from doorstop.reqsense.ambiguity import detect_ambiguity


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
        default="ambiguity",
        help="which checks to run, comma separated (for now: ambiguity)",
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


def run(args, cwd, error, tree):
    """Run the analysis. Return True if OK, False to make the command fail."""
    requirements = get_requirements(tree, args.prefix)
    checks = [c.strip() for c in args.checks.split(",")]

    findings = []
    if "ambiguity" in checks:
        findings += detect_ambiguity(requirements)

    # print the results
    text_of = dict(requirements)  # uid -> text, to show the requirement next to each finding
    for f in findings:
        print(f"{' <-> '.join(f.uids)}: {f.defect} [{f.score:.2f}] {f.rationale}")
        for uid in f.uids:
            print(f"    {uid}: {text_of[uid]}")
        print(f"    suggestion: {f.suggestion}")
    print(f"\nReqSense: {len(findings)} issue(s) in {len(requirements)} requirements")
    return True