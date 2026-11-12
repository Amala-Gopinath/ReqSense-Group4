"""Turn a list of Findings into a report: console text, Markdown or JSON."""

import json

CHECKS = ["ambiguity", "redundancy", "conflict"]


def summary(findings):
    """How many findings per check, e.g. {'ambiguity': 7, 'redundancy': 2, 'conflict': 2}."""
    return {c: sum(1 for f in findings if f.check == c) for c in CHECKS}


def render(findings, requirements, fmt="console", stats=None):
    """Return the report as a string. fmt = 'console', 'md' or 'json'."""
    stats = stats or {}
    text_of = dict(requirements)  # uid -> requirement text
    counts = summary(findings)

    if fmt == "json":
        return json.dumps(
            {
                "summary": counts,
                "requirements": len(requirements),
                "stats": stats,
                "findings": [f.to_dict() for f in findings],
            },
            indent=2,
        )

    if fmt == "md":
        lines = [
            "# ReqSense analysis report",
            "",
            f"{len(requirements)} requirements analysed. " + ", ".join(f"{k}: {v}" for k, v in stats.items()),
            "",
            "| Check | Findings |",
            "|---|---|",
        ]
        lines += [f"| {c} | {n} |" for c, n in counts.items()]
        for check in CHECKS:
            group = [f for f in findings if f.check == check]
            if not group:
                continue
            lines += ["", f"## {check.capitalize()} ({len(group)})", ""]
            lines += ["| Item(s) | Defect | Score | Technique | Rationale | Suggestion |", "|---|---|---|---|---|---|"]
            for f in group:
                cells = [", ".join(f.uids), f.defect, f"{f.score:.2f}", f.stage, f.rationale, f.suggestion]
                lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
        return "\n".join(lines) + "\n"

    # console (default)
    lines = []
    for f in findings:
        lines.append(f"[{f.check.upper()}] {' <-> '.join(f.uids)}: {f.defect} [{f.score:.2f}, {f.stage}] {f.rationale}")
        for uid in f.uids:
            lines.append(f"    {uid}: {text_of.get(uid, '')}")
        lines.append(f"    suggestion: {f.suggestion}")
    total = ", ".join(f"{n} {c}" for c, n in counts.items())
    lines.append(f"\nReqSense: {len(findings)} issue(s) in {len(requirements)} requirements ({total})")
    return "\n".join(lines) + "\n"