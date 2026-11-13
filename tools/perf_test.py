"""Performance test (RQ3): runtime and memory of each ReqSense check on 1,000 / 5,000 / 10,000 requirements.

Usage (from C:\\projects\\doorstop):
  python tools/perf_test.py                       # fast checks only (rules, TF-IDF, heuristic)
  python tools/perf_test.py --ai                  # also Sentence-BERT and the tiered sim+NLI (slower)
  python tools/perf_test.py --sizes 1000 --tree   # also an end-to-end `doorstop analyze` on a real tree

Synthetic requirements are generated from templates (fixed random seed, so every run uses the same data).
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from doorstop.reqsense import ambiguity, conflict, redundancy

ACTORS = ["The system", "The web portal", "The mobile app", "The admin console", "The billing service",
          "The scheduler", "The report module", "The API gateway", "The audit service", "The payment service",
          "The notification service", "The search engine", "The user", "The administrator", "The operator"]
ACTIONS = ["shall display", "shall store", "shall export", "shall validate", "shall send", "shall encrypt",
           "shall archive", "shall record", "shall delete", "shall update", "shall generate", "shall print",
           "shall upload", "shall approve", "shall reject", "shall calculate", "shall synchronise", "shall lock",
           "shall not display", "shall not store", "may export", "should send", "shall quickly generate"]
OBJECTS = ["the invoice", "the order history", "the patient record", "the login attempt", "the error message",
           "the monthly report", "the payment receipt", "the user profile", "the shipment status", "the backup file",
           "the audit log", "the price list", "the course schedule", "the flight booking", "the insurance claim",
           "the stock level", "the sensor reading", "the support ticket", "the contract", "the timesheet",
           "the customer address", "the delivery note", "the exam result", "the parking permit", "the tax form"]
OWNERS = ["of each customer", "of the selected supplier", "of every branch", "of the current user",
          "of all warehouses", "of the clinic", "of the project", "of the vehicle", "of the student", "of the account"]
CONDITIONS = ["within {n} seconds", "every {n} hours", "after {n} failed attempts", "for {n} years",
              "when the user logs out", "after each transaction", "using AES-256", "in PDF format",
              "to authorised users only", "before midnight", "if possible", "in a user-friendly way",
              "at least {n} times per day", "for up to {n} concurrent users"]


def synthetic(n, seed=42):
    """n template-based requirements (some deliberately vague, negated or near-duplicate)."""
    rnd = random.Random(seed)
    out = []
    for k in range(n):
        cond = rnd.choice(CONDITIONS).format(n=rnd.randint(1, 500))
        text = f"{rnd.choice(ACTORS)} {rnd.choice(ACTIONS)} {rnd.choice(OBJECTS)} {rnd.choice(OWNERS)} {cond}."
        out.append((f"S{k + 1:05d}", text))
    return out


def memory_mb():
    """Peak memory of this Python process so far (MB), if psutil is installed."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 2**20  # Windows: peak working set
    except ImportError:
        return float("nan")


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def build_tree(reqs, root):
    """Create a real Doorstop project (Git repo + YAML files) with these requirements."""
    from doorstop.core.builder import build

    subprocess.run(["git", "init", "-q", str(root)], check=True)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        doc = build(cwd=str(root), root=str(root)).create_document(str(root / "reqs"), "REQ", digits=5)
        for _, text in reqs:
            item = doc.add_item(reorder=False)
            item.text = text
            item.save()
    finally:
        os.chdir(cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--ai", action="store_true", help="also run Sentence-BERT and sim+NLI")
    parser.add_argument("--tree", action="store_true", help="also time `doorstop analyze` end-to-end on a real tree")
    args = parser.parse_args()

    checks = [
        ("F1 ambiguity (rules)", lambda r: ambiguity.detect_ambiguity(r)),
        ("F2 redundancy (TF-IDF)", lambda r: redundancy.detect_redundancy(r, "tfidf")),
        ("F3 conflict (heuristic)", lambda r: conflict.detect_conflicts(r, "heuristic")),
    ]
    if args.ai:
        checks += [
            ("F2 redundancy (SBERT)", lambda r: redundancy.detect_redundancy(r, "sbert")),
            ("F3 conflict (sim+NLI)", lambda r: conflict.detect_conflicts(r, "sim+nli", "sbert")),
        ]
        redundancy.embed(["warm up"], "sbert")  # load models once, so loading time is not counted per size
        conflict.load_nli()

    lines = [f"# ReqSense performance ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Python {sys.version.split()[0]} on {sys.platform}, CPU only.", "",
             "| Requirements | Check | Time (s) | Findings | Pairs checked by the model | Peak memory (MB) |",
             "|---|---|---|---|---|---|"]
    for n in args.sizes:
        reqs = synthetic(n)
        for name, fn in checks:
            findings, seconds = timed(lambda: fn(reqs))
            checked = f"{conflict.LAST_PAIRS_CHECKED:,} of {n * (n - 1) // 2:,}" if "conflict" in name else "-"
            row = f"| {n:,} | {name} | {seconds:.1f} | {len(findings)} | {checked} | {memory_mb():.0f} |"
            lines.append(row)
            print(row, flush=True)
        if args.tree:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "perf"
                _, build_s = timed(lambda: build_tree(reqs, root))
                cmd = [sys.executable, "-m", "doorstop.cli.main", "analyze", "--format", "json", "-o", "r.json"]
                result, run_s = timed(lambda: subprocess.run(cmd, cwd=root, capture_output=True, text=True))
                ok = "OK" if result.returncode == 0 else f"exit {result.returncode}"
                row = (f"| {n:,} | end-to-end `doorstop analyze` on a real tree (default checks) | {run_s:.1f} "
                       f"| {ok} | (creating the {n:,} YAML files took {build_s:.0f} s) | - |")
                lines.append(row)
                print(row, flush=True)

    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"performance_{datetime.now():%Y%m%d-%H%M}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    main()