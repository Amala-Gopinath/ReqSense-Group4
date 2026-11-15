"""Turn the dataset CSV files into real Doorstop trees (one Doorstop document per dataset).

Usage (from C:\\projects\\doorstop, after tools/prepare_dataset.py):
  python tools/csv_to_tree.py C:\\projects\\dataset-tree                    # everything (~600 items)
  python tools/csv_to_tree.py C:\\projects\\coffee-tree --only Coffee       # just one dataset (fast)

Then:
  cd C:\\projects\\dataset-tree
  doorstop analyze COFFEE --checks conflict --conflict-method sim+nli

Documents created: a top document DATA (one information item), and below it DEF (the 131 labelled
defect requirements) and one per conflict dataset: P2, P6, P10, BROKER, ETCS, LIBRARY, COFFEE.
Each item keeps its dataset id ("source_id") and, for DEF, its expert label ("label"),
so the original data can always be traced back.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prefix_for(name):
    """Dataset name -> Doorstop document prefix, e.g. 'Coffee' -> 'COFFEE'."""
    return re.sub(r"[^A-Za-z0-9]", "", name).upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="new folder for the Doorstop project")
    parser.add_argument("--data", default="data", help="folder with defects.csv and conflict_reqs.csv")
    parser.add_argument("--only", nargs="+", help="only these datasets, e.g. Coffee ETCS DEF")
    args = parser.parse_args()

    data = Path(args.data).resolve()
    groups = defaultdict(list)  # prefix -> [(source_id, text, label)]
    for row in read_csv(data / "defects.csv"):
        groups["DEF"].append((row["id"], row["text"], row["label"]))
    for row in read_csv(data / "conflict_reqs.csv"):
        groups[prefix_for(row["dataset"])].append((row["id"], row["text"], None))
    if args.only:
        wanted = {prefix_for(name) for name in args.only}
        groups = {p: items for p, items in groups.items() if p in wanted}
        if not groups:
            sys.exit(f"no dataset called {args.only}; choose from DEF P2 P6 P10 Broker ETCS Library Coffee")

    root = Path(args.folder).resolve()
    if any((root / p.lower()).exists() for p in groups):
        sys.exit(f"{root} already contains these documents - choose another folder")
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    os.chdir(root)  # Doorstop works relative to the current folder

    from doorstop.core.builder import build

    tree = build(cwd=str(root), root=str(root))
    # A Doorstop project is a tree: one top document DATA, each dataset is a child document of it.
    top = tree.create_document(str(root / "data"), "DATA")
    info = top.add_item()
    info.text = "Datasets of Fazelnia et al. (2024), NLI-RE-Datasets, Zenodo record 11000349 (CC BY 4.0)."
    info.normative = False  # an information item, not a requirement (ReqSense skips it)
    info.save()
    for prefix, items in groups.items():
        doc = tree.create_document(str(root / prefix.lower()), prefix, parent="DATA")
        for source_id, text, label in items:
            item = doc.add_item(reorder=False)  # reorder=False: much faster for many items
            item.text = text
            item.derived = True  # independent requirement: no link to a parent item is expected
            item.set("source_id", source_id)
            if label:
                item.set("label", label)
            item.save()
        print(f"created document {prefix:8s} with {len(items):4d} requirements")
    print(f"\nDoorstop project ready in {root}")
    print(f"Try:  cd {root}   then   doorstop analyze {next(iter(groups))}")


if __name__ == "__main__":
    main()