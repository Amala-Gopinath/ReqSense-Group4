"""Convert NLI-RE-Datasets.xlsx (Fazelnia et al. 2024, Zenodo 11000349) into simple CSV files.

Creates in the output folder:
  defects.csv         id, text, label             (task 2: 131 requirements, 6 defect classes)
  conflict_reqs.csv   id, text, dataset           (task 3: requirements of the 7 datasets)
  conflict_pairs.csv  id1, id2, dataset           (task 3: the known conflicting pairs)

Usage:  python tools/prepare_dataset.py data/NLI-RE-Datasets.xlsx data
"""

import csv
import re
import sys
from pathlib import Path

import openpyxl

LABELS = {  # dataset label -> our defect class name (same as ambiguity.py)
    "ambiguous": "ambiguous",
    "optional": "optional",
    "non-atomic": "non_atomic",
    "non-measurable": "non_measurable",
    "uncertain": "uncertain",
    "directive": "directive",
}


def read_sheet(workbook, name):
    """Return the rows of a sheet as a list of dicts (column name -> value)."""
    rows = list(workbook[name].iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else f"col{i}" for i, h in enumerate(rows[0])]
    return [dict(zip(header, r)) for r in rows[1:] if any(v not in (None, "") for v in r)]


def clean(text):
    """Tidy spaces and line breaks."""
    return " ".join(str(text).split())


def key(text):
    """Text reduced to letters and digits, used to match the same requirement in two sheets."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def num(value):
    """'12.0' -> '12'"""
    return str(int(float(value)))


def main(xlsx, out_dir):
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Task 2: defects ----------------
    defects = []
    for i, row in enumerate(read_sheet(wb, "task2-raw-data"), start=1):
        label = LABELS[str(row["Class"]).strip().lower()]
        defects.append({"id": f"D{i:03d}", "text": clean(row["Requirement"]), "label": label})

    # ---------------- Task 3: conflicts ----------------
    reqs, pairs = [], []

    def add_dataset(name, texts_with_ids, conflict_rows, get_pair):
        """texts_with_ids: [(original_id, text)]; get_pair(row) -> (id_or_text_1, id_or_text_2, by_text)"""
        id_of_key, id_of_orig = {}, {}
        for n, (orig, text) in enumerate(texts_with_ids, start=1):
            rid = f"{name}-{n:03d}"
            reqs.append({"id": rid, "text": clean(text), "dataset": name})
            id_of_key.setdefault(key(text), rid)
            if orig is not None:
                id_of_orig[orig] = rid
        missing, seen = 0, set()
        for row in conflict_rows:
            a, b, by_text = get_pair(row)
            ida = id_of_key.get(key(a)) if by_text else id_of_orig.get(a)
            idb = id_of_key.get(key(b)) if by_text else id_of_orig.get(b)
            if not (ida and idb and ida != idb):
                missing += 1
            elif frozenset((ida, idb)) not in seen:  # skip pairs listed twice
                seen.add(frozenset((ida, idb)))
                pairs.append({"id1": ida, "id2": idb, "dataset": name})
        print(f"  {name:8s} {len(texts_with_ids):4d} requirements, {len(seen):3d} conflict pairs"
              + (f" ({missing} could not be matched)" if missing else ""))

    print("Conflict datasets:")
    promise_conflicts = read_sheet(wb, "task3-conflicts-p2-p6-p10")
    for p in (2, 6, 10):
        sheet = read_sheet(wb, f"task3-project{p}")
        rows = [r for r in promise_conflicts if num(r["project-ID"]) == str(p)]
        add_dataset(f"P{p}", [(None, r["RequirementText"]) for r in sheet], rows,
                    lambda r: (r["Requirement1"], r["Requirement2"], True))

    broker = read_sheet(wb, "task3-Broker-All")
    add_dataset("Broker", [(None, r["Requirements"]) for r in broker], read_sheet(wb, "task3-conflicts-Broker"),
                lambda r: (r["Requirement1"], r["Requirement2"], True))

    for name, sheet, conf in [("ETCS", "task3-ETCS-GOLD", "task3-ETCS-GOLD-Conflict"),
                              ("Library", "task3-Library-Gold", "task3-Library-Gold-Conflict"),
                              ("Coffee", "task3-Coffee-GOLD", "task3-Coffee-GOLD-Conflict")]:
        gold = [r for r in read_sheet(wb, sheet) if r.get("Requirements")]
        add_dataset(name, [(num(r["ID"]), r["Requirements"]) for r in gold], read_sheet(wb, conf),
                    lambda r: (num(r["R1"]), num(r["R2"]), False))

    # ---------------- write CSV files ----------------
    for filename, data, cols in [("defects.csv", defects, ["id", "text", "label"]),
                                 ("conflict_reqs.csv", reqs, ["id", "text", "dataset"]),
                                 ("conflict_pairs.csv", pairs, ["id1", "id2", "dataset"])]:
        with open(out / filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            writer.writerows(data)
        print(f"wrote {out / filename} ({len(data)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/prepare_dataset.py <NLI-RE-Datasets.xlsx> <output folder>")
    main(sys.argv[1], sys.argv[2])