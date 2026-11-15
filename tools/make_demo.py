"""Create the ReqSense demo project: a Doorstop tree with 16 requirements containing planted problems.

Usage (from C:\\projects\\doorstop):
  python tools/make_demo.py C:\\projects\\demo2
  cd C:\\projects\\demo2
  doorstop analyze

Planted problems: 6 ambiguity defects (REQ002-REQ007), 2 duplicates (REQ008/009, REQ010/011),
2 conflicts (REQ012/013, REQ014/015). REQ001 and REQ016 are clean.
"""

import os
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = [
    "The system shall lock a user account after 5 consecutive failed login attempts.",
    "The system shall respond to search queries quickly.",
    "The application may send a confirmation email to the customer if necessary.",
    "The user interface shall be user-friendly and intuitive.",
    "The system shall generate the monthly sales report and email it to the store manager.",
    "The export format shall be as shown in Figure 3.",
    "The system shall store relevant customer data and/or order data.",
    "The system shall encrypt all stored passwords using bcrypt.",
    "All stored passwords shall be encrypted by the system with bcrypt.",
    "The system shall display the order total including tax on the checkout page.",
    "The checkout page shall show the order total including all applicable taxes.",
    "The system shall allow guest users to place orders without creating an account.",
    "Guest users shall not be able to place orders without creating an account.",
    "The system shall retain audit logs for 7 years.",
    "The system shall delete audit logs after 90 days.",
    "Session tokens shall expire after 30 minutes of inactivity.",
]


def main(folder):
    from doorstop.core.builder import build

    root = Path(folder).resolve()
    if (root / "reqs").exists():
        sys.exit(f"{root} already contains a demo project - choose another folder")
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    os.chdir(root)  # Doorstop works relative to the current folder
    tree = build(cwd=str(root), root=str(root))
    doc = tree.create_document(str(root / "reqs"), "REQ")
    for text in REQUIREMENTS:
        item = doc.add_item()
        item.text = text
        item.save()
        print("added", item.uid, "-", text)
    print(f"\nDemo project ready in {root}. Next:  cd {root}  then  doorstop analyze")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/make_demo.py <new folder>")
    main(sys.argv[1])