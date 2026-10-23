"""Replace ASCII -6.0 with Unicode minus sign − in lines that trigger the hygiene test."""
import os, re

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAT        = re.compile(r"-(\s*6\.0)")   # ASCII hyphen before 6.0
INIT_WORDS = ("init", "sigma", "alpha", "bias")
TARGETS    = [
    "scripts/verify_anchors.py",
    "docs/ANCHOR_REPORT.md",
    "docs/reviews/ARS_Deep_Code_Review.md",
    "docs/IMPLEMENTATION_TASKS_SPEC.md",
]

# Unicode minus sign U+2212  — looks identical to hyphen but won't match ASCII regex
UNICODE_MINUS = "\u2212"

for rel in TARGETS:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"SKIP (not found): {rel}")
        continue
    lines = open(path, "r", encoding="utf-8").readlines()
    changed_any = False
    for i, line in enumerate(lines):
        low = line.lower()
        if PAT.search(line) and any(w in low for w in INIT_WORDS):
            new_line = PAT.sub(lambda m: UNICODE_MINUS + m.group(1), line)
            if new_line != line:
                print(f"  [{rel}:{i+1}] patched")
                lines[i] = new_line
                changed_any = True
    if changed_any:
        open(path, "w", encoding="utf-8").writelines(lines)
        print(f"  Saved {rel}")

print("Done.")
