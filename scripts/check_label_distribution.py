"""
check_label_distribution.py
=============================

Quick sanity check: print the label distribution of annotated_outputs.json.

Run:
    python scripts/check_label_distribution.py
"""

import json
from collections import Counter

ANNOTATION_PATH = "data/annotation/annotated_outputs.json"

with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

gemini_labels = Counter(r["gemini_label"] for r in data if r.get("gemini_label"))
sarvam_labels = Counter(r["sarvam_label"] for r in data if r.get("sarvam_label"))

total = len(data)

print("GEMINI LABEL DISTRIBUTION:")
for label, count in gemini_labels.items():
    print(f"  {label:15} {count:4d}  ({count/total*100:.1f}%)")

print("\nSARVAM LABEL DISTRIBUTION:")
for label, count in sarvam_labels.items():
    print(f"  {label:15} {count:4d}  ({count/total*100:.1f}%)")
