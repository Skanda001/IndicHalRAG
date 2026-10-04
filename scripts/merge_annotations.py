"""
merge_annotations.py
=====================

Merges all 4 Label Studio annotation exports into one
clean annotated_outputs.json file.

Run:
    python scripts/merge_annotations.py

Input (Label Studio JSON export format):
    data/annotation/person1_export.json
    data/annotation/person2_export.json
    data/annotation/person3_export.json
    data/annotation/person4_export.json

Output:
    data/annotation/annotated_outputs.json
    data/annotation/annotation_stats.json

HOW TO EXPORT FROM LABEL STUDIO
---------------------------------
In each project: Export → JSON → Download
Rename the file to person1_export.json, person2_export.json, etc.
Place all 4 files in data/annotation/
"""

import json
import os
import sys
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ANNOTATION_DIR = "data/annotation"
OUTPUT_PATH    = "data/annotation/annotated_outputs.json"
STATS_PATH     = "data/annotation/annotation_stats.json"

EXPORT_FILES = [
    "person1_export.json",
    "person2_export.json",
    "person3_export.json",
    "person4_export.json",
]


def extract_label(result_list, from_name):
    """Extract a single-choice label from Label Studio result list.
    
    Label Studio export format:
    "result": [
        {
            "from_name": "gemini_label",
            "to_name":   "gemini_text",
            "type":      "choices",
            "value":     {"choices": ["correct"]}
        },
        ...
    ]
    """
    for item in result_list:
        if item.get("from_name") == from_name:
            choices = item.get("value", {}).get("choices", [])
            if choices:
                return choices[0]
    return None


def parse_label_studio_export(rows):
    """Parse Label Studio JSON export into flat annotated rows."""
    parsed = []
    for task in rows:
        data = task.get("data", {})

        # Label Studio wraps annotations in a list; take the first completed one
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        # Pick the most recent completed annotation
        annotation = None
        for ann in annotations:
            if not ann.get("was_cancelled", False):
                annotation = ann
                break

        if not annotation:
            continue

        result = annotation.get("result", [])

        gemini_label = extract_label(result, "gemini_label")
        sarvam_label = extract_label(result, "sarvam_label")

        parsed.append({
            "id":              data.get("item_id"),
            "question":        data.get("question"),
            "passage":         data.get("passage"),
            "gold_answer":     data.get("gold_answer"),
            "gemini_response": data.get("gemini_response"),
            "sarvam_response": data.get("sarvam_response"),
            "gemini_label":    gemini_label,
            "sarvam_label":    sarvam_label,
        })

    return parsed


# ── Load and merge all 4 exports ─────────────────────────────────────
print("Loading annotation exports...")

all_rows = []

for filename in EXPORT_FILES:
    path = os.path.join(ANNOTATION_DIR, filename)

    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        continue

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed = parse_label_studio_export(raw)
    print(f"  {filename}: {len(raw)} tasks → {len(parsed)} annotated")
    all_rows.extend(parsed)

print(f"\nTotal annotated rows merged: {len(all_rows)}")


# ── Check for missing labels ──────────────────────────────────────────
missing_gemini = [r for r in all_rows if not r["gemini_label"]]
missing_sarvam = [r for r in all_rows if not r["sarvam_label"]]

if missing_gemini:
    print(f"WARNING: {len(missing_gemini)} rows missing Gemini label")
if missing_sarvam:
    print(f"WARNING: {len(missing_sarvam)} rows missing Sarvam label")


# ── Only keep rows with BOTH labels ──────────────────────────────────
complete_rows = [r for r in all_rows if r["gemini_label"] and r["sarvam_label"]]
print(f"Complete rows (both labels): {len(complete_rows)}")

if not complete_rows:
    print("\nERROR: No complete rows found!")
    print("Make sure you have exported from Label Studio and placed the files in data/annotation/")
    sys.exit(1)


# ── Save merged output ────────────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(complete_rows, f, ensure_ascii=False, indent=2)

print(f"\nSaved --> {OUTPUT_PATH}")


# ── Calculate and save stats ──────────────────────────────────────────
LABELS = ["correct", "hallucinated", "partial", "refused"]
total = len(complete_rows)

gemini_counts = Counter(r["gemini_label"] for r in complete_rows)
sarvam_counts = Counter(r["sarvam_label"] for r in complete_rows)


def pct(count, total):
    return round(count / total * 100, 2) if total > 0 else 0


stats = {
    "total_annotated": total,
    "gemini": {
        label: {"count": gemini_counts.get(label, 0),
                "percentage": pct(gemini_counts.get(label, 0), total)}
        for label in LABELS
    },
    "sarvam": {
        label: {"count": sarvam_counts.get(label, 0),
                "percentage": pct(sarvam_counts.get(label, 0), total)}
        for label in LABELS
    }
}

with open(STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)


# ── Print summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANNOTATION STATISTICS")
print("=" * 60)
print(f"\nTotal questions annotated: {total}")

for model, counts in [("GEMINI", gemini_counts), ("SARVAM", sarvam_counts)]:
    print(f"\n{model} LABELS:")
    for label in LABELS:
        c = counts.get(label, 0)
        p = pct(c, total)
        bar = "#" * int(p // 2)
        print(f"  {label:15} {c:4d}  ({p:5.1f}%)  {bar}")

print(f"\nStats saved --> {STATS_PATH}")
print("=" * 60)
print("\nNEXT STEP: python scripts/phase1_analysis.py")
