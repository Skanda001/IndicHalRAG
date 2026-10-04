"""
merge_annotations.py
=====================

Merges all 4 annotation exports into one
clean annotated_outputs.json file.

Run:
    python scripts/merge_annotations.py

Input:
    data/annotation/person1_export.json
    data/annotation/person2_export.json
    data/annotation/person3_export.json
    data/annotation/person4_export.json

Output:
    data/annotation/annotated_outputs.json
    data/annotation/annotation_stats.json
"""

import json
import os
from collections import Counter

ANNOTATION_DIR = "data/annotation"
OUTPUT_PATH = "data/annotation/annotated_outputs.json"
STATS_PATH = "data/annotation/annotation_stats.json"

EXPORT_FILES = [
    "person1_export.json",
    "person2_export.json",
    "person3_export.json",
    "person4_export.json",
]


# ── Load and merge all 4 exports ─────────────────────────────────────

print("Loading annotation exports...")

all_rows = []

for filename in EXPORT_FILES:

    path = os.path.join(ANNOTATION_DIR, filename)

    if not os.path.exists(path):
        print(f"  ⚠️ NOT FOUND: {path} — skipping")
        continue

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    print(f"  ✅ {filename}: {len(rows)} rows")

    for row in rows:

        # Your files are already in the required structure.
        # Just extract the fields we need.

        clean_row = {
            "id": row.get("item_id"),
            "question": row.get("question"),
            "passage": row.get("passage"),
            "gold_answer": row.get("gold_answer"),
            "gemini_response": row.get("gemini_response"),
            "sarvam_response": row.get("sarvam_response"),
            "gemini_label": row.get("gemini_label"),
            "sarvam_label": row.get("sarvam_label"),
        }

        all_rows.append(clean_row)


print(f"\nTotal annotated rows merged: {len(all_rows)}")


# ── Check for missing labels ─────────────────────────────────────────

missing_gemini = [
    r for r in all_rows
    if not r["gemini_label"]
]

missing_sarvam = [
    r for r in all_rows
    if not r["sarvam_label"]
]

if missing_gemini:
    print(f"⚠️ {len(missing_gemini)} rows missing Gemini label")

if missing_sarvam:
    print(f"⚠️ {len(missing_sarvam)} rows missing Sarvam label")


# ── Only keep rows with BOTH labels ──────────────────────────────────

complete_rows = [
    r for r in all_rows
    if r["gemini_label"] and r["sarvam_label"]
]

print(f"Complete rows (both labels): {len(complete_rows)}")


# ── Save merged output ────────────────────────────────────────────────

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    json.dump(
        complete_rows,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\n✅ Saved → {OUTPUT_PATH}")


# ── Calculate and save stats ─────────────────────────────────────────

gemini_counts = Counter(
    r["gemini_label"]
    for r in complete_rows
)

sarvam_counts = Counter(
    r["sarvam_label"]
    for r in complete_rows
)

total = len(complete_rows)


def pct(count, total):

    return round(
        count / total * 100,
        2
    ) if total > 0 else 0


stats = {

    "total_annotated": total,

    "gemini": {
        label: {
            "count": gemini_counts.get(label, 0),
            "percentage": pct(
                gemini_counts.get(label, 0),
                total
            )
        }

        for label in [
            "correct",
            "hallucinated",
            "partial",
            "refused"
        ]
    },

    "sarvam": {
        label: {
            "count": sarvam_counts.get(label, 0),
            "percentage": pct(
                sarvam_counts.get(label, 0),
                total
            )
        }

        for label in [
            "correct",
            "hallucinated",
            "partial",
            "refused"
        ]
    }
}


with open(STATS_PATH, "w", encoding="utf-8") as f:

    json.dump(
        stats,
        f,
        ensure_ascii=False,
        indent=2
    )


# ── Print summary ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ANNOTATION STATISTICS")
print("=" * 60)

print(f"\nTotal questions annotated: {total}")

print("\nGEMINI LABELS:")

for label in [
    "correct",
    "hallucinated",
    "partial",
    "refused"
]:

    c = gemini_counts.get(label, 0)
    p = pct(c, total)

    bar = "█" * int(p // 2)

    print(
        f"  {label:15} "
        f"{c:4d}  "
        f"({p:5.1f}%)  "
        f"{bar}"
    )


print("\nSARVAM LABELS:")

for label in [
    "correct",
    "hallucinated",
    "partial",
    "refused"
]:

    c = sarvam_counts.get(label, 0)
    p = pct(c, total)

    bar = "█" * int(p // 2)

    print(
        f"  {label:15} "
        f"{c:4d}  "
        f"({p:5.1f}%)  "
        f"{bar}"
    )


print(f"\n✅ Stats saved → {STATS_PATH}")

print("=" * 60)

print("\nNEXT STEP: python scripts/phase1_analysis.py")
