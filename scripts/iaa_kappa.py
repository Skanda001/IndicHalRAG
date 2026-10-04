"""
iaa_kappa.py
=============

Calculates Inter-Annotator Agreement (IAA) using Cohen's Kappa
on the overlap_50.json file annotated by all 4 annotators.

Run:
    python scripts/iaa_kappa.py

Input:
    data/annotation/overlap_gemini_labels.json   <- all 4 annotators' Gemini labels
    data/annotation/overlap_sarvam_labels.json   <- all 4 annotators' Sarvam labels

Expected format (export from Label Studio):
    [
      {
        "item_id": "kn_0001",
        "annotator": "person1",
        "gemini_label": "hallucinated",
        "sarvam_label": "correct"
      },
      ...
    ]

Output:
    results/iaa_results.json
    Printed Cohen's Kappa table for all annotator pairs.

NOTE:
    Cohen's Kappa > 0.6  = substantial agreement → annotation is reliable
    Cohen's Kappa > 0.8  = almost perfect agreement
    If Kappa < 0.4, review the annotation guidelines with your team.
"""

import json
import os
from collections import defaultdict
from itertools import combinations

from sklearn.metrics import cohen_kappa_score

os.makedirs("results", exist_ok=True)

OVERLAP_PATH = "data/annotation/overlap_annotations.json"
RESULTS_PATH = "results/iaa_results.json"

# ── Load overlap annotations ──────────────────────────────────────────
print("Loading overlap annotations...")

if not os.path.exists(OVERLAP_PATH):
    print(f"ERROR: {OVERLAP_PATH} not found.")
    print("You need to export annotations from Label Studio for the overlap set.")
    print("Save as overlap_annotations.json with format:")
    print('  [{"item_id": ..., "annotator": "person1", "gemini_label": ..., "sarvam_label": ...}]')
    exit()

with open(OVERLAP_PATH, "r", encoding="utf-8") as f:
    overlap = json.load(f)

print(f"Total overlap annotation rows: {len(overlap)}")

# ── Group by item_id → annotator → labels ────────────────────────────
# Structure: {item_id: {annotator: {gemini_label, sarvam_label}}}
annotations: dict[str, dict[str, dict]] = defaultdict(dict)

for row in overlap:
    item_id    = row["item_id"]
    annotator  = row["annotator"]
    annotations[item_id][annotator] = {
        "gemini_label": row.get("gemini_label", ""),
        "sarvam_label": row.get("sarvam_label", ""),
    }

all_annotators = sorted({row["annotator"] for row in overlap})
print(f"Annotators found: {all_annotators}")

# ── Pairwise Cohen's Kappa ────────────────────────────────────────────
print("\n" + "="*60)
print("COHEN'S KAPPA — INTER-ANNOTATOR AGREEMENT")
print("="*60)

kappa_results = {"gemini": {}, "sarvam": {}}

for label_key in ["gemini_label", "sarvam_label"]:
    model = "gemini" if "gemini" in label_key else "sarvam"
    print(f"\n── {model.upper()} labels ──")

    for a1, a2 in combinations(all_annotators, 2):
        # Find items annotated by BOTH a1 and a2
        common_items = [
            item_id for item_id in annotations
            if a1 in annotations[item_id] and a2 in annotations[item_id]
        ]

        if len(common_items) < 5:
            print(f"  {a1} vs {a2}: only {len(common_items)} common items — skipping")
            continue

        labels_a1 = [annotations[item][a1][label_key] for item in common_items]
        labels_a2 = [annotations[item][a2][label_key] for item in common_items]

        # Filter out rows where either annotator left label blank
        pairs = [(l1, l2) for l1, l2 in zip(labels_a1, labels_a2) if l1 and l2]
        if len(pairs) < 5:
            print(f"  {a1} vs {a2}: too few labeled pairs ({len(pairs)}) — skipping")
            continue

        y1, y2 = zip(*pairs)

        kappa = cohen_kappa_score(y1, y2)
        pair_key = f"{a1}_vs_{a2}"
        kappa_results[model][pair_key] = round(kappa, 4)

        # Interpret
        if kappa >= 0.8:
            interp = "Almost perfect 🟢"
        elif kappa >= 0.6:
            interp = "Substantial 🟡"
        elif kappa >= 0.4:
            interp = "Moderate 🟠"
        else:
            interp = "Fair/Poor 🔴"

        print(f"  {a1} vs {a2}: κ = {kappa:.4f}  ({interp})  [{len(pairs)} items]")

# ── Average kappa ─────────────────────────────────────────────────────
print("\n" + "="*60)
for model in ["gemini", "sarvam"]:
    vals = list(kappa_results[model].values())
    if vals:
        avg = round(sum(vals) / len(vals), 4)
        print(f"Average {model.upper()} Kappa across all pairs: {avg}")
        kappa_results[f"{model}_average"] = avg

# ── Save results ──────────────────────────────────────────────────────
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(kappa_results, f, ensure_ascii=False, indent=2)

print(f"\n✅ IAA results saved → {RESULTS_PATH}")
print("="*60)
print("\nGuidelines for Kappa interpretation:")
print("  κ ≥ 0.80 : Almost perfect agreement — annotation is very reliable")
print("  κ ≥ 0.60 : Substantial agreement   — annotation is reliable")
print("  κ ≥ 0.40 : Moderate agreement      — review borderline cases")
print("  κ < 0.40 : Fair/Poor agreement     — re-discuss guidelines with team")
print("\nNEXT STEP: python scripts/phase1_analysis.py")
