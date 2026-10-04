"""
prepare_annotation.py
======================

Converts kannada_llm_outputs.json into FIVE files:

1. data/annotation/person1_tasks1_125.json
   → Person 1 (You) — questions 1-125

2. data/annotation/person2_tasks126_250.json
   → Person 2 (Friend) — questions 126-250

3. data/annotation/person3_tasks251_375.json
   → Person 3 (Friend) — questions 251-375

4. data/annotation/person4_tasks376_495.json
   → Person 4 (Friend) — questions 376-495

5. data/annotation/overlap_50.json
   → Same 50 questions that ALL 4 annotators label
   → Used to calculate Inter-Annotator Agreement (Kappa)

Run this ONCE before starting Label Studio.
Each person imports their assigned JSON file into their own Label Studio project.
"""

import json
import random
import os

INPUT_PATH   = "data/kannada_llm_outputs.json"
OUTPUT_DIR   = "data/annotation"

OVERLAP_COUNT = 50
RANDOM_SEED   = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ── Load generated responses ──────────────────────────────────────────
print("Loading generated responses...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter out any rows that still have ERROR responses
clean = []
errors = []
for row in data:
    g = row.get("gemini_response", "")
    s = row.get("sarvam_response", "")
    if str(g).startswith("ERROR") or str(s).startswith("ERROR"):
        errors.append(row["id"])
    else:
        clean.append(row)

print(f"Total rows        : {len(data)}")
print(f"Clean rows        : {len(clean)}")
print(f"Error rows skipped: {len(errors)}")
if errors:
    print(f"Error IDs         : {errors}")

# ── Helper function to convert rows to Label Studio tasks ──────────────
def rows_to_tasks(rows):
    """Convert list of rows to Label Studio task format."""
    tasks = []
    for row in rows:
        task = {
            "data": {
                "item_id":         row["id"],
                "question":        row["question"],
                "passage":         row.get("top_passage", ""),
                "gold_answer":     row.get("gold_answer", ""),
                "gemini_response": row.get("gemini_response", ""),
                "sarvam_response": row.get("sarvam_response", ""),
            }
        }
        tasks.append(task)
    return tasks

# ── Split into 4 equal chunks for 4 annotators ─────────────────────────
print("\n--- Splitting 495 questions into 4 annotators ---")

total = len(clean)
chunk_size = total // 4

person1_rows = clean[0           : chunk_size]
person2_rows = clean[chunk_size  : chunk_size*2]
person3_rows = clean[chunk_size*2 : chunk_size*3]
person4_rows = clean[chunk_size*3 : ]

print(f"Person 1: {len(person1_rows)} questions (IDs {person1_rows[0]['id']}-{person1_rows[-1]['id']})")
print(f"Person 2: {len(person2_rows)} questions")
print(f"Person 3: {len(person3_rows)} questions")
print(f"Person 4: {len(person4_rows)} questions")

# ── Convert to Label Studio format and save ──────────────────────────────
splits = {
    "person1_tasks1_125.json":   person1_rows,
    "person2_tasks126_250.json": person2_rows,
    "person3_tasks251_375.json": person3_rows,
    "person4_tasks376_495.json": person4_rows,
}

print("\n--- Saving person files ---")
for filename, rows in splits.items():
    path = os.path.join(OUTPUT_DIR, filename)
    tasks = rows_to_tasks(rows)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"✅ {filename}: {len(tasks)} tasks → {path}")

# ── Extract 50 overlap questions for Inter-Annotator Agreement ──────────
print("\n--- Creating overlap set (all 4 annotators) ---")
overlap_rows = random.sample(clean, min(OVERLAP_COUNT, len(clean)))
overlap_tasks = rows_to_tasks(overlap_rows)

overlap_path = os.path.join(OUTPUT_DIR, "overlap_50.json")
with open(overlap_path, "w", encoding="utf-8") as f:
    json.dump(overlap_tasks, f, ensure_ascii=False, indent=2)

print(f"✅ overlap_50.json: {len(overlap_tasks)} tasks → {overlap_path}")
print(f"   These 50 questions should be annotated by ALL 4 people")
print(f"   Used to calculate Cohen's Kappa (Inter-Annotator Agreement)")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total questions prepared: {len(clean)}")
print(f"Split into 4 files: ~{total//4} questions per person")
print(f"Overlap for IAA: 50 questions (all 4 people annotate)")
print()
print("📁 Files created in data/annotation/:")
print("   ✅ person1_tasks1_125.json")
print("   ✅ person2_tasks126_250.json")
print("   ✅ person3_tasks251_375.json")
print("   ✅ person4_tasks376_495.json")
print("   ✅ overlap_50.json")
print()
print("📋 NEXT STEPS:")
print("   1. Go to https://app.heartex.com")
print("   2. Sign up (free)")
print("   3. Create 5 projects:")
print("      - Annotation - Person 1")
print("      - Annotation - Person 2")
print("      - Annotation - Person 3")
print("      - Annotation - Person 4")
print("      - IAA Overlap - All 4")
print("   4. Upload corresponding JSON to each project")
print("   5. Paste XML config in each project's Labeling Interface")
print("   6. Invite friends to their projects")
print("="*70)
