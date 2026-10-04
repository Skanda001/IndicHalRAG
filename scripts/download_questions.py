"""
download_questions.py
======================

Downloads IndicQA questions for KANNADA ONLY and caps at 500 questions.

CHANGES FROM ORIGINAL:
- Hindi removed completely.
- Added MAX_QUESTIONS = 500 cap (you had 1500, which is too many to annotate).
- Added stratified sampling: instead of just taking the first 500 questions
  (which might all come from the same 5-10 Wikipedia articles), we take a
  maximum of MAX_PER_CONTEXT questions from each unique context/article.
  This ensures your 500 questions cover a WIDE variety of topics,
  making your hallucination dataset more representative and your
  annotations more meaningful.
- Supports direct Hugging Face Hub download (avoids datasets 3.0+ loading script deprecation).
"""

import json
import os
import random
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from huggingface_hub import hf_hub_download

# ── Config ──────────────────────────────────────────────────────────────
OUTPUT_PATH      = "data/questions.json"
REPO_ID          = "ai4bharat/IndicQA"
FILE_NAME        = "data/indicqa.kn.json"
LANG_CODE        = "kn"
LANGUAGE         = "kannada"
MAX_QUESTIONS    = 500      # Final cap — change this if you want more/fewer
MAX_PER_CONTEXT  = 4        # Max questions taken from any single Wikipedia article
                            # Prevents dataset being dominated by 1-2 long articles
RANDOM_SEED      = 42       # Fixed seed → reproducible sampling every time you run

os.makedirs("data", exist_ok=True)
random.seed(RANDOM_SEED)

# ── Load dataset ─────────────────────────────────────────────────────────
print("Loading IndicQA dataset (Kannada only) from Hugging Face...")

try:
    # First try direct download of SQuAD JSON from Hugging Face repo
    data_file = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, repo_type="dataset")
    with open(data_file, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    raw_items = []
    for article in raw_json.get("data", []):
        for p in article.get("paragraphs", []):
            ctx = p.get("context", "").strip()
            for qa in p.get("qas", []):
                q_text = qa.get("question", "").strip()
                ans_list = qa.get("answers", [])
                gold = ans_list[0].get("text", "").strip() if ans_list else ""
                raw_items.append({
                    "id": str(qa.get("id", f"{LANG_CODE}_{len(raw_items)}")),
                    "question": q_text,
                    "context": ctx,
                    "gold_answer": gold,
                })
    print(f"Total questions in dataset: {len(raw_items)}")

except Exception as e:
    print(f"Falling back to load_dataset due to: {e}")
    from datasets import load_dataset
    dataset = load_dataset("ai4bharat/IndicQA", "indicqa.kn", split="test")
    raw_items = []
    for i, item in enumerate(dataset):
        answers = item.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        raw_items.append({
            "id": item.get("id", f"{LANG_CODE}_{i}"),
            "question": item.get("question", "").strip(),
            "context": item.get("context", "").strip(),
            "gold_answer": answer_texts[0].strip() if answer_texts else "",
        })

# ── Collect all valid questions ───────────────────────────────────────────
print("\nCollecting valid questions...")

# Group by unique context (each context = one Wikipedia article/passage)
context_groups: dict[str, list[dict]] = {}
skipped_empty = 0

for item in raw_items:
    question    = item.get("question", "").strip()
    context     = item.get("context", "").strip()
    gold_answer = item.get("gold_answer", "").strip()

    if not question or not context:
        skipped_empty += 1
        continue

    entry = {
        "id":           item.get("id"),
        "language":     LANGUAGE,
        "lang_code":    LANG_CODE,
        "question":     question,
        "gold_context": context,
        "gold_answer":  gold_answer,
    }

    if context not in context_groups:
        context_groups[context] = []
    context_groups[context].append(entry)

print(f"Valid questions found   : {sum(len(v) for v in context_groups.values())}")
print(f"Unique contexts (topics): {len(context_groups)}")
print(f"Skipped (empty q/ctx)   : {skipped_empty}")

# ── Stratified sampling ───────────────────────────────────────────────────
# Take at most MAX_PER_CONTEXT questions from each unique context,
# then shuffle and cap the final list at MAX_QUESTIONS.
print(f"\nSampling up to {MAX_PER_CONTEXT} questions per context, "
      f"capping total at {MAX_QUESTIONS}...")

sampled = []
for ctx_text, entries in context_groups.items():
    take = random.sample(entries, min(MAX_PER_CONTEXT, len(entries)))
    sampled.extend(take)

# Shuffle so the final file isn't grouped by context
random.shuffle(sampled)

# Apply final cap
final_questions = sampled[:MAX_QUESTIONS]

print(f"After sampling          : {len(sampled)} questions")
print(f"After final cap         : {len(final_questions)} questions")

# Count how many unique contexts made it into the final set
final_contexts = {q["gold_context"] for q in final_questions}
print(f"Unique contexts covered : {len(final_contexts)}")

# ── Save ─────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_questions, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(final_questions)} questions to: {OUTPUT_PATH}")
print(f"   These cover {len(final_contexts)} different Wikipedia articles/topics.")
print(f"   Next step: python scripts/retrieve_passages.py")
