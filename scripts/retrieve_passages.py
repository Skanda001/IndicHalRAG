"""
retrieve_passages.py
=====================

PHASE 1 — GOLD CONTEXT PIPELINE (Kannada only)

Creates raw_triplets.json using IndicQA gold_context directly.

NEW IN THIS VERSION — answer-in-context validation
----------------------------------------------------
You correctly noticed some passages were "illogical" for their
question. Example found in your own qwen_demo_outputs.json: a question
asking what Nepal's PM said about Kalam's death was paired with a
gold_context about the Dalai Lama and Bhutan instead. That paragraph
never contained the answer at all.

This is NOT a bug in this script's logic (it always copied
gold_context 1:1, which is correct). It is a data-quality issue in
IndicQA itself: some rows are pulled from multi-paragraph articles
where the assigned `context` doesn't actually contain that row's
answer.

Feeding a model a passage that structurally cannot contain the right
answer, then grading its response for "hallucination", makes your
results meaningless for those rows -- the model either correctly says
"not found" (which looks right for the wrong reason) or answers from
its own parametric knowledge (which then looks like hallucination when
it's actually just... correct, from a passage that couldn't help it).

So: every row is now checked for whether gold_answer actually appears
inside gold_context (normalized for whitespace/punctuation/case).
Rows that fail this check are pulled OUT of raw_triplets.json and saved
separately to data/flagged_context_mismatch.json so you can inspect and
decide (drop them, or manually fix the context) instead of them
silently corrupting your dataset.

Pipeline
--------
questions.json -> retrieve_passages.py -> raw_triplets.json -> generate_responses.py
                                        -> flagged_context_mismatch.json (for review)
"""

import json
import os
import re
import sys
import unicodedata

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

INPUT_PATH = "data/questions.json"
OUTPUT_PATH = "data/raw_triplets.json"
FLAGGED_PATH = "data/flagged_context_mismatch.json"


def normalize(text: str) -> str:
    """Normalize text for a robust substring check: unicode-normalize,
    strip punctuation, collapse whitespace, casefold. Works fine for
    Kannada script since we're only stripping ASCII punctuation/spaces,
    not touching the script itself."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\"'""''.,!?;:()\[\]।]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def answer_in_context(gold_answer: str, gold_context: str) -> bool:
    if not gold_answer:
        # No gold answer to check against -- can't validate, let it
        # through (these are already counted separately as
        # "missing_answer" below).
        return True
    return normalize(gold_answer) in normalize(gold_context)


# =========================================================
# CONTEXT WINDOWING
# =========================================================
# Instead of feeding the model (and your manual annotators) the full
# multi-sentence paragraph, we trim it down to just the sentence that
# contains the answer, plus a small buffer of sentences before/after
# for surrounding context. This keeps annotation fast without turning
# the task into pure extraction from a single spoon-fed sentence
# (which would stop testing faithfulness at all).
#
# Tune these two numbers if 3 sentences feels too short/long:
SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")
WINDOW_SIZE = 1          # sentences kept before/after the answer sentence
MAX_TRIMMED_CHARS = 500  # hard safety cap


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def trim_context(gold_context: str, gold_answer: str) -> str:
    sentences = split_sentences(gold_context)
    if not sentences:
        return gold_context[:MAX_TRIMMED_CHARS]

    norm_answer = normalize(gold_answer) if gold_answer else ""
    hit_idx = None
    for i, s in enumerate(sentences):
        if norm_answer and norm_answer in normalize(s):
            hit_idx = i
            break

    if hit_idx is None:
        # Answer wasn't found sentence-by-sentence (can happen if it
        # spans a sentence boundary, even though the full-paragraph
        # check passed). Fall back to the first couple of sentences
        # rather than dropping the row.
        trimmed = " ".join(sentences[: 2 * WINDOW_SIZE + 1])
    else:
        start = max(0, hit_idx - WINDOW_SIZE)
        end = min(len(sentences), hit_idx + WINDOW_SIZE + 1)
        trimmed = " ".join(sentences[start:end])

    if len(trimmed) > MAX_TRIMMED_CHARS:
        trimmed = trimmed[:MAX_TRIMMED_CHARS].rsplit(" ", 1)[0] + "..."

    return trimmed


print("Loading questions...")
if not os.path.exists(INPUT_PATH):
    print(f"ERROR: {INPUT_PATH} not found")
    print("Run: python scripts/download_questions.py  first")
    exit()

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Questions loaded: {len(questions)}")

triplets = []
flagged = []
missing_context = 0
missing_answer = 0
skipped = 0
skipped_non_kannada = 0
context_mismatch = 0
total_full_chars = 0
total_trimmed_chars = 0

print("\nCreating triplets...\n")

for i, q in enumerate(questions):
    try:
        question = q.get("question", "").strip()
        gold_context = q.get("gold_context", "").strip()
        gold_answer = q.get("gold_answer", "").strip()
        language = q.get("language", "")
        lang_code = q.get("lang_code", "")
        qid = q.get("id", f"q_{i}")

        # Safety filter: only Kannada rows survive, even if the input
        # file still has other languages in it from an old run.
        if lang_code != "kn":
            skipped_non_kannada += 1
            continue

        if question == "":
            skipped += 1
            continue
        if gold_context == "":
            missing_context += 1
            skipped += 1
            continue
        if gold_answer == "":
            missing_answer += 1

        # ── Answer-in-context validation ──
        # Catches rows like the Nepal/Dalai-Lama example: a passage
        # that structurally cannot contain the answer to its question.
        if gold_answer and not answer_in_context(gold_answer, gold_context):
            context_mismatch += 1
            flagged.append({
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "gold_context": gold_context,
                "reason": "gold_answer not found in gold_context",
            })
            continue

        # Trim the paragraph down to just the sentences around the
        # answer -- this is what actually gets fed to the LLMs and
        # what you'll read during manual annotation.
        trimmed_context = trim_context(gold_context, gold_answer)

        triplet = {
            "id": qid,
            "language": language,
            "lang_code": lang_code,
            "question": question,
            "gold_answer": gold_answer,
            # TRIMMED CONTEXT -- what the LLM sees, and what you annotate
            "top_passage": trimmed_context,
            # ORIGINAL FULL PARAGRAPH -- kept for audit/reference only,
            # not sent to the LLM. Delete this key later if you want a
            # smaller file once you trust the trimming.
            "full_context": gold_context,
            "passage_source": "gold_trimmed",
            "all_passages": [
                {"text": trimmed_context, "lang": lang_code, "score": 1.0, "kw_hits": 999, "source": "gold_trimmed"}
            ],
            # generation placeholders — filled by two CLOUD models
            "gemini_response": None,
            "sarvam_response": None,
        }
        triplets.append(triplet)
        total_full_chars += len(gold_context)
        total_trimmed_chars += len(trimmed_context)

    except Exception as e:
        print(f"ERROR at question {i}: {e}")

print("Saving raw triplets...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(triplets, f, ensure_ascii=False, indent=2)

if flagged:
    with open(FLAGGED_PATH, "w", encoding="utf-8") as f:
        json.dump(flagged, f, ensure_ascii=False, indent=2)

print("\n===================================")
print("TRIPLETS CREATED SUCCESSFULLY")
print(f"Total triplets       : {len(triplets)}")
print(f"Non-Kannada skipped  : {skipped_non_kannada}")
print(f"Missing context      : {missing_context}")
print(f"Missing answers      : {missing_answer}")
print(f"Context mismatches   : {context_mismatch}  (gold_answer not found in gold_context)")
print(f"Skipped              : {skipped}")
if triplets:
    avg_full = total_full_chars / len(triplets)
    avg_trimmed = total_trimmed_chars / len(triplets)
    print(f"\nAvg context length   : {avg_full:.0f} chars (full)  ->  {avg_trimmed:.0f} chars (trimmed)")
print(f"\nSaved triplets to    : {OUTPUT_PATH}")
if flagged:
    print(f"Saved flagged rows to: {FLAGGED_PATH}  <-- review these, they are excluded from the main file")
print("===================================")
