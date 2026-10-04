"""
generate_responses.py
======================

Runs Gemini 2.5 Flash + Sarvam-30B on every triplet in raw_triplets.json
and saves answers to kannada_llm_outputs.json.

USAGE
-----
    # Quick test — only process 10 questions:
    python scripts/generate_responses.py --limit 10

    # Full run — process all questions:
    python scripts/generate_responses.py

RESUME / CHECKPOINT
-------------------
If the script crashes or you ran --limit 10 first, it will automatically
skip questions that already have BOTH gemini_response and sarvam_response
filled in. So you can always re-run safely without re-spending API quota.

RATE LIMITS
-----------
Gemini free tier: 15 requests/minute.
We call Gemini once and Sarvam once per question, then sleep
DELAY_BETWEEN_REQUESTS seconds. At 5 seconds delay:
  - 2 calls per question × 60s / 5s delay = ~24 API calls/min total
  - Split across 2 providers = ~12 RPM each. Well under both limits.

If you see 429 errors, increase DELAY_BETWEEN_REQUESTS in llm_clients.py.
"""

import argparse
import json
import os
import sys
import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Allow running from repo root: python scripts/generate_responses.py
sys.path.insert(0, os.path.dirname(__file__))
from llm_clients import QA_PROMPT, DELAY_BETWEEN_REQUESTS, get_gemini_llm, get_sarvam_llm

INPUT_JSON  = "data/raw_triplets.json"
OUTPUT_JSON = "data/kannada_llm_outputs.json"
SAVE_EVERY  = 10   # write to disk every N questions (checkpoint)


# ── Retry wrapper for 429 / rate-limit errors ────────────────────────
class RateLimitError(Exception):
    pass


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
def _call_with_retry(fn):
    try:
        return fn()
    except Exception as e:
        if _is_rate_limit(e):
            raise RateLimitError(str(e)) from e
        raise


# ── Per-model answer generators ──────────────────────────────────────
def generate_gemini_answer(gemini_llm, question: str, passage: str) -> str:
    prompt_text = QA_PROMPT.format(passage=passage, question=question)

    def _run():
        result = gemini_llm.invoke(prompt_text)
        content = result.content
        # Gemini 3.x returns content as a list of blocks, not a plain string
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        return content.strip()

    return _call_with_retry(_run)


def generate_sarvam_answer(sarvam_llm, question: str, passage: str) -> str:
    prompt_text = QA_PROMPT.format(passage=passage, question=question)
    return _call_with_retry(lambda: sarvam_llm.invoke(prompt_text).strip())


# ── Main ─────────────────────────────────────────────────────────────
def main(limit: int = None):

    # Load triplets
    print(f"Loading triplets from {INPUT_JSON} ...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Kannada only (safety filter)
    items = [d for d in data if d.get("lang_code") == "kn"]
    print(f"Total Kannada triplets : {len(items)}")

    # Apply --limit
    if limit:
        items = items[:limit]
        print(f"Limiting to first      : {limit} questions (--limit flag)")

    # Load existing output for resume/checkpoint
    existing: dict[str, dict] = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            existing_list = json.load(f)
        existing = {row["id"]: row for row in existing_list}
        already_done = sum(
            1 for r in existing.values()
            if r.get("gemini_response") and r.get("sarvam_response")
            and not str(r["gemini_response"]).startswith("ERROR")
            and not str(r["sarvam_response"]).startswith("ERROR")
        )
        print(f"Already completed      : {already_done} questions (will skip these)")

    # Init LLMs
    print("\nInitializing LLM clients...")
    gemini_llm = get_gemini_llm()
    sarvam_llm = get_sarvam_llm()
    print("Ready.\n")

    results = list(existing.values())   # start with what we already have
    processed_ids = set(existing.keys())

    skipped  = 0
    errors   = 0

    for i, item in enumerate(tqdm(items, desc="Generating")):
        qid      = item["id"]
        question = item.get("question", "")
        passage  = item.get("top_passage", "")

        # Skip if both answers already exist and are not errors
        if qid in existing:
            prev = existing[qid]
            g_ok = prev.get("gemini_response") and not str(prev["gemini_response"]).startswith("ERROR")
            s_ok = prev.get("sarvam_response") and not str(prev["sarvam_response"]).startswith("ERROR")
            if g_ok and s_ok:
                skipped += 1
                continue

        # Copy item so we don't mutate the original
        row = dict(item)

        # Gemini
        try:
            row["gemini_response"] = generate_gemini_answer(gemini_llm, question, passage)
        except Exception as e:
            row["gemini_response"] = f"ERROR: {e}"
            errors += 1
            tqdm.write(f"  [Gemini error] {qid}: {e}")

        # Sarvam
        try:
            row["sarvam_response"] = generate_sarvam_answer(sarvam_llm, question, passage)
        except Exception as e:
            row["sarvam_response"] = f"ERROR: {e}"
            errors += 1
            tqdm.write(f"  [Sarvam error] {qid}: {e}")

        # Update results list
        if qid in processed_ids:
            # Replace the old entry
            results = [r for r in results if r["id"] != qid]
        results.append(row)
        processed_ids.add(qid)

        # Checkpoint
        if (i + 1) % SAVE_EVERY == 0:
            _save(results)
            tqdm.write(f"  Checkpoint saved ({len(results)} rows)")

        # Rate-limit delay (keeps us under 15 RPM on Gemini free tier)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    _save(results)

    print("\n" + "=" * 50)
    print("DONE")
    print(f"Total rows in output : {len(results)}")
    print(f"Skipped (done)       : {skipped}")
    print(f"Errors               : {errors}")
    print(f"Saved to             : {OUTPUT_JSON}")
    print("=" * 50)

    if errors > 0:
        print(f"\n⚠️  {errors} errors occurred. Re-run the script — "
              "completed rows will be skipped automatically.")


def _save(results):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N questions. "
             "E.g. --limit 10 for a quick test. "
             "Omit for full run."
    )
    args = parser.parse_args()
    main(limit=args.limit)
