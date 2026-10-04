"""
generate_responses.py
======================

Runs Gemini 2.5 Flash Lite + Sarvam-105B on every triplet in raw_triplets.json
and saves answers to kannada_llm_outputs.json.

USAGE
-----
    python scripts/generate_responses.py --limit 10   # quick test
    python scripts/generate_responses.py              # full run

RESUME / CHECKPOINT
-------------------
Automatically skips questions that already have BOTH gemini_response and
sarvam_response filled in. Safe to re-run anytime.

MULTI-KEY ROTATION
------------------
Add keys to .env as GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
When one key hits its daily quota the script auto-rotates to the next key
and prints a warning. When all keys are exhausted it stops and tells you.
"""

import argparse
import json
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from llm_clients import QA_PROMPT, DELAY_BETWEEN_REQUESTS, MAX_OUTPUT_TOKENS, get_sarvam_llm

INPUT_JSON  = "data/raw_triplets.json"
OUTPUT_JSON = "data/kannada_llm_outputs.json"
SAVE_EVERY  = 10


# ── Load all Gemini keys from .env ────────────────────────────────────
def _load_gemini_keys() -> list[str]:
    keys = []
    # GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        keys.append(k)
    i = 2
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if not k:
            break
        keys.append(k)
        i += 1
    return keys


def _make_gemini_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


# ── Quota / rate-limit helpers ────────────────────────────────────────
def _is_daily_quota(exc: Exception) -> bool:
    """True if this is a hard daily quota exhaustion (retry in Xh)."""
    msg = str(exc).lower()
    return ("retry in " in msg and ("h" in msg or "d" in msg)) or \
           ("generaterequestsperday" in msg)


def _is_rate_limit(exc: Exception) -> bool:
    """True if this is a transient rate limit (should retry)."""
    if _is_daily_quota(exc):
        return False
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg


class RateLimitError(Exception):
    pass

class DailyQuotaError(Exception):
    pass


# ── Key-rotating Gemini caller ────────────────────────────────────────
class GeminiKeyRotator:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.idx = 0
        self.llm = _make_gemini_llm(keys[self.idx])
        tqdm.write(f"  [Keys] Loaded {len(keys)} Gemini API key(s). Starting with key #{self.idx+1}.")

    def _rotate(self) -> bool:
        """Switch to next key. Returns False if all keys exhausted."""
        self.idx += 1
        if self.idx >= len(self.keys):
            return False
        tqdm.write(f"\n  ⚠️  Key #{self.idx} exhausted → switching to key #{self.idx+1}")
        self.llm = _make_gemini_llm(self.keys[self.idx])
        return True

    def call(self, prompt_text: str) -> str:
        while True:
            try:
                result = self._invoke(prompt_text)
                return result
            except Exception as e:
                if _is_daily_quota(e):
                    tqdm.write(f"  [Quota] Key #{self.idx+1} daily limit hit.")
                    if not self._rotate():
                        raise DailyQuotaError(
                            f"ALL {len(self.keys)} Gemini keys exhausted for today. "
                            "Add more keys (GEMINI_API_KEY_3, _4 ...) to .env and re-run tomorrow."
                        ) from e
                    # retry with new key
                    continue
                elif _is_rate_limit(e):
                    # Transient 429 — wait and retry same key
                    tqdm.write(f"  [Rate limit] Waiting 15s ...")
                    time.sleep(15)
                    continue
                raise

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(RateLimitError),
    )
    def _invoke(self, prompt_text: str) -> str:
        result = self.llm.invoke(prompt_text)
        content = result.content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        return content.strip()


# ── Sarvam caller ────────────────────────────────────────────────────
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
def _sarvam_call(sarvam_llm, prompt_text: str) -> str:
    try:
        return sarvam_llm.invoke(prompt_text).strip()
    except Exception as e:
        if _is_rate_limit(e):
            raise RateLimitError(str(e)) from e
        raise


# ── Main ─────────────────────────────────────────────────────────────
def main(limit: int = None):

    print(f"Loading triplets from {INPUT_JSON} ...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [d for d in data if d.get("lang_code") == "kn"]
    print(f"Total Kannada triplets : {len(items)}")

    if limit:
        items = items[:limit]
        print(f"Limiting to first      : {limit} questions (--limit flag)")

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

    print("\nInitializing LLM clients...")
    gemini_keys = _load_gemini_keys()
    if not gemini_keys:
        raise ValueError("No GEMINI_API_KEY found in .env!")
    rotator    = GeminiKeyRotator(gemini_keys)
    sarvam_llm = get_sarvam_llm()
    print("Ready.\n")

    results       = list(existing.values())
    processed_ids = set(existing.keys())
    skipped = errors = 0

    for i, item in enumerate(tqdm(items, desc="Generating")):
        qid      = item["id"]
        question = item.get("question", "")
        passage  = item.get("top_passage", "")

        if qid in existing:
            prev = existing[qid]
            g_ok = prev.get("gemini_response") and not str(prev["gemini_response"]).startswith("ERROR")
            s_ok = prev.get("sarvam_response") and not str(prev["sarvam_response"]).startswith("ERROR")
            if g_ok and s_ok:
                skipped += 1
                continue

        row = dict(item)

        # Gemini (with key rotation)
        try:
            prompt_text = QA_PROMPT.format(passage=passage, question=question)
            row["gemini_response"] = rotator.call(prompt_text)
        except DailyQuotaError as e:
            _save(results)
            print(f"\n\n{'='*60}")
            print("ALL GEMINI KEYS EXHAUSTED FOR TODAY")
            print(f"{'='*60}")
            print(f"Completed so far : {len([r for r in results if r.get('gemini_response') and not str(r.get('gemini_response','')).startswith('ERROR')])} rows")
            print(f"Remaining        : {len(items) - i} questions")
            print(f"\nTo continue: add more keys to .env as GEMINI_API_KEY_{len(gemini_keys)+1}, etc.")
            print("Then re-run: python scripts/generate_responses.py")
            print(f"{'='*60}\n")
            return
        except Exception as e:
            row["gemini_response"] = f"ERROR: {e}"
            errors += 1
            tqdm.write(f"  [Gemini error] {qid}: {e}")

        # Sarvam
        try:
            prompt_text = QA_PROMPT.format(passage=passage, question=question)
            row["sarvam_response"] = _sarvam_call(sarvam_llm, prompt_text)
        except Exception as e:
            row["sarvam_response"] = f"ERROR: {e}"
            errors += 1
            tqdm.write(f"  [Sarvam error] {qid}: {e}")

        if qid in processed_ids:
            results = [r for r in results if r["id"] != qid]
        results.append(row)
        processed_ids.add(qid)

        if (i + 1) % SAVE_EVERY == 0:
            _save(results)
            tqdm.write(f"  Checkpoint saved ({len(results)} rows)")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    _save(results)

    print("\n" + "=" * 50)
    print("DONE")
    print(f"Total rows in output : {len(results)}")
    print(f"Skipped (done)       : {skipped}")
    print(f"Errors               : {errors}")
    print(f"Saved to             : {OUTPUT_JSON}")
    print("=" * 50)


def _save(results):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(limit=args.limit)
