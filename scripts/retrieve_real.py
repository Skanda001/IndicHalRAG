"""
retrieve_real.py — Phase 2: Real RAG Generation & Evaluation
============================================================

1. Retrieves top-3 passages from Kannada Wikipedia FAISS index using BAAI/bge-m3.
2. Prompts both Gemini (with multi-key rotation) and Sarvam-105B.
3. Automatically checkpoints to data/real_llm_outputs.json every question.

Run:
    python scripts/retrieve_real.py --retrieve-only   # 0 API calls, retrieval check
    python scripts/retrieve_real.py --limit 30         # generate for first 30 questions
    python scripts/retrieve_real.py                   # full run across all available quota
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

import faiss
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from llm_clients import (
    DELAY_BETWEEN_REQUESTS,
    MAX_OUTPUT_TOKENS,
    QA_PROMPT,
    get_sarvam_llm,
)

INDEX_DIR = "data/faiss_index"
OUT = "data/real_llm_outputs.json"
TOP_K = 3


def norm(s):
    return "".join(s.split()).strip(".").casefold()


def ok(x):
    return bool(x) and not str(x).startswith("ERROR")


# ── Load all Gemini keys from .env ────────────────────────────────────
def _load_gemini_keys() -> list[str]:
    keys = []
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


def _is_daily_quota(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("retry in " in msg and ("h" in msg or "d" in msg)) or \
           ("generaterequestsperday" in msg)


def _is_rate_limit(exc: Exception) -> bool:
    if _is_daily_quota(exc):
        return False
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg


class RateLimitError(Exception):
    pass


class DailyQuotaError(Exception):
    pass


class GeminiKeyRotator:
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("No GEMINI_API_KEY found in environment or .env file.")
        self.keys = keys
        self.idx = 0
        self.llm = _make_gemini_llm(keys[self.idx])
        tqdm.write(f"  [Gemini Keys] Loaded {len(keys)} key(s). Active: key #{self.idx+1}.")

    def _rotate(self) -> bool:
        self.idx += 1
        if self.idx >= len(self.keys):
            return False
        tqdm.write(f"\n  ⚠️  Gemini Key #{self.idx} exhausted → switching to key #{self.idx+1}")
        self.llm = _make_gemini_llm(self.keys[self.idx])
        return True

    def call(self, prompt_text: str) -> str:
        while True:
            try:
                result = self._invoke(prompt_text)
                return result
            except Exception as e:
                if _is_daily_quota(e):
                    tqdm.write(f"  [Quota] Gemini key #{self.idx+1} daily limit reached.")
                    if not self._rotate():
                        raise DailyQuotaError(
                            f"All {len(self.keys)} Gemini API keys have exhausted daily quota."
                        ) from e
                    continue
                elif _is_rate_limit(e):
                    tqdm.write("  [Rate limit] 429 hit — pausing 15s...")
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
        return str(content).strip()


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


def main(limit=None, retrieve_only=False):
    print("=" * 70)
    print("PHASE 2: REAL RAG RETRIEVAL & GENERATION")
    print("=" * 70)

    # 1. Load FAISS index & metadata
    print(f"Loading FAISS index from {INDEX_DIR}/wiki.index ...")
    index = faiss.read_index(f"{INDEX_DIR}/wiki.index")
    print(f"FAISS index loaded: {index.ntotal:,} vectors.")

    print(f"Loading metadata from {INDEX_DIR}/metadata.json ...")
    with open(f"{INDEX_DIR}/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(f"{INDEX_DIR}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    embed_model_name = config.get("embed_model", "BAAI/bge-m3")

    print(f"Loading embedding model: {embed_model_name} ...")
    embed_model = SentenceTransformer(embed_model_name)

    # 2. Load questions
    with open("data/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]
    print(f"Questions to process: {len(questions)}")

    # 3. Load existing results for checkpointing
    existing_map = {}
    if os.path.exists(OUT):
        with open(OUT, "r", encoding="utf-8") as f:
            old_list = json.load(f)
        existing_map = {r["id"]: r for r in old_list}
        print(f"Existing rows loaded from {OUT}: {len(existing_map)}")

    # 4. Initialize LLMs if not retrieve-only
    gemini_rotator = None
    sarvam_llm = None
    if not retrieve_only:
        gemini_keys = _load_gemini_keys()
        gemini_rotator = GeminiKeyRotator(gemini_keys)
        sarvam_llm = get_sarvam_llm()
        print("LLM clients initialized successfully.\n")

    results = dict(existing_map)
    quota_exhausted = False
    new_generated = 0

    for q in tqdm(questions, desc="Phase 2"):
        qid = q["id"]
        prev = existing_map.get(qid, {})

        # If both answers already exist and are valid, reuse
        has_gemini = ok(prev.get("gemini_response"))
        has_sarvam = ok(prev.get("sarvam_response"))

        if has_gemini and has_sarvam and not retrieve_only:
            results[qid] = prev
            continue

        # If passages were already retrieved, reuse them to save compute
        if prev.get("retrieved_passages") and prev.get("top_passage"):
            passages = prev["retrieved_passages"]
            scores = prev.get("retrieval_scores", [])
            gold_in_retrieved = prev.get("gold_in_retrieved")
            top_passage = prev["top_passage"]
        else:
            qv = embed_model.encode([q["question"]], normalize_embeddings=True).astype("float32")
            search_scores, idx = index.search(qv, TOP_K)
            passages = [meta[i]["text"] for i in idx[0] if i >= 0]
            scores = [float(s) for s in search_scores[0]]
            gold = q.get("gold_answer", "")
            gold_in_retrieved = any(norm(gold) in norm(p) for p in passages) if gold else None
            top_passage = "\n\n".join(passages)

        row = {
            "id": qid,
            "question": q["question"],
            "gold_answer": q.get("gold_answer", ""),
            "retrieved_passages": passages,
            "retrieval_scores": scores,
            "gold_in_retrieved": gold_in_retrieved,
            "top_passage": top_passage,
            "passage_source": "faiss_retrieved",
            "gemini_response": prev.get("gemini_response"),
            "sarvam_response": prev.get("sarvam_response"),
        }

        # LLM Generation
        if not retrieve_only and not quota_exhausted:
            prompt = QA_PROMPT.format(passage=top_passage, question=q["question"])

            # Gemini
            if not ok(row["gemini_response"]):
                try:
                    row["gemini_response"] = gemini_rotator.call(prompt)
                except DailyQuotaError as e:
                    tqdm.write(f"\n[Quota Alert] {e}")
                    tqdm.write("Daily Gemini limit reached for all keys. Stopping generation.")
                    quota_exhausted = True
                except Exception as e:
                    row["gemini_response"] = f"ERROR: {e}"
                    tqdm.write(f"  [Gemini error] {qid}: {e}")

            # Sarvam
            if not ok(row["sarvam_response"]) and not quota_exhausted:
                try:
                    row["sarvam_response"] = _sarvam_call(sarvam_llm, prompt)
                except Exception as e:
                    row["sarvam_response"] = f"ERROR: {e}"
                    tqdm.write(f"  [Sarvam error] {qid}: {e}")

            new_generated += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)

        results[qid] = row

        # Checkpoint to disk
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)

        if quota_exhausted:
            break

    # Summary report
    rows = list(results.values())
    known = sum(r["gold_in_retrieved"] is not None for r in rows)
    hit = sum(r["gold_in_retrieved"] is True for r in rows)

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"Total rows in {OUT}        : {len(rows)}")
    print(f"Questions with gold answer   : {known}")
    if known > 0:
        print(f"Gold answer retrieved (R@3)  : {hit}/{known} ({hit / known:.1%})")

    if not retrieve_only:
        gemini_ok = sum(ok(r.get("gemini_response")) for r in rows)
        sarvam_ok = sum(ok(r.get("sarvam_response")) for r in rows)
        print(f"Gemini responses completed   : {gemini_ok}/{len(rows)}")
        print(f"Sarvam responses completed   : {sarvam_ok}/{len(rows)}")
        print(f"Newly generated this session : {new_generated}")

    print(f"File updated                 : {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N questions")
    parser.add_argument("--retrieve-only", action="store_true", help="Only run FAISS retrieval, no LLM calls")
    args = parser.parse_args()
    main(limit=args.limit, retrieve_only=args.retrieve_only)
