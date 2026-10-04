"""
retrieve_real.py  (FIXED)  -  Phase 2: real RAG
================================================

Changes vs previous version
- uses the SAME embedding model the index was built with (config.json)
- records `gold_in_retrieved` per question:
      True  = gold answer exists and was retrieved
      False = gold answer exists but was NOT retrieved
      None  = no gold answer available
- NO fake refusal text is ever written; a failed API call stays an ERROR
  and is retried on the next run (only errored rows are re-run)
- --retrieve-only : inspect retrieval with ZERO API calls
- --limit N       : test on N questions first
- 429 handling: stops cleanly after repeated quota errors instead of
  burning through all rows

Run order:
    python scripts/retrieve_real.py --retrieve-only
    python scripts/retrieve_real.py --limit 10
    python scripts/retrieve_real.py
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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from llm_clients import (
    QA_PROMPT,
    DELAY_BETWEEN_REQUESTS,
    get_gemini_llm,
    get_sarvam_llm,
)


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

INDEX_DIR = "data/faiss_index"
OUT = "data/real_llm_outputs.json"

TOP_K = 3


# ─────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────

def norm(s):
    return "".join(s.split()).strip(".").casefold()


# ─────────────────────────────────────────────────────────────────────
# GEMINI RESPONSE TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def gem_text(r):
    c = r.content

    if isinstance(c, list):
        c = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in c
        )

    return c.strip()


# ─────────────────────────────────────────────────────────────────────
# CHECK WHETHER RESPONSE IS VALID
# ─────────────────────────────────────────────────────────────────────

def ok(x):
    return bool(x) and not str(x).startswith("ERROR")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main(limit, retrieve_only):

    # ─────────────────────────────────────────────────────────────────
    # LOAD FAISS INDEX
    # ─────────────────────────────────────────────────────────────────

    print("Loading FAISS index...")

    index = faiss.read_index(
        f"{INDEX_DIR}/wiki.index"
    )

    print(
        f"FAISS index loaded: {index.ntotal} vectors"
    )


    # ─────────────────────────────────────────────────────────────────
    # LOAD METADATA
    # ─────────────────────────────────────────────────────────────────

    meta = json.load(
        open(
            f"{INDEX_DIR}/metadata.json",
            encoding="utf-8"
        )
    )


    # ─────────────────────────────────────────────────────────────────
    # LOAD SAME EMBEDDING MODEL USED TO BUILD INDEX
    # ─────────────────────────────────────────────────────────────────

    config = json.load(
        open(
            f"{INDEX_DIR}/config.json",
            encoding="utf-8"
        )
    )

    embed_model = config["embed_model"]

    print(
        f"Loading embedding model: {embed_model}"
    )

    model = SentenceTransformer(
        embed_model
    )


    # ─────────────────────────────────────────────────────────────────
    # LOAD QUESTIONS
    # ─────────────────────────────────────────────────────────────────

    questions = json.load(
        open(
            "data/questions.json",
            encoding="utf-8"
        )
    )

    if limit:
        questions = questions[:limit]

    print(
        f"Questions to process: {len(questions)}"
    )


    # ─────────────────────────────────────────────────────────────────
    # LOAD PREVIOUS RESULTS
    # ─────────────────────────────────────────────────────────────────

    old = {}

    if os.path.exists(OUT):

        old = {
            r["id"]: r
            for r in json.load(
                open(
                    OUT,
                    encoding="utf-8"
                )
            )
        }

        print(
            f"Previous results loaded: {len(old)}"
        )


    # ─────────────────────────────────────────────────────────────────
    # LOAD LLM CLIENTS
    # ─────────────────────────────────────────────────────────────────

    if not retrieve_only:

        gem = get_gemini_llm()
        sar = get_sarvam_llm()

    results = {}

    quota_hits = 0


    # ─────────────────────────────────────────────────────────────────
    # PROCESS QUESTIONS
    # ─────────────────────────────────────────────────────────────────

    for q in tqdm(
        questions,
        desc="Phase 2"
    ):

        qid = q["id"]

        prev = old.get(qid)


        # ─────────────────────────────────────────────────────────────
        # KEEP COMPLETED PREVIOUS ROW
        # ─────────────────────────────────────────────────────────────

        if (
            prev
            and ok(prev.get("gemini_response"))
            and ok(prev.get("sarvam_response"))
        ):

            results[qid] = prev

            continue


        # ─────────────────────────────────────────────────────────────
        # EMBED QUESTION
        # ─────────────────────────────────────────────────────────────

        qv = model.encode(
            [q["question"]],
            normalize_embeddings=True
        ).astype("float32")


        # ─────────────────────────────────────────────────────────────
        # FAISS RETRIEVAL
        # ─────────────────────────────────────────────────────────────

        scores, idx = index.search(
            qv,
            TOP_K
        )


        passages = [
            meta[i]["text"]
            for i in idx[0]
        ]


        # ─────────────────────────────────────────────────────────────
        # GOLD ANSWER
        # ─────────────────────────────────────────────────────────────

        gold = q.get(
            "gold_answer",
            ""
        )


        # ─────────────────────────────────────────────────────────────
        # GOLD ANSWER RETRIEVAL CHECK
        #
        # True  -> gold answer exists and is in retrieved passages
        # False -> gold answer exists but is NOT in retrieved passages
        # None  -> no gold answer available
        # ─────────────────────────────────────────────────────────────

        gold_in_retrieved = (
            any(
                norm(gold) in norm(p)
                for p in passages
            )
            if gold
            else None
        )


        # ─────────────────────────────────────────────────────────────
        # CREATE RESULT ROW
        # ─────────────────────────────────────────────────────────────

        row = {
            "id": qid,

            "question": q["question"],

            "gold_answer": gold,

            "retrieved_passages": passages,

            "retrieval_scores": [
                float(s)
                for s in scores[0]
            ],

            "gold_in_retrieved": gold_in_retrieved,

            "top_passage": "\n\n".join(
                passages
            ),

            "passage_source": "faiss_retrieved",

            "gemini_response": (
                prev.get("gemini_response")
                if prev
                else None
            ),

            "sarvam_response": (
                prev.get("sarvam_response")
                if prev
                else None
            ),
        }


        # ─────────────────────────────────────────────────────────────
        # LLM GENERATION
        # ─────────────────────────────────────────────────────────────

        if not retrieve_only:

            prompt = QA_PROMPT.format(
                passage=row["top_passage"],
                question=q["question"]
            )


            # ─────────────────────────────────────────────────────────
            # GEMINI
            # ─────────────────────────────────────────────────────────

            if not ok(
                row["gemini_response"]
            ):

                try:

                    row["gemini_response"] = gem_text(
                        gem.invoke(prompt)
                    )

                except Exception as e:

                    row["gemini_response"] = (
                        f"ERROR: {e}"
                    )

                    if (
                        "429" in str(e)
                        or "RESOURCE_EXHAUSTED" in str(e)
                    ):
                        quota_hits += 1


            # ─────────────────────────────────────────────────────────
            # SARVAM
            # ─────────────────────────────────────────────────────────

            if not ok(
                row["sarvam_response"]
            ):

                try:

                    row["sarvam_response"] = (
                        sar.invoke(prompt).strip()
                    )

                except Exception as e:

                    row["sarvam_response"] = (
                        f"ERROR: {e}"
                    )


            # ─────────────────────────────────────────────────────────
            # DELAY BETWEEN API REQUESTS
            # ─────────────────────────────────────────────────────────

            time.sleep(
                DELAY_BETWEEN_REQUESTS
            )


        # ─────────────────────────────────────────────────────────────
        # SAVE CURRENT RESULT
        # ─────────────────────────────────────────────────────────────

        results[qid] = row

        json.dump(
            list(results.values()),
            open(
                OUT,
                "w",
                encoding="utf-8"
            ),
            ensure_ascii=False,
            indent=2
        )


        # ─────────────────────────────────────────────────────────────
        # STOP AFTER REPEATED QUOTA ERRORS
        # ─────────────────────────────────────────────────────────────

        if quota_hits >= 5:

            print(
                "\nGemini daily quota exhausted - stopping. "
                "Re-run tomorrow; finished rows are kept."
            )

            break


    # ─────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────

    rows = list(
        results.values()
    )


    # Questions where gold answer actually exists
    known = sum(
        r["gold_in_retrieved"] is not None
        for r in rows
    )


    # Questions where gold answer exists AND was retrieved
    hit = sum(
        r["gold_in_retrieved"] is True
        for r in rows
    )


    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)

    print(
        f"Rows processed: {len(rows)}"
    )

    print(
        f"Questions with known gold answer: "
        f"{known}/{len(rows)}"
    )

    if known > 0:

        print(
            f"Gold answer inside retrieved passages: "
            f"{hit}/{known} "
            f"({hit / known:.1%})"
        )

    else:

        print(
            "Gold answer inside retrieved passages: "
            "N/A"
        )


    # ─────────────────────────────────────────────────────────────────
    # LLM SUMMARY
    # ─────────────────────────────────────────────────────────────────

    if not retrieve_only:

        print(
            "Gemini OK:",
            sum(
                ok(r["gemini_response"])
                for r in rows
            )
        )

        print(
            "Sarvam OK:",
            sum(
                ok(r["sarvam_response"])
                for r in rows
            )
        )

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--limit",
        type=int
    )

    ap.add_argument(
        "--retrieve-only",
        action="store_true"
    )

    a = ap.parse_args()

    main(
        a.limit,
        a.retrieve_only
    )
