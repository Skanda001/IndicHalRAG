"""
build_faiss_index.py  (FIXED + CUDA)
=====================================

WHY THE OLD INDEX GAVE USELESS PASSAGES
---------------------------------------
Old settings: CHUNK_WORDS=200 but MAX_CHARS=800.
A 200-word Kannada chunk is ~1,500+ characters, so it was ALWAYS rejected
by the MAX_CHARS filter. Only tiny stub articles (<800 chars total) survived:
median retrieved chunk = 43 words / 354 chars (checked on your output).
So Tagore, Tanjavur, Assam tea etc. were simply NOT in the index.
That is why 8,247 chunks came out of 31,437 articles, and why only 9/500
retrieved sets contained the gold answer.

FIXES
-----
1. Chunks are ~80 words (~700 chars) with 40-word overlap; MAX_CHARS=1500.
2. IndicQA gold contexts (full paragraphs) are added to the corpus, the
   standard open-domain-QA setup, so every question's answer paragraph
   EXISTS somewhere in the corpus and retrieval can be measured honestly.
3. Better retriever: BAAI/bge-m3 (multilingual, trained for retrieval,
   supports Kannada). LaBSE is a translation-pair model, weak for
   question -> passage search. Set EMBED_MODEL back to LaBSE if needed.
4. IndexFlatIP (exact search) - correct for this corpus and no IVF warnings.
5. Built-in sanity report: answer-coverage of the corpus and recall@3
   of retrieval. Do NOT spend API calls until recall@3 looks reasonable.
6. Uses CUDA automatically when available. Falls back to CPU if CUDA
   is unavailable.

Run:
    python scripts/build_faiss_index.py
"""

import os
import sys
import json
import hashlib
import statistics

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import faiss
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

WIKI_CONFIG   = "20231101.kn"

EMBED_MODEL   = "BAAI/bge-m3"

CHUNK_WORDS   = 80
CHUNK_STRIDE  = 40

MIN_CHARS     = 80
MAX_CHARS     = 1500

# Keep this moderate for RTX 4050.
# If GPU memory is comfortable later, this can be increased.
BATCH_SIZE    = 16

QUESTIONS     = "data/questions.json"
INDEX_DIR     = "data/faiss_index"

os.makedirs(INDEX_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 0. DEVICE
# ─────────────────────────────────────────────────────────────────────

print("=" * 70)
print("DEVICE CHECK")
print("=" * 70)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Optional informational output
    gpu_props = torch.cuda.get_device_properties(0)
    print(
        f"GPU memory: "
        f"{gpu_props.total_memory / (1024 ** 3):.2f} GB"
    )
else:
    DEVICE = "cpu"
    print("GPU: NONE")
    print("WARNING: CUDA is not available.")
    print("Embedding will run on CPU and may be very slow.")

print(f"Embedding device: {DEVICE}")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────
# 1. CHUNKING
# ─────────────────────────────────────────────────────────────────────

def chunk_text(text):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = start + CHUNK_WORDS

        chunk = " ".join(words[start:end])

        if len(chunk) >= MIN_CHARS:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += CHUNK_STRIDE

    return chunks


# ─────────────────────────────────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────────────────────────────────

def valid(t):
    t = t.strip()

    if not (MIN_CHARS <= len(t) <= MAX_CHARS):
        return False

    alpha_ratio = sum(c.isalpha() for c in t) / len(t)

    return alpha_ratio >= 0.35


# ─────────────────────────────────────────────────────────────────────
# 3. HASHING / DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────

def h(t):
    return hashlib.md5(
        t.encode("utf-8")
    ).hexdigest()


texts = []
meta = []
seen = set()


def add(chunk, title, source):

    if valid(chunk):

        chunk_hash = h(chunk)

        if chunk_hash not in seen:

            seen.add(chunk_hash)

            texts.append(chunk)

            meta.append(
                {
                    "title": title,
                    "text": chunk,
                    "source": source
                }
            )


# ─────────────────────────────────────────────────────────────────────
# 4. LOAD KANNADA WIKIPEDIA
# ─────────────────────────────────────────────────────────────────────

print("\nLoading Kannada Wikipedia...")

ds = load_dataset(
    "wikimedia/wikipedia",
    WIKI_CONFIG,
    split="train"
)

for row in tqdm(ds, desc="Wikipedia"):

    title = (
        row.get("title") or ""
    ).strip()

    body = (
        row.get("text") or ""
    ).strip()

    if body:

        for c in chunk_text(
            f"{title}. {body}"
        ):

            add(
                c,
                title,
                "wikipedia"
            )


n_wiki = len(texts)


# ─────────────────────────────────────────────────────────────────────
# 5. LOAD INDICQA GOLD CONTEXTS
# ─────────────────────────────────────────────────────────────────────

questions = json.load(
    open(
        QUESTIONS,
        encoding="utf-8"
    )
)

for q in questions:

    ctx = (
        q.get("gold_context") or ""
    ).strip().strip('"')

    for c in chunk_text(ctx):

        add(
            c,
            "indicqa",
            "indicqa_gold"
        )


print(
    f"Chunks: "
    f"wikipedia={n_wiki}  "
    f"indicqa={len(texts) - n_wiki}  "
    f"total={len(texts)}"
)


# ─────────────────────────────────────────────────────────────────────
# 6. CHUNK STATISTICS
# ─────────────────────────────────────────────────────────────────────

L = [
    len(t)
    for t in texts
]

print(
    f"Chunk chars: "
    f"median={statistics.median(L):.0f} "
    f"max={max(L)} "
    f"(old broken index: median 354)"
)


# ─────────────────────────────────────────────────────────────────────
# 7. ANSWER COVERAGE
# ─────────────────────────────────────────────────────────────────────

def norm(s):

    return (
        "".join(s.split())
        .strip(".")
        .casefold()
    )


corpus_norm = [
    norm(t)
    for t in texts
]


with_ans = [
    q
    for q in questions
    if q.get("gold_answer")
]


covered = sum(
    1
    for q in with_ans
    if any(
        norm(q["gold_answer"]) in c
        for c in corpus_norm
    )
)


print(
    f"Answer coverage in corpus: "
    f"{covered}/{len(with_ans)}"
)


# ─────────────────────────────────────────────────────────────────────
# 8. LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────

print(
    f"\nEmbedding with {EMBED_MODEL} ..."
)

model = SentenceTransformer(
    EMBED_MODEL,
    device=DEVICE
)

print(
    f"Model device: {model.device}"
)


# ─────────────────────────────────────────────────────────────────────
# 9. EMBED CORPUS
# ─────────────────────────────────────────────────────────────────────

print(
    f"\nEmbedding {len(texts):,} chunks..."
)

emb = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")


print(
    f"Embedding shape: {emb.shape}"
)


# ─────────────────────────────────────────────────────────────────────
# 10. BUILD FAISS INDEX
# ─────────────────────────────────────────────────────────────────────

print("\nBuilding FAISS IndexFlatIP...")

index = faiss.IndexFlatIP(
    emb.shape[1]
)

index.add(emb)


# ─────────────────────────────────────────────────────────────────────
# 11. SAVE INDEX
# ─────────────────────────────────────────────────────────────────────

INDEX_PATH = os.path.join(
    INDEX_DIR,
    "wiki.index"
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.json"
)

CONFIG_PATH = os.path.join(
    INDEX_DIR,
    "config.json"
)


faiss.write_index(
    index,
    INDEX_PATH
)


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        meta,
        f,
        ensure_ascii=False
    )


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "embed_model": EMBED_MODEL
        },
        f,
        indent=2
    )


print(
    f"\nIndex saved: {index.ntotal} vectors"
)


# ─────────────────────────────────────────────────────────────────────
# 12. RETRIEVAL SANITY CHECK
# ─────────────────────────────────────────────────────────────────────

print(
    "\nRunning retrieval sanity check..."
)

qv = model.encode(
    [
        q["question"]
        for q in with_ans
    ],
    normalize_embeddings=True,
    batch_size=BATCH_SIZE
).astype("float32")


# Top 3 retrieval
_, I = index.search(
    qv,
    3
)


# ─────────────────────────────────────────────────────────────────────
# 13. RECALL@3
# ─────────────────────────────────────────────────────────────────────

hit = sum(
    1
    for q, row in zip(
        with_ans,
        I
    )
    if any(
        norm(q["gold_answer"]) in corpus_norm[i]
        for i in row
    )
)


recall = (
    hit / len(with_ans)
    if len(with_ans) > 0
    else 0
)


print(
    f"\nRECALL@3 "
    f"(gold answer inside top-3 chunks): "
    f"{hit}/{len(with_ans)} "
    f"= {recall:.1%}"
)

print(
    "Old broken index: 9/500 = 1.8%"
)


# ─────────────────────────────────────────────────────────────────────
# 14. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("BUILD COMPLETE")
print("=" * 70)

print(
    f"Documents/chunks indexed : {index.ntotal:,}"
)

print(
    f"Wikipedia chunks          : {n_wiki:,}"
)

print(
    f"IndicQA chunks             : {len(texts) - n_wiki:,}"
)

print(
    f"Answer coverage            : "
    f"{covered}/{len(with_ans)}"
)

print(
    f"Recall@3                   : "
    f"{hit}/{len(with_ans)} = {recall:.1%}"
)

print(
    f"Embedding model            : {EMBED_MODEL}"
)

print(
    f"Embedding device           : {DEVICE}"
)

print(
    f"FAISS index                : IndexFlatIP"
)

print(
    f"Index path                 : {INDEX_PATH}"
)

print("=" * 70)
