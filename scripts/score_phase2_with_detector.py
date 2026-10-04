"""
score_phase2_with_detector.py
==============================

Uses the fine-tuned mBERT hallucination detector (models/detector_combined)
to score the Phase 2 real RAG outputs (data/real_llm_outputs.json) and assign:
  - gemini_label
  - sarvam_label

Labels:
  - "hallucinated" : model predicted hallucination (class 1)
  - "refused"      : model explicitly refused ("ಈ ಮಾಹಿತಿ ಪ್ಯಾಸೇಜ್ನಲ್ಲಿ ಇಲ್ಲ")
  - "correct"      : model predicted faithful / supported answer (class 0)

Run:
    python scripts/score_phase2_with_detector.py
"""

import json
import os
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_DIR = "models/detector_combined"
DATA_PATH = "data/real_llm_outputs.json"
MAX_LEN   = 256

REFUSAL_PHRASES = [
    "ಈ ಮಾಹಿತಿ ಪ್ಯಾಸೇಜ್ನಲ್ಲಿ ಇಲ್ಲ",
    "ಪ್ಯಾಸೇಜ್ನಲ್ಲಿ ಇಲ್ಲ",
    "ಮಾಹಿತಿ ಇಲ್ಲ",
    "ಪಠ್ಯದಲ್ಲಿ ಇಲ್ಲ",
    "ತಿಳಿದುಬರುವುದಿಲ್ಲ"
]


def is_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return any(p in t for p in REFUSAL_PHRASES)


def main():
    print("=" * 65)
    print("SCORING PHASE 2 REAL RAG OUTPUTS WITH MBERT DETECTOR")
    print("=" * 65)

    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory {MODEL_DIR} not found. Run train_detector.py first.")
        sys.exit(1)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file {DATA_PATH} not found.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading mBERT detector from {MODEL_DIR}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total rows in {DATA_PATH}: {len(data)}")

    def classify_response(question, passage, response):
        if not response or str(response).startswith("ERROR"):
            return None

        if is_refusal(response):
            return "refused"

        text_a = f"{question} [SEP] {passage}"
        text_b = response

        inputs = tokenizer(
            text_a,
            text_b,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        return "hallucinated" if pred == 1 else "correct"

    scored_count = 0

    for row in data:
        q = row.get("question", "")
        p = row.get("top_passage", "")
        g_resp = row.get("gemini_response")
        s_resp = row.get("sarvam_response")

        if g_resp and not str(g_resp).startswith("ERROR"):
            row["gemini_label"] = classify_response(q, p, g_resp)

        if s_resp and not str(s_resp).startswith("ERROR"):
            row["sarvam_label"] = classify_response(q, p, s_resp)

        if row.get("gemini_label") and row.get("sarvam_label"):
            scored_count += 1

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nScoring complete!")
    print(f"Rows with both labels: {scored_count}")
    print(f"Saved to: {DATA_PATH}")

    # Label summary
    gemini_labels = [r["gemini_label"] for r in data if r.get("gemini_label")]
    sarvam_labels = [r["sarvam_label"] for r in data if r.get("sarvam_label")]

    from collections import Counter
    gc = Counter(gemini_labels)
    sc = Counter(sarvam_labels)

    print("\nPhase 2 Label Distribution (Detector Predictions):")
    print(f"{'Label':<15} {'Gemini':>10} {'Sarvam':>10}")
    print("-" * 40)
    for l in ["correct", "hallucinated", "refused"]:
        print(f"{l:<15} {gc.get(l, 0):>10} {sc.get(l, 0):>10}")

    print("\nNEXT STEP: python scripts/phase2_analysis.py")


if __name__ == "__main__":
    main()
