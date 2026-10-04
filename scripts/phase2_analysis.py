"""
phase2_analysis.py
===================

Phase 2 analysis: Compares hallucination rates when using REAL RAG
retrieval (FAISS) vs Phase 1 gold context.

Run AFTER:
    python scripts/build_faiss_index.py
    python scripts/retrieve_real.py

Then:
    python scripts/phase2_analysis.py

Input:
    data/real_llm_outputs.json         (Phase 2: FAISS-retrieved passages)
    results/phase1_hallucination_rates.json  (Phase 1 baseline, for comparison)

Output:
    results/phase2_hallucination_rates.json
    results/phase2_charts/  (PNG charts)

KEY RESEARCH QUESTION:
    Does hallucination rate increase when the model gets imperfect RAG
    passages vs the guaranteed gold context?
    i.e., does retrieval quality affect hallucination behavior?
"""

import json
import os
import sys
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.makedirs("results", exist_ok=True)
os.makedirs("results/phase2_charts", exist_ok=True)

INPUT_PATH   = "data/real_llm_outputs.json"
P1_PATH      = "results/phase1_hallucination_rates.json"
RESULTS_PATH = "results/phase2_hallucination_rates.json"

# ── Load Phase 2 outputs ──────────────────────────────────────────────
print("Loading Phase 2 LLM outputs (FAISS-retrieved passages)...")
if not os.path.exists(INPUT_PATH):
    print(f"ERROR: {INPUT_PATH} not found.")
    print("Run: python scripts/retrieve_real.py first")
    exit()

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Only rows that have BOTH labels (i.e., have been annotated)
data = [r for r in data if r.get("gemini_label") and r.get("sarvam_label")]
total = len(data)
print(f"Annotated Phase 2 rows: {total}")

if total == 0:
    print("⚠️  No annotated rows found in Phase 2 output.")
    print("   You need to annotate real_llm_outputs.json first.")
    exit()

# ── Count labels ──────────────────────────────────────────────────────
gemini_labels = [r["gemini_label"] for r in data if r.get("gemini_label")]
sarvam_labels = [r["sarvam_label"] for r in data if r.get("sarvam_label")]

gemini_counts = Counter(gemini_labels)
sarvam_counts = Counter(sarvam_labels)

LABELS = ["correct", "hallucinated", "partial", "refused"]

def pct(count, total):
    return round(count / total * 100, 2) if total > 0 else 0

# ── Load Phase 1 baseline ─────────────────────────────────────────────
p1_gemini_halluc = None
p1_sarvam_halluc = None

if os.path.exists(P1_PATH):
    with open(P1_PATH, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    p1_gemini_halluc = p1["key_metrics"]["gemini_hallucination_rate"]
    p1_sarvam_halluc = p1["key_metrics"]["sarvam_hallucination_rate"]
    print(f"\nPhase 1 baseline loaded:")
    print(f"  Gemini hallucination (gold ctx): {p1_gemini_halluc}%")
    print(f"  Sarvam hallucination (gold ctx): {p1_sarvam_halluc}%")

# ── Retrieval quality stats ───────────────────────────────────────────
gold_retrieved = [r for r in data if r.get("gold_in_retrieved") is True]
gold_not_retrieved = [r for r in data if r.get("gold_in_retrieved") is False]
no_gold = [r for r in data if r.get("gold_in_retrieved") is None]

print(f"\nRetrieval quality:")
print(f"  Gold answer in retrieved passages : {len(gold_retrieved)} ({pct(len(gold_retrieved), total):.1f}%)")
print(f"  Gold answer NOT retrieved          : {len(gold_not_retrieved)} ({pct(len(gold_not_retrieved), total):.1f}%)")
print(f"  No gold answer available           : {len(no_gold)} ({pct(len(no_gold), total):.1f}%)")

# ── Print Phase 2 results ─────────────────────────────────────────────
p2_gemini_halluc = pct(gemini_counts.get("hallucinated", 0), total)
p2_sarvam_halluc = pct(sarvam_counts.get("hallucinated", 0), total)

print("\n" + "="*65)
print("PHASE 2 RESULTS — REAL RAG HALLUCINATION ANALYSIS")
print("="*65)

print(f"\n{'Label':<15} {'Gemini':>12} {'Gemini%':>10} {'Sarvam':>10} {'Sarvam%':>10}")
print("-"*65)
for label in LABELS:
    gc = gemini_counts.get(label, 0)
    sc = sarvam_counts.get(label, 0)
    gp = pct(gc, total)
    sp = pct(sc, total)
    print(f"{label:<15} {gc:>12} {gp:>9.1f}% {sc:>10} {sp:>9.1f}%")

print("\n" + "="*65)
print("PHASE 1 vs PHASE 2 COMPARISON")
print("="*65)

if p1_gemini_halluc is not None:
    g_delta = round(p2_gemini_halluc - p1_gemini_halluc, 2)
    s_delta = round(p2_sarvam_halluc - p1_sarvam_halluc, 2)
    print(f"\n  Gemini hallucination:  Phase1={p1_gemini_halluc}%  Phase2={p2_gemini_halluc}%  Δ={g_delta:+.2f}%")
    print(f"  Sarvam hallucination:  Phase1={p1_sarvam_halluc}%  Phase2={p2_sarvam_halluc}%  Δ={s_delta:+.2f}%")

    if g_delta > 5 or s_delta > 5:
        print(f"\n  → Hallucination INCREASES with real retrieval (imperfect passages)")
        print(f"    Retrieval quality is a significant factor.")
    elif g_delta < -5 or s_delta < -5:
        print(f"\n  → Hallucination DECREASES with real retrieval (unexpected)")
        print(f"    Possible cause: retrieved passages are longer/richer than gold trims.")
    else:
        print(f"\n  → Hallucination rate is SIMILAR across both phases.")
        print(f"    Models are robust to retrieval quality for this dataset.")

# ── Save results ──────────────────────────────────────────────────────
results = {
    "phase": "Phase 2 - Real RAG",
    "total_questions": total,
    "retrieval_stats": {
        "gold_in_retrieved": len(gold_retrieved),
        "gold_not_retrieved": len(gold_not_retrieved),
        "no_gold_answer": len(no_gold),
        "recall_at_k": pct(len(gold_retrieved), len(gold_retrieved) + len(gold_not_retrieved))
    },
    "gemini": {
        label: {"count": gemini_counts.get(label, 0), "percentage": pct(gemini_counts.get(label, 0), total)}
        for label in LABELS
    },
    "sarvam": {
        label: {"count": sarvam_counts.get(label, 0), "percentage": pct(sarvam_counts.get(label, 0), total)}
        for label in LABELS
    },
    "key_metrics": {
        "phase2_gemini_hallucination_rate": p2_gemini_halluc,
        "phase2_sarvam_hallucination_rate": p2_sarvam_halluc,
        "phase1_gemini_hallucination_rate": p1_gemini_halluc,
        "phase1_sarvam_hallucination_rate": p1_sarvam_halluc,
    }
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Results saved → {RESULTS_PATH}")

# ── Generate charts ───────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import numpy as np

    # Chart 1 — Phase 1 vs Phase 2 hallucination comparison
    if p1_gemini_halluc is not None:
        fig, ax = plt.subplots(figsize=(9, 6))

        x = np.arange(2)  # Gemini, Sarvam
        width = 0.3

        phase1_vals = [p1_gemini_halluc, p1_sarvam_halluc]
        phase2_vals = [p2_gemini_halluc, p2_sarvam_halluc]

        bars1 = ax.bar(x - width/2, phase1_vals, width,
                       label="Phase 1 (Gold Context)", color="#4285F4", alpha=0.85)
        bars2 = ax.bar(x + width/2, phase2_vals, width,
                       label="Phase 2 (Real RAG)", color="#EA4335", alpha=0.85)

        ax.set_ylabel("Hallucination Rate (%)", fontsize=12)
        ax.set_title("Phase 1 vs Phase 2: Hallucination Rate\nGold Context vs Real RAG Retrieval",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Gemini 2.5 Flash", "Sarvam-105B"], fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0, max(max(phase1_vals), max(phase2_vals)) + 15)

        for bar in bars1 + bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig("results/phase2_charts/phase1_vs_phase2.png", dpi=150)
        plt.close()
        print("✅ Chart saved → results/phase2_charts/phase1_vs_phase2.png")

    # Chart 2 — Hallucination by retrieval success
    if gold_retrieved and gold_not_retrieved:
        def model_halluc_rate(rows, label_key):
            c = Counter(r.get(label_key) for r in rows)
            t = len(rows)
            return pct(c.get("hallucinated", 0), t)

        cats = ["Gold Retrieved", "Gold NOT Retrieved"]
        g_rates = [model_halluc_rate(gold_retrieved, "gemini_label"),
                   model_halluc_rate(gold_not_retrieved, "gemini_label")]
        s_rates = [model_halluc_rate(gold_retrieved, "sarvam_label"),
                   model_halluc_rate(gold_not_retrieved, "sarvam_label")]

        fig, ax = plt.subplots(figsize=(9, 6))
        x = np.arange(len(cats))
        bars1 = ax.bar(x - width/2, g_rates, width, label="Gemini", color="#4285F4", alpha=0.85)
        bars2 = ax.bar(x + width/2, s_rates, width, label="Sarvam", color="#FF6B35", alpha=0.85)

        ax.set_ylabel("Hallucination Rate (%)", fontsize=12)
        ax.set_title("Hallucination Rate by Retrieval Success\n(Does retrieval quality affect hallucination?)",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 100)

        for bar in bars1 + bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig("results/phase2_charts/hallucination_by_retrieval.png", dpi=150)
        plt.close()
        print("✅ Chart saved → results/phase2_charts/hallucination_by_retrieval.png")

except ImportError:
    print("⚠️  matplotlib not found. Install with: pip install matplotlib")

print("\n" + "="*65)
print("NEXT STEP: python scripts/train_detector.py")
print("="*65)
