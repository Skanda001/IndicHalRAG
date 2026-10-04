"""
phase1_analysis.py
===================

Performs complete Phase 1 analysis:

1. Hallucination rates for Gemini vs Sarvam
2. Label distribution comparison
3. Per-label breakdown
4. Saves charts and results

Run:
    python scripts/phase1_analysis.py

Output:
    results/phase1_hallucination_rates.json
    results/phase1_charts/  (PNG charts)
"""

import json
import os
from collections import Counter

os.makedirs("results", exist_ok=True)
os.makedirs("results/phase1_charts", exist_ok=True)

INPUT_PATH   = "data/annotation/annotated_outputs.json"
RESULTS_PATH = "results/phase1_hallucination_rates.json"

# ── Load annotated data ───────────────────────────────────────────────
print("Loading annotated data...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
print(f"Total annotated rows: {total}")

# ── Count labels ──────────────────────────────────────────────────────
gemini_labels = [r["gemini_label"] for r in data if r.get("gemini_label")]
sarvam_labels = [r["sarvam_label"] for r in data if r.get("sarvam_label")]

gemini_counts = Counter(gemini_labels)
sarvam_counts = Counter(sarvam_labels)

LABELS = ["correct", "hallucinated", "partial", "refused"]

def pct(count, total):
    return round(count / total * 100, 2) if total > 0 else 0

# ── Print results ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("PHASE 1 RESULTS — GOLD CONTEXT HALLUCINATION ANALYSIS")
print("="*65)

print(f"\nTotal Questions: {total}")
print(f"Total Responses: {total * 2} (both models combined)")

print(f"\n{'Label':<15} {'Gemini':>12} {'Gemini%':>10} {'Sarvam':>10} {'Sarvam%':>10}")
print("-"*65)
for label in LABELS:
    gc = gemini_counts.get(label, 0)
    sc = sarvam_counts.get(label, 0)
    gp = pct(gc, total)
    sp = pct(sc, total)
    print(f"{label:<15} {gc:>12} {gp:>9.1f}% {sc:>10} {sp:>9.1f}%")

# ── Key metrics ───────────────────────────────────────────────────────
gemini_halluc_rate = pct(gemini_counts.get("hallucinated", 0), total)
sarvam_halluc_rate = pct(sarvam_counts.get("hallucinated", 0), total)

gemini_correct_rate = pct(gemini_counts.get("correct", 0), total)
sarvam_correct_rate = pct(sarvam_counts.get("correct", 0), total)

# Combined hallucination (hallucinated + partial)
gemini_combined = gemini_counts.get("hallucinated", 0) + gemini_counts.get("partial", 0)
sarvam_combined = sarvam_counts.get("hallucinated", 0) + sarvam_counts.get("partial", 0)

print("\n" + "="*65)
print("KEY FINDINGS")
print("="*65)
print(f"\nGemini  hallucination rate : {gemini_halluc_rate}%")
print(f"Sarvam  hallucination rate : {sarvam_halluc_rate}%")
print(f"\nGemini  correct rate       : {gemini_correct_rate}%")
print(f"Sarvam  correct rate       : {sarvam_correct_rate}%")
print(f"\nGemini  combined bad rate  : {pct(gemini_combined, total)}%")
print(f"  (hallucinated + partial)")
print(f"Sarvam  combined bad rate  : {pct(sarvam_combined, total)}%")
print(f"  (hallucinated + partial)")

if gemini_halluc_rate > sarvam_halluc_rate:
    diff = round(gemini_halluc_rate - sarvam_halluc_rate, 2)
    print(f"\n→ Sarvam hallucinates LESS than Gemini by {diff}%")
    print(f"  Indic-specialized training shows benefit for Kannada")
elif sarvam_halluc_rate > gemini_halluc_rate:
    diff = round(sarvam_halluc_rate - gemini_halluc_rate, 2)
    print(f"\n→ Gemini hallucinates LESS than Sarvam by {diff}%")
    print(f"  General frontier model performs better on Kannada")
else:
    print(f"\n→ Both models hallucinate at similar rates")

# ── Save results ──────────────────────────────────────────────────────
results = {
    "phase": "Phase 1 - Gold Context",
    "total_questions": total,
    "gemini": {
        label: {
            "count": gemini_counts.get(label, 0),
            "percentage": pct(gemini_counts.get(label, 0), total)
        }
        for label in LABELS
    },
    "sarvam": {
        label: {
            "count": sarvam_counts.get(label, 0),
            "percentage": pct(sarvam_counts.get(label, 0), total)
        }
        for label in LABELS
    },
    "key_metrics": {
        "gemini_hallucination_rate": gemini_halluc_rate,
        "sarvam_hallucination_rate": sarvam_halluc_rate,
        "gemini_correct_rate": gemini_correct_rate,
        "sarvam_correct_rate": sarvam_correct_rate,
        "better_model": "sarvam" if sarvam_halluc_rate < gemini_halluc_rate else "gemini"
    }
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Results saved → {RESULTS_PATH}")

# ── Generate charts ───────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import numpy as np

    # Chart 1 — Label Distribution Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(LABELS))
    width = 0.35

    gemini_vals = [pct(gemini_counts.get(l, 0), total) for l in LABELS]
    sarvam_vals = [pct(sarvam_counts.get(l, 0), total) for l in LABELS]

    bars1 = ax.bar(x - width/2, gemini_vals, width,
                   label="Gemini 2.5 Flash", color="#4285F4", alpha=0.85)
    bars2 = ax.bar(x + width/2, sarvam_vals, width,
                   label="Sarvam-105B", color="#FF6B35", alpha=0.85)

    ax.set_xlabel("Label Category", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Phase 1: Label Distribution — Gemini vs Sarvam\n(Gold Context, Kannada QA)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in LABELS], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/phase1_charts/label_distribution.png", dpi=150)
    plt.close()
    print("✅ Chart saved → results/phase1_charts/label_distribution.png")

    # Chart 2 — Hallucination Rate Only (Simple Bar)
    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Gemini 2.5 Flash", "Sarvam-105B"]
    halluc_rates = [gemini_halluc_rate, sarvam_halluc_rate]
    colors = ["#4285F4", "#FF6B35"]

    bars = ax.bar(models, halluc_rates, color=colors, alpha=0.85, width=0.4)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=12)
    ax.set_title("Phase 1: Hallucination Rate\nGemini vs Sarvam — Kannada Gold Context",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, halluc_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/phase1_charts/hallucination_rate.png", dpi=150)
    plt.close()
    print("✅ Chart saved → results/phase1_charts/hallucination_rate.png")

except ImportError:
    print("⚠️  matplotlib not found. Install with: pip install matplotlib")
    print("   Charts not generated but results JSON is saved.")

print("\n" + "="*65)
print("NEXT STEP: python scripts/train_detector.py")
print("="*65)
