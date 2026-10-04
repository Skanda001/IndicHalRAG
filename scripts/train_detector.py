"""
train_detector.py
=================

Trains a binary hallucination detector on the annotated data.

Model: mBERT (bert-base-multilingual-cased)

Input:
    question + passage + model_response

Output:
    hallucinated (1) / not hallucinated (0)

Label mapping:
    correct      → 0
    partial      → 0
    refused      → 0
    hallucinated → 1

We train TWO separate detectors:
    1. Detector for Gemini responses
    2. Detector for Sarvam responses

And ONE combined detector.

Run:
    python scripts/train_detector.py

Output:
    models/detector_gemini/
    models/detector_sarvam/
    models/detector_combined/
    results/detector_evaluation.json
"""

import json
import os
import random
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from sklearn.utils.class_weight import compute_class_weight


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH     = "data/annotation/annotated_outputs.json"
RESULTS_PATH   = "results/detector_evaluation.json"

MODEL_NAME     = "bert-base-multilingual-cased"

MAX_LEN        = 256
BATCH_SIZE     = 16
EPOCHS         = 4
LR             = 2e-5

SEED           = 42
TRAIN_SPLIT    = 0.8


# ── Directories ───────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Reproducibility ───────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ── Label mapping ─────────────────────────────────────────────────────────────

def label_to_binary(label: str) -> int:
    """
    Convert 4-class label to binary:

    hallucinated = 1
    correct      = 0
    partial      = 0
    refused      = 0
    """

    if label == "hallucinated":
        return 1

    return 0


# ── Dataset class ─────────────────────────────────────────────────────────────

class HallucinationDataset(Dataset):

    def __init__(
        self,
        rows,
        tokenizer,
        response_key
    ):

        self.rows = rows
        self.tokenizer = tokenizer
        self.response_key = response_key


    def __len__(self):

        return len(self.rows)


    def __getitem__(self, idx):

        row = self.rows[idx]

        question = row.get(
            "question",
            ""
        )

        passage = row.get(
            "passage",
            ""
        )

        response = row.get(
            self.response_key,
            ""
        )

        label_str = row.get(
            "gemini_label"
            if self.response_key == "gemini_response"
            else "sarvam_label",
            "correct"
        )

        # Handle combined detector which uses "label" key directly
        if self.response_key == "combined_response":
            label_str = row.get("combined_label", "correct")

        label = label_to_binary(
            label_str
        )

        # Input:
        # question + passage + response

        text_a = (
            f"{question} [SEP] {passage}"
        )

        text_b = response

        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(),

            "attention_mask":
                encoding["attention_mask"].squeeze(),

            "label":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }


# ── Training function ─────────────────────────────────────────────────────────

def train_and_evaluate(
    rows,
    tokenizer,
    response_key,
    model_save_path,
    name
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Training detector: {name}"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"Total rows: {len(rows)}"
    )


    # ── Count labels ──────────────────────────────────────────────────────────

    def _get_label(r):
        if response_key == "combined_response":
            return r.get("combined_label", "correct")
        elif response_key == "gemini_response":
            return r.get("gemini_label", "correct")
        else:
            return r.get("sarvam_label", "correct")

    labels = [label_to_binary(_get_label(r)) for r in rows]

    pos = sum(labels)
    neg = len(labels) - pos

    print(
        f"Hallucinated (1): "
        f"{pos} "
        f"({pos / len(labels) * 100:.1f}%)"
    )

    print(
        f"Not hallucinated (0): "
        f"{neg} "
        f"({neg / len(labels) * 100:.1f}%)"
    )


    # ── Train/test split ─────────────────────────────────────────────────────

    random.shuffle(rows)

    split_idx = int(
        len(rows) * TRAIN_SPLIT
    )

    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]


    # ── Class weights ─────────────────────────────────────────────────────────
    #
    # IMPORTANT:
    # Calculate weights using TRAINING DATA ONLY.
    # Do not use the test set.

    train_labels_array = np.array([
        label_to_binary(_get_label(r))
        for r in train_rows
    ])


    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels_array
    )


    weight_tensor = torch.FloatTensor(
        class_weights
    ).to(device)


    print(
        f"\nClass weights:"
    )

    print(
        f"  Not hallucinated (0): "
        f"{class_weights[0]:.4f}"
    )

    print(
        f"  Hallucinated (1): "
        f"{class_weights[1]:.4f}"
    )


    print(
        f"\nTrain: {len(train_rows)} | "
        f"Test: {len(test_rows)}"
    )


    # ── Datasets ──────────────────────────────────────────────────────────────

    train_ds = HallucinationDataset(
        train_rows,
        tokenizer,
        response_key
    )

    test_ds = HallucinationDataset(
        test_rows,
        tokenizer,
        response_key
    )


    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ── Model ─────────────────────────────────────────────────────────────────

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)


    # ── Optimizer ────────────────────────────────────────────────────────────

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        eps=1e-8
    )


    # ── Scheduler ────────────────────────────────────────────────────────────

    total_steps = (
        len(train_dl) * EPOCHS
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )


    # ── Weighted loss ─────────────────────────────────────────────────────────

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weight_tensor
    )


    # ── Training loop ─────────────────────────────────────────────────────────

    best_f1 = -1.0


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0


        for batch in train_dl:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attn_mask = batch[
                "attention_mask"
            ].to(device)

            labels_t = batch[
                "label"
            ].to(device)


            model.zero_grad()


            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )


            # Weighted CrossEntropyLoss
            loss = loss_fn(
                outputs.logits,
                labels_t
            )


            total_loss += loss.item()


            # Backpropagation
            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            optimizer.step()

            scheduler.step()


        avg_loss = (
            total_loss /
            len(train_dl)
        )


        # ── Evaluation ───────────────────────────────────────────────────────

        model.eval()

        all_preds = []
        all_labels = []


        with torch.no_grad():

            for batch in test_dl:

                input_ids = batch[
                    "input_ids"
                ].to(device)

                attn_mask = batch[
                    "attention_mask"
                ].to(device)

                labels_t = batch[
                    "label"
                ].to(device)


                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask
                )


                preds = torch.argmax(
                    outputs.logits,
                    dim=1
                )


                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    labels_t.cpu().numpy()
                )


        f1 = f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )

        acc = accuracy_score(
            all_labels,
            all_preds
        )


        print(
            f"  Epoch {epoch}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Macro F1: {f1:.4f} | "
            f"Accuracy: {acc:.4f}"
        )


        # ── Save best model ──────────────────────────────────────────────────

        if f1 > best_f1:

            best_f1 = f1

            os.makedirs(
                model_save_path,
                exist_ok=True
            )

            model.save_pretrained(
                model_save_path
            )

            tokenizer.save_pretrained(
                model_save_path
            )

            print(
                f"    ✅ Best model saved "
                f"(F1: {f1:.4f})"
            )


    # ── Final evaluation ─────────────────────────────────────────────────────

    print(
        f"\n--- Final Evaluation: {name} ---"
    )


    model = BertForSequenceClassification.from_pretrained(
        model_save_path
    ).to(device)

    model.eval()


    final_preds = []
    final_labels = []


    with torch.no_grad():

        for batch in test_dl:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attn_mask = batch[
                "attention_mask"
            ].to(device)

            labels_t = batch[
                "label"
            ].to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )


            preds = torch.argmax(
                outputs.logits,
                dim=1
            )


            final_preds.extend(
                preds.cpu().numpy()
            )

            final_labels.extend(
                labels_t.cpu().numpy()
            )


    print(
        classification_report(
            final_labels,
            final_preds,
            target_names=[
                "not_hallucinated",
                "hallucinated"
            ],
            zero_division=0
        )
    )


    # ── Metrics ───────────────────────────────────────────────────────────────

    metrics = {

        "name":
            name,

        "total_rows":
            len(rows),

        "train_rows":
            len(train_rows),

        "test_rows":
            len(test_rows),

        "accuracy":
            round(
                accuracy_score(
                    final_labels,
                    final_preds
                ),
                4
            ),

        "f1_macro":
            round(
                f1_score(
                    final_labels,
                    final_preds,
                    average="macro",
                    zero_division=0
                ),
                4
            ),

        "f1_hallucinated":
            round(
                f1_score(
                    final_labels,
                    final_preds,
                    pos_label=1,
                    average="binary",
                    zero_division=0
                ),
                4
            ),

        "precision":
            round(
                precision_score(
                    final_labels,
                    final_preds,
                    average="macro",
                    zero_division=0
                ),
                4
            ),

        "recall":
            round(
                recall_score(
                    final_labels,
                    final_preds,
                    average="macro",
                    zero_division=0
                ),
                4
            ),

        "model_path":
            model_save_path
    }


    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    print(
        "Loading annotated data..."
    )


    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    # Filter complete rows

    data = [
        r for r in data
        if r.get("gemini_label")
        and r.get("sarvam_label")
    ]


    print(
        f"Complete annotated rows: "
        f"{len(data)}"
    )


    # ── Tokenizer ─────────────────────────────────────────────────────────────

    print(
        f"\nLoading tokenizer: "
        f"{MODEL_NAME}"
    )


    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )


    all_metrics = {}


    # ── Gemini detector ───────────────────────────────────────────────────────

    gemini_rows = [
        r.copy()
        for r in data
    ]


    gemini_metrics = train_and_evaluate(
        gemini_rows,
        tokenizer,
        response_key="gemini_response",
        model_save_path="models/detector_gemini",
        name="Gemini Hallucination Detector"
    )


    all_metrics[
        "gemini_detector"
    ] = gemini_metrics


    # ── Sarvam detector ───────────────────────────────────────────────────────

    sarvam_rows = [
        r.copy()
        for r in data
    ]


    sarvam_metrics = train_and_evaluate(
        sarvam_rows,
        tokenizer,
        response_key="sarvam_response",
        model_save_path="models/detector_sarvam",
        name="Sarvam Hallucination Detector"
    )


    all_metrics[
        "sarvam_detector"
    ] = sarvam_metrics


    # ── Combined detector ─────────────────────────────────────────────────────

    combined_training_rows = []


    for r in data:

        combined_training_rows.append({
            **r,
            "combined_response": r["gemini_response"],
            "combined_label": r["gemini_label"]
        })


        combined_training_rows.append({
            **r,
            "combined_response": r["sarvam_response"],
            "combined_label": r["sarvam_label"]
        })


    print(
        f"\n{'=' * 60}"
    )

    print(
        "Training COMBINED detector "
        "(Gemini + Sarvam rows)"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"Combined rows: "
        f"{len(combined_training_rows)}"
    )


    combined_metrics = train_and_evaluate(
        combined_training_rows,
        tokenizer,
        response_key="combined_response",
        model_save_path="models/detector_combined",
        name="Combined Hallucination Detector"
    )


    all_metrics[
        "combined_detector"
    ] = combined_metrics


    # ── Save metrics ──────────────────────────────────────────────────────────

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_metrics,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        f"\n✅ All evaluation results saved → "
        f"{RESULTS_PATH}"
    )


    # ── Final summary ─────────────────────────────────────────────────────────

    print(
        "\n" + "=" * 60
    )

    print(
        "DETECTOR TRAINING COMPLETE — SUMMARY"
    )

    print(
        "=" * 60
    )


    for key, m in all_metrics.items():

        print(
            f"\n{m['name']}:"
        )

        print(
            f"  Accuracy       : "
            f"{m['accuracy']}"
        )

        print(
            f"  Macro F1       : "
            f"{m['f1_macro']}"
        )

        print(
            f"  Halluc F1      : "
            f"{m['f1_hallucinated']}"
        )

        print(
            f"  Precision      : "
            f"{m['precision']}"
        )

        print(
            f"  Recall         : "
            f"{m['recall']}"
        )

        print(
            f"  Saved to       : "
            f"{m['model_path']}"
        )


    print(
        "\nNEXT STEP: "
        "python scripts/build_faiss_index.py"
    )

    print(
        "           Then: "
        "python scripts/retrieve_real.py"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
