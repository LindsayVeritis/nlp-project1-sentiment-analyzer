"""
Sentiment Analyzer — Project 1
NLP Course | Dr. Priyamvada Tripathi

Uses HuggingFace's distilbert-base-uncased-finetuned-sst-2-english pipeline
to classify text as POSITIVE or NEGATIVE with a confidence score.
"""

import sys
import csv
import argparse
from transformers import pipeline

# ── Model ──────────────────────────────────────────────────────────────────────

def load_model():
    """Load the HuggingFace sentiment-analysis pipeline (cached after first run)."""
    print("Loading model (this may take a moment on first run)…")
    clf = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    print("Model ready.\n")
    return clf


# ── Core function ──────────────────────────────────────────────────────────────

def analyze(clf, text: str) -> dict:
    """
    Classify a single piece of text.

    Returns
    -------
    dict with keys:
        text   – the original input
        label  – POSITIVE | NEGATIVE | NEUTRAL
        score  – confidence (0.0 – 1.0)
    """
    text = text.strip()
    if not text:
        return {"text": text, "label": "NEUTRAL", "score": 1.0}

    result = clf(text, truncation=True, max_length=512)[0]

    # cardiffnlp model returns "positive", "negative", "neutral" (lowercase)
    label = result["label"].upper()
    score = round(result["score"], 4)

    return {"text": text, "label": label, "score": score}


def analyze_batch(clf, texts: list[str]) -> list[dict]:
    """Classify a list of texts and return a list of result dicts."""
    return [analyze(clf, t) for t in texts]


# ── CLI mode ───────────────────────────────────────────────────────────────────

def run_interactive(clf):
    """REPL: type text, get sentiment, repeat."""
    print("=" * 55)
    print("  Sentiment Analyzer  |  type 'quit' to exit")
    print("=" * 55)
    while True:
        try:
            text = input("\nEnter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            continue
        r = analyze(clf, text)
        emoji = {"POSITIVE": "😊", "NEGATIVE": "😠", "NEUTRAL": "😐"}[r["label"]]
        print(f"  {emoji}  {r['label']}  (confidence: {r['score']:.2%})")


# ── Evaluation mode ────────────────────────────────────────────────────────────

def evaluate(clf, csv_path: str):
    """
    Read a CSV with columns [text, true_label] and print accuracy.
    Writes predictions back to results.csv.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    correct = 0
    output_rows = []
    for row in rows:
        pred = analyze(clf, row["text"])
        match = pred["label"].upper() == row["true_label"].strip().upper()
        correct += match
        output_rows.append({
            "text":        row["text"],
            "true_label":  row["true_label"].strip().upper(),
            "pred_label":  pred["label"],
            "score":       pred["score"],
            "correct":     "YES" if match else "NO",
        })

    accuracy = correct / len(rows) * 100
    print(f"\nAccuracy: {correct}/{len(rows)} = {accuracy:.1f}%")

    out_path = "results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "true_label", "pred_label", "score", "correct"]
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Results saved → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analyzer — NLP Project 1"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Classify a single text string and exit.",
    )
    parser.add_argument(
        "--eval", "-e",
        type=str,
        metavar="CSV_PATH",
        help="Evaluate on a labeled CSV (columns: text, true_label).",
    )
    args = parser.parse_args()

    clf = load_model()

    if args.text:
        r = analyze(clf, args.text)
        print(f"Label: {r['label']}  |  Confidence: {r['score']:.2%}")
    elif args.eval:
        evaluate(clf, args.eval)
    else:
        run_interactive(clf)


if __name__ == "__main__":
    main()
