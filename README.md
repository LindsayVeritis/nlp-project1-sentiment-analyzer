# Sentiment Analyzer NLP Final Project 

> **Course:** Natural Language Processing

Classifies any text as **POSITIVE**, **NEGATIVE**, or **NEUTRAL** using a pre-trained DistilBERT transformer model from HuggingFace. No training required, just a clean inference pipeline.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Interactive CLI (type text, get sentiment)

```bash
python main.py
```

### 3. Classify a single string

```bash
python main.py --text "I loved this movie!"
# Label: POSITIVE  |  Confidence: 99.84%
```

### 4. Evaluate on labeled CSV

```bash
python main.py --eval test_examples.csv
# Prints accuracy and saves results.csv
```

### 5. Launch Gradio web UI

```bash
python app.py
# Open http://localhost:7860
```

---

## Project Structure

```
sentiment_analyzer/
├── main.py            # Core model + CLI
├── app.py             # Gradio web UI
├── test_examples.csv  # 50 hand-labeled test cases
├── results.csv        # Generated after running --eval
├── requirements.txt
└── README.md
```

---

## Model

| Detail       | Value |
|-------------|-------|
| Model        | `distilbert-base-uncased-finetuned-sst-2-english` |
| Source       | HuggingFace Hub |
| Task         | Sequence Classification (SST-2) |
| Raw labels   | POSITIVE / NEGATIVE |
| Neutral rule | Score < 0.60 → mapped to NEUTRAL |

The model is downloaded automatically on first run and cached locally.

---

## Evaluation

The `test_examples.csv` file contains 50 hand-labeled examples:
- 15 clearly POSITIVE (movie reviews & product listings)
- 20 clearly NEGATIVE
- 15 NEUTRAL / mixed

Run evaluation:

```bash
python main.py --eval test_examples.csv
```

This generates `results.csv` with columns:

| text | true_label | pred_label | score | correct |
|------|------------|------------|-------|---------|

**Target:** >85% agreement with hand labels.

---

## Neutral Handling

DistilBERT SST-2 is a binary classifier (POSITIVE/NEGATIVE). Neutral is derived from confidence:

- Score ≥ 0.60 → use raw label (POSITIVE or NEGATIVE)
- Score < 0.60 → override to NEUTRAL

---

## Resources

- [HuggingFace Pipelines Docs](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [Gradio Quickstart](https://www.gradio.app/guides/quickstart)
