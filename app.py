"""
app.py — Gradio web UI for the Sentiment Analyzer
Run:  python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from main import load_model, analyze, analyze_batch

clf = load_model()

EMOJI = {"POSITIVE": "😊 POSITIVE", "NEGATIVE": "😠 NEGATIVE", "NEUTRAL": "😐 NEUTRAL"}
COLOR = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#94a3b8"}


def single_predict(text):
    if not text.strip():
        return "—", "—"
    r = analyze(clf, text)
    label_str = EMOJI[r["label"]]
    score_str  = f"{r['score']:.2%}"
    return label_str, score_str


def batch_predict(raw):
    """Accept newline-separated texts; return a markdown table."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return "No input provided."
    results = analyze_batch(clf, lines)
    rows = ["| # | Text | Label | Confidence |", "|---|------|-------|------------|"]
    for i, r in enumerate(results, 1):
        preview = (r["text"][:60] + "…") if len(r["text"]) > 60 else r["text"]
        rows.append(f"| {i} | {preview} | {EMOJI[r['label']]} | {r['score']:.2%} |")
    return "\n".join(rows)


# ── Layout ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Sentiment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Sentiment Analyzer\nPowered by `distilbert-base-uncased-finetuned-sst-2-english`")

    with gr.Tab("Single Text"):
        inp  = gr.Textbox(label="Enter text", placeholder="I loved this movie!", lines=3)
        btn  = gr.Button("Analyze", variant="primary")
        with gr.Row():
            out_label = gr.Textbox(label="Sentiment", interactive=False)
            out_score = gr.Textbox(label="Confidence", interactive=False)
        btn.click(single_predict, inputs=inp, outputs=[out_label, out_score])
        gr.Examples(
            examples=[
                ["I absolutely loved this film — the acting was superb!"],
                ["Terrible experience. The product broke after one day."],
                ["It was okay. Nothing special but not bad either."],
                ["The visuals were stunning but the plot made no sense."],
            ],
            inputs=inp,
        )

    with gr.Tab("Batch Mode"):
        gr.Markdown("Enter one sentence per line:")
        batch_inp = gr.Textbox(
            label="Texts (one per line)",
            lines=8,
            placeholder="Great product!\nHorrible customer service.\nIt was fine.",
        )
        batch_btn = gr.Button("Analyze All", variant="primary")
        batch_out = gr.Markdown()
        batch_btn.click(batch_predict, inputs=batch_inp, outputs=batch_out)

    gr.Markdown(
        "---\n*NLP Course — Project 1 | Model: DistilBERT SST-2*"
    )

if __name__ == "__main__":
    demo.launch()
