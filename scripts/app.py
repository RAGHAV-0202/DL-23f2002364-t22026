"""Streamlit UI for the Smart MCQ Solver (DeBERTa-v3 pairwise scorer).

Loads the fine-tuned DebertaV2ForSequenceClassification model from
`models/deberta_v3_full`, scores each (prompt, option) pair, and returns
the top-3 ranked answers by MAP@3-style probability.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent.parent / "models" / "deberta_v3_full"
OPTION_LETTERS = ["A", "B", "C", "D", "E"]

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠")


@st.cache_resource(show_spinner="Loading DeBERTa model…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def score_options(prompt: str, options: dict[str, str], tokenizer, model):
    """Return {letter: prob} where prob is the model's positive-class score."""
    letters = [ltr for ltr, txt in options.items() if txt.strip()]
    texts_a = [prompt] * len(letters)
    texts_b = [options[ltr] for ltr in letters]

    enc = tokenizer(
        texts_a,
        texts_b,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    logits = model(**enc).logits
    # Positive-class probability (label == 1 means "correct option").
    if logits.shape[-1] == 1:
        probs = torch.sigmoid(logits.squeeze(-1))
    else:
        probs = F.softmax(logits, dim=-1)[:, -1]
    return {ltr: float(p) for ltr, p in zip(letters, probs)}


st.title("🧠 Smart MCQ Solver")
st.caption("Fine-tuned DeBERTa-v3 · predicts the top-3 most probable answers (MAP@3)")

tokenizer, model = load_model()

prompt = st.text_area("Question", height=100, placeholder="Enter the MCQ question…")

st.write("**Options** (leave unused options blank)")
options = {}
for letter in OPTION_LETTERS:
    options[letter] = st.text_input(f"Option {letter}", key=f"opt_{letter}")

if st.button("Solve", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a question.")
    elif sum(bool(v.strip()) for v in options.values()) < 2:
        st.warning("Please enter at least two options.")
    else:
        scores = score_options(prompt, options, tokenizer, model)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        st.subheader("Top-3 predictions")
        for rank, (letter, prob) in enumerate(ranked[:3], start=1):
            st.markdown(f"**{rank}. Option {letter}** — {options[letter]}  \n`score: {prob:.4f}`")

        with st.expander("All scores"):
            for letter, prob in ranked:
                st.write(f"{letter}: {prob:.4f} — {options[letter]}")
        st.code(" ".join(ltr for ltr, _ in ranked[:3]), language=None)
