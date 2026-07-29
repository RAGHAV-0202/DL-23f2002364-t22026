"""Smart MCQ Solver — Streamlit Cloud Deployment.

Fine-tuned DeBERTa-v3-small with LoRA adapter for multi-choice question
answering. Predicts the top-3 most probable answers ranked by MAP@3.

Author: Raghav Kapoor (23f2002364)
Course: DL & GenAI, IIT Madras BS in Data Science, T2-2026
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# ── Paths ───────────────────────────────────────────────────────────────
ADAPTER_DIR = Path(__file__).parent / "models" / "deberta_lora"
BASE_MODEL = "microsoft/deberta-v3-small"
OPTION_LETTERS = ["A", "B", "C", "D", "E"]

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered",
)

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Project Info")
    st.markdown(
        """
        **Course:** DL & GenAI  
        **Student:** Raghav Kapoor  
        **Roll:** 23f2002364  
        **Model:** DeBERTa-v3-small + LoRA  
        **Metric:** MAP@3  
        """
    )
    st.divider()
    st.markdown(
        "The model scores each *(question, option)* pair independently "
        "and ranks options by predicted probability of being correct."
    )


# ── Model loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DeBERTa + LoRA adapter …")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


# ── Scoring logic ──────────────────────────────────────────────────────
@torch.no_grad()
def score_options(
    prompt: str, options: dict[str, str], tokenizer, model
) -> dict[str, float]:
    letters = [ltr for ltr, txt in options.items() if txt.strip()]
    if not letters:
        return {}
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

    if logits.shape[-1] == 1:
        probs = torch.sigmoid(logits.squeeze(-1))
    else:
        probs = F.softmax(logits, dim=-1)[:, 1]

    return {ltr: float(p) for ltr, p in zip(letters, probs)}


# ── UI ─────────────────────────────────────────────────────────────────
st.title("🧠 Smart MCQ Solver")
st.caption(
    "Fine-tuned DeBERTa-v3-small with LoRA · "
    "Predicts the top-3 most probable answers (MAP@3)"
)

tokenizer, model = load_model()

st.divider()

prompt = st.text_area(
    "📝 Question",
    height=120,
    placeholder="Paste the MCQ question here …",
)

st.markdown("**Options** (leave unused options blank)")
cols = st.columns(5)
options = {}
for i, letter in enumerate(OPTION_LETTERS):
    with cols[i]:
        options[letter] = st.text_input(f"{letter}", key=f"opt_{letter}")

st.divider()

if st.button("🚀 Solve", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a question.")
    elif sum(bool(v.strip()) for v in options.values()) < 2:
        st.warning("Please provide at least two options.")
    else:
        with st.spinner("Scoring options …"):
            scores = score_options(prompt, options, tokenizer, model)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        st.subheader("🏆 Top-3 Predictions")

        for rank, (letter, prob) in enumerate(ranked[:3], start=1):
            medal = ["🥇", "🥈", "🥉"][rank - 1]
            pct = prob * 100
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{medal} Option {letter}:** {options[letter]}")
                st.progress(prob)
            with col2:
                st.metric("Score", f"{pct:.1f}%")

        with st.expander("📊 All Scores"):
            for letter, prob in ranked:
                st.write(f"**{letter}:** {prob:.4f} — {options[letter]}")

        st.code(
            "Prediction: " + " ".join(ltr for ltr, _ in ranked[:3]),
            language=None,
        )
