"""
app.py
Interactive Streamlit Application for Selective Prediction & Abstention in RAG.

Academic Research Project:
Knowing When Not to Answer: Abstention in Retrieval-Augmented Question Answering
ENSA Tétouan — Université Abdelmalek Essaâdi (2026–2027)
Authors: Youssef Ech-Chahlaoui & Zouhair Fezzaz
Supervisor: Prof. Anas Belcaid
"""

import os
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import streamlit as st

# Ensure repository root is in Python module path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import create_pipeline, RAGAbstentionPipeline, FEATURES

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Abstention Demo | ENSA Tétouan",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS Styling ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-answerable {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #31C48D;
    }
    .badge-unanswerable {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #F98080;
    }
    .passage-card {
        background-color: #F9FAFB;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .stat-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Pipeline Caching ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BM25 index & RoBERTa QA reader...")
def get_pipeline():
    """Load and cache the RAGAbstentionPipeline."""
    return create_pipeline()


# ── Curated Test Examples ──────────────────────────────────────────────────────
PRESET_EXAMPLES = [
    {
        "label": "🟢 Answerable: Melbourne Airport",
        "question": "What is another name for Melborne Airport?",
        "type": "Answerable",
        "description": "Direct factual question with strong evidence in the Melbourne corpus.",
    },
    {
        "label": "🟢 Answerable: Historic Partnership",
        "question": "When did the partnership between Jones and Appleton end?",
        "type": "Answerable",
        "description": "Date extraction with clear temporal evidence in the text.",
    },
    {
        "label": "🟢 Answerable: DATANET 1 Network",
        "question": "What was DATANET 1",
        "type": "Answerable",
        "description": "Entity definition with high lexical and semantic overlap.",
    },
    {
        "label": "🔴 Unanswerable (Adversarial): Capital Punishment",
        "question": "How many execution methods are currently outlawed in the United States?",
        "type": "Unanswerable",
        "description": "Adversarial SQuAD 2.0 query with high BM25 match, but the text never states which methods are outlawed.",
    },
    {
        "label": "🔴 Unanswerable (Adversarial): Comprehensive School",
        "question": "What two kinds of schools were developed in response to the costs of post-18 comprehensive school education?",
        "type": "Unanswerable",
        "description": "Passage discusses comprehensive schools, but not post-18 cost responses. Baseline reader hallucinates an answer.",
    },
    {
        "label": "🔴 Out-of-Corpus / Nonsense",
        "question": "Who was the supreme commander of the Martian fleet in the year 2341?",
        "type": "Unanswerable",
        "description": "Query completely outside the Wikipedia corpus. Tests zero-retrieval handling.",
    },
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/ENSA_T%C3%A9touan-Research_Report-blue?style=for-the-badge",
        use_container_width=True,
    )
    st.title("⚙️ Control Panel")

    # Strategy Selection
    st.subheader("1. Abstention Strategy")
    strategy_choice = st.radio(
        "Select Decision Mechanism:",
        options=[
            "Answerability Classifier (Recommended)",
            "BM25 Threshold Heuristic",
            "Always Answer (Baseline)",
        ],
        index=0,
        help="Choose how the pipeline decides whether to answer or abstain.",
    )

    strategy_map = {
        "Answerability Classifier (Recommended)": "classifier",
        "BM25 Threshold Heuristic": "bm25_threshold",
        "Always Answer (Baseline)": "always_answer",
    }
    selected_strategy = strategy_map[strategy_choice]

    # Threshold Controls
    st.subheader("2. Decision Threshold")
    if selected_strategy == "classifier":
        threshold = st.slider(
            "Confidence Threshold (tau):",
            min_value=0.05,
            max_value=0.95,
            value=0.28,
            step=0.01,
            help="Optimal tau = 0.28 selected via grid search on validation data to maximize F1.",
        )
        st.caption(
            "💡 **Optimal validation threshold:** `tau = 0.28` (balances coverage and false-answer rate)."
        )
    elif selected_strategy == "bm25_threshold":
        threshold = st.slider(
            "BM25 Score Cutoff:",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Abstain if top BM25 retrieval score is below this cutoff.",
        )
    else:
        threshold = 0.0
        st.info("The Always Answer baseline operates at 100% coverage with zero abstention.")

    # Retrieval Top-k
    st.subheader("3. Retrieval Top-k")
    top_k = st.slider("Passages to retrieve (k):", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 🎓 Academic Information")
    st.markdown(
        """
        - **Authors:** Youssef Ech-Chahlaoui & Zouhair Fezzaz  
        - **Supervisor:** Prof. Anas Belcaid  
        - **Institution:** ENSA Tétouan (UAE)  
        - **Year:** 2026–2027  
        - **Model:** `deepset/roberta-base-squad2` + Okapi BM25
        """
    )


# ── Main Content Header ───────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🛡️ Knowing When Not to Answer: Abstention in RAG</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Selective Prediction & Hallucination Mitigation in Retrieval-Augmented Question Answering</div>',
    unsafe_allow_html=True,
)

# Load pipeline
try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()


# ── Query Input Section ───────────────────────────────────────────────────────
col_query, col_preset = st.columns([1.6, 1.4], gap="large")

with col_preset:
    st.markdown("##### 📌 Benchmark Test Examples")
    preset_names = [ex["label"] for ex in PRESET_EXAMPLES]
    selected_preset_label = st.selectbox(
        "Select a curated SQuAD 2.0 query:",
        options=preset_names,
        index=0,
    )
    current_preset = next(p for p in PRESET_EXAMPLES if p["label"] == selected_preset_label)
    st.caption(f"**Type:** `{current_preset['type']}` — *{current_preset['description']}*")

with col_query:
    st.markdown("##### 🔍 Query Input")
    default_question = current_preset["question"]
    query_text = st.text_area(
        "Enter query text:",
        value=default_question,
        height=95,
        placeholder="Type a factual or unanswerable question...",
    )
    run_button = st.button("🚀 Analyze & Query Pipeline", type="primary", use_container_width=True)


# ── Inference Execution ───────────────────────────────────────────────────────
if run_button or query_text:
    with st.spinner("Executing BM25 retrieval, RoBERTa reading, and feature evaluation..."):
        start_time = time.time()
        result = pipeline.predict(
            question=query_text,
            top_k=top_k,
            strategy=selected_strategy,
            threshold=threshold,
        )
        elapsed_sec = time.time() - start_time

    st.markdown("---")

    # ── Decision Banner ───────────────────────────────────────────────────────
    if result.status == "ANSWERED":
        st.success(
            f"### ✅ Pipeline Decision: ANSWER ACCEPTED\n\n"
            f"**Extracted Answer:** `{result.answer}`\n\n"
            f"*The model judged the evidence sufficient with decision confidence **{result.decision_probability:.4f}** (Threshold: {result.threshold:.2f}).*",
            icon="✅",
        )
    else:
        # Determine exact reason for abstention/empty output
        if selected_strategy == "always_answer":
            reason_text = (
                f"**Strategy Note:** You selected **Always Answer (Baseline)**, which does not gate or withhold queries "
                f"(Threshold: `{result.threshold:.2f}`). However, the RoBERTa reader found **no answer span** "
                f"in the top-{top_k} retrieved Wikipedia passages (it assigned high confidence to the empty/impossible span). "
                f"Because the text does not contain the answer, no text could be returned."
            )
        elif selected_strategy == "bm25_threshold":
            reason_text = (
                f"**Gating Reason:** Top BM25 retrieval score was **{result.top_bm25_score:.2f}**, which is below "
                f"the BM25 cutoff threshold of **{result.threshold:.2f}**."
            )
        else:  # classifier
            if result.decision_probability < result.threshold:
                reason_text = (
                    f"**Gating Reason:** The Answerability Classifier predicted $P(\\text{{is\\_answerable}}) = "
                    f"\\mathbf{{{result.decision_probability:.4f}}}$, which falls below the decision threshold $\\tau = \\mathbf{{{result.threshold:.2f}}}$. "
                    f"The answer was withheld to prevent a hallucinated or false answer."
                )
            else:
                reason_text = (
                    f"**Reader Note:** The classifier allowed answering ($P = {result.decision_probability:.4f} \\ge {result.threshold:.2f}$), "
                    f"but the neural reader extracted an empty span from the retrieved context."
                )

        st.warning(
            f"### 🛑 Pipeline Decision: NO ANSWER RETURNED (∅)\n\n"
            f"{reason_text}",
            icon="🛑",
        )

    # ── Metric Cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Status", result.status)
    with m2:
        st.metric("P(Answerable)", f"{result.decision_probability:.3f}")
    with m3:
        st.metric("Threshold (tau)", f"{result.threshold:.2f}")
    with m4:
        st.metric("Top BM25 Score", f"{result.top_bm25_score:.2f}")
    with m5:
        st.metric("Latency", f"{elapsed_sec:.2f}s")

    # ── Detailed Tabs ─────────────────────────────────────────────────────────
    tab_features, tab_passages, tab_benchmark = st.tabs(
        ["📊 11 Engineered Features", "📚 Retrieved Passages (Top-k)", "📈 Benchmark Performance"]
    )

    with tab_features:
        st.markdown("#### Operationalized Feature Vector")
        st.markdown(
            "These 11 signals combine retrieval dynamics (BM25) and reader uncertainty (RoBERTa) to drive the abstention decision."
        )

        feat_dict = result.features
        feat_df = pd.DataFrame(
            [
                {
                    "Feature": k,
                    "Value": round(v, 4) if pd.notna(v) else "Imputed (Median)",
                    "Category": "Retrieval" if "bm25" in k or "source" in k else "Reader / Query",
                }
                for k, v in feat_dict.items()
            ]
        )

        col_table, col_chart = st.columns([1.2, 1.8], gap="medium")
        with col_table:
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        with col_chart:
            # Highlight reader probabilities and retrieval scores
            st.markdown("##### Key Signal Breakdown")
            chart_data = pd.DataFrame(
                {
                    "Signal": ["QA Confidence", "QA No-Answer Prob", "P(Is Answerable)"],
                    "Probability": [
                        feat_dict["qa_score"],
                        feat_dict["qa_no_answer_probability"],
                        result.decision_probability,
                    ],
                }
            ).set_index("Signal")
            st.bar_chart(chart_data)

    with tab_passages:
        st.markdown(f"#### Top-{len(result.retrieved_passages)} Retrieved Passages")
        if not result.retrieved_passages:
            st.info("No passages were retrieved.")
        else:
            for p in result.retrieved_passages:
                is_best = (
                    result.best_passage is not None
                    and p["paragraph_id"] == result.best_passage["paragraph_id"]
                )
                border_color = "#10B981" if is_best else "#3B82F6"
                best_tag = " ⭐ (Selected Candidate Context)" if is_best else ""

                st.markdown(
                    f"""
                    <div style="border-left: 5px solid {border_color}; background-color: #F8FAFC; padding: 12px; margin-bottom: 12px; border-radius: 6px;">
                        <div style="display: flex; justify-content: space-between; font-weight: 600; color: #1E293B;">
                            <span>Rank #{p['rank']} — Article: {p['article_title']}{best_tag}</span>
                            <span style="color: #2563EB;">BM25 Score: {p['bm25_score']:.2f}</span>
                        </div>
                        <p style="margin-top: 8px; color: #334155; font-size: 0.95rem; line-height: 1.5;">{p['context']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_benchmark:
        st.markdown("#### Test Set Comparative Evaluation (1,200 Held-Out Questions)")
        st.markdown(
            "Empirical results obtained on the strictly held-out test partition (600 answerable, 600 unanswerable):"
        )

        benchmark_df = pd.DataFrame(
            [
                {
                    "Strategy": "Always Answer (Baseline)",
                    "Precision": "0.4820",
                    "Recall": "1.0000",
                    "F1 Score": "0.6505",
                    "Coverage": "1.0000 (100%)",
                    "False-Answer Rate (FAR)": "1.0000 (100%)",
                    "False-Abstention Rate (FAbR)": "0.0000 (0%)",
                },
                {
                    "Strategy": "BM25 Threshold (tau)",
                    "Precision": "0.4810",
                    "Recall": "0.9959",
                    "F1 Score": "0.6486",
                    "Coverage": "0.9980 (99.8%)",
                    "False-Answer Rate (FAR)": "1.0000 (100%)",
                    "False-Abstention Rate (FAbR)": "0.0042 (0.4%)",
                },
                {
                    "Strategy": "Answerability Classifier (tau=0.28)",
                    "Precision": "0.8231",
                    "Recall": "0.8880",
                    "F1 Score": "0.8543",
                    "Coverage": "0.5200 (52%)",
                    "False-Answer Rate (FAR)": "0.1776 (17.8%)",
                    "False-Abstention Rate (FAbR)": "0.1120 (11.2%)",
                },
            ]
        )
        st.table(benchmark_df)
        st.caption(
            "⚡ **Key Finding:** The Answerability Classifier achieves an **82.2% reduction in False-Answer Rate** while increasing answer precision from 48.2% to 82.3%."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ENSA Tétouan — Université Abdelmalek Essaâdi | Academic Research Project 2026–2027 | Supervised by Prof. Anas Belcaid</small></center>",
    unsafe_allow_html=True,
)
