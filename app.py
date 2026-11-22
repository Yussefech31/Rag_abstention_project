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
    .passage-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.95rem;
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


# ── Curated Test Examples from 500-Example Split ──────────────────────────────
PRESET_EXAMPLES = [
    {
        "label": "🟢 Answerable: Melbourne Airport",
        "question": "What is another name for Melborne Airport?",
        "type": "Answerable",
        "description": "Direct factual question with clear evidence in the indexed test split.",
    },
    {
        "label": "🟢 Answerable: Historic Partnership",
        "question": "When did the partnership between Jones and Appleton end?",
        "type": "Answerable",
        "description": "Temporal extraction with clear text support in the passage.",
    },
    {
        "label": "🟢 Answerable: DATANET 1 Network",
        "question": "What was DATANET 1",
        "type": "Answerable",
        "description": "Entity definition query with high lexical overlap.",
    },
    {
        "label": "🔴 Unanswerable (Adversarial): Capital Punishment",
        "question": "How many execution methods are currently outlawed in the United States?",
        "type": "Unanswerable",
        "description": "Adversarial SQuAD 2.0 query with high BM25 match, but the text lacks the required fact.",
    },
    {
        "label": "🔴 Unanswerable (Adversarial): Comprehensive School",
        "question": "What two kinds of schools were developed in response to the costs of post-18 comprehensive school education?",
        "type": "Unanswerable",
        "description": "Passage discusses comprehensive schools, but not post-18 cost responses.",
    },
    {
        "label": "🔴 Out-of-Corpus / Zero-Evidence",
        "question": "Who was the supreme commander of the Martian fleet in the year 2341?",
        "type": "Unanswerable",
        "description": "Nonsense query completely outside the corpus to test zero-evidence handling.",
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
        # Load tuned threshold if available
        threshold = st.slider(
            "Confidence Threshold (tau):",
            min_value=0.05,
            max_value=0.95,
            value=0.42,
            step=0.01,
            help="Tuned via grid search on validation data to maximize F1 (optimal tau = 0.42).",
        )
        st.caption("💡 **Validation-tuned threshold:** `tau = 0.42` balances coverage and precision.")
    elif selected_strategy == "bm25_threshold":
        threshold = st.slider(
            "BM25 Score Cutoff:",
            min_value=0.0,
            max_value=50.0,
            value=13.15,
            step=0.25,
            help="Abstain if top BM25 retrieval score is below this cutoff (validation-tuned tau = 13.15).",
        )
    else:
        threshold = 0.0
        st.info("The Always Answer baseline operates at 100% coverage with zero gating.")

    # Retrieval Top-k
    st.subheader("3. Retrieval Top-k")
    top_k = st.slider("Passages to retrieve (k):", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 🎓 Academic Context")
    st.markdown(
        """
        - **Project:** RAG Selective Abstention  
        - **Dataset:** SQuAD 2.0 (500 Balanced Examples)  
        - **Splits:** 300 Train / 100 Val / 100 Test  
        - **Models:** BM25Okapi + RoBERTa-SQuAD2 + LogReg (10 Features)  
        - **Institution:** ENSA Tétouan (UAE)  
        - **Authors:** Youssef Ech-Chahlaoui & Zouhair Fezzaz  
        - **Supervisor:** Prof. Anas Belcaid (2026–2027)  
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


# ── Query Input Section (Form to prevent accidental re-runs) ─────────────────
col_query, col_preset = st.columns([1.6, 1.4], gap="large")

with col_preset:
    st.markdown("##### 📌 Benchmark Test Examples")
    preset_names = [ex["label"] for ex in PRESET_EXAMPLES]
    selected_preset_label = st.selectbox(
        "Load a preset test query:",
        options=preset_names,
        index=0,
    )
    current_preset = next(p for p in PRESET_EXAMPLES if p["label"] == selected_preset_label)
    st.caption(f"**Type:** `{current_preset['type']}` — *{current_preset['description']}*")

with col_query:
    st.markdown("##### 🔍 Query Input")
    default_question = current_preset["question"]

    # Use st.form to guarantee inference runs ONLY when the user clicks Submit
    with st.form("rag_query_form"):
        query_text = st.text_area(
            "Enter question:",
            value=default_question,
            height=90,
            placeholder="Type a question to evaluate...",
        )
        submit_button = st.form_submit_button("🚀 Analyze & Query Pipeline", type="primary", use_container_width=True)


# ── Inference Execution ───────────────────────────────────────────────────────
if submit_button:
    if not query_text.strip():
        st.warning("Please enter a question before running inference.")
    else:
        with st.spinner("Executing BM25 retrieval, RoBERTa reading, and 10-feature evaluation..."):
            start_time = time.time()
            result = pipeline.predict(
                question=query_text.strip(),
                top_k=top_k,
                strategy=selected_strategy,
                threshold=threshold,
            )
            elapsed_sec = time.time() - start_time

        st.markdown("---")

        # ── Decision Banner ───────────────────────────────────────────────────
        if result.status == "ANSWERED":
            st.success(
                f"### ✅ Pipeline Decision: ANSWER ACCEPTED\n\n"
                f"**Extracted Answer:** `{result.answer}`\n\n"
                f"*The model judged the evidence sufficient with decision confidence **{result.decision_probability:.4f}** (Threshold: {result.threshold:.2f}).*",
                icon="✅",
            )
        else:
            # Differentiate gating rejection vs. empty span from reader
            if selected_strategy == "always_answer":
                reason_text = (
                    f"**Strategy Note:** You selected **Always Answer (Baseline)**, which does not gate or withhold queries "
                    f"(Threshold: `{result.threshold:.2f}`). However, the RoBERTa reader found **no answer span** "
                    f"in the top-{top_k} retrieved Wikipedia passages (it assigned high confidence to the empty/impossible span). "
                    f"Because the text does not contain the answer, no answer text could be returned."
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
                        f"The answer was withheld to prevent an overconfident hallucination."
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

        # ── Metric Cards ──────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Pipeline Status", result.status)
        with m2:
            st.metric("P(Is Answerable)", f"{result.decision_probability:.3f}")
        with m3:
            st.metric("Decision Cutoff (tau)", f"{result.threshold:.2f}")
        with m4:
            st.metric("Top BM25 Score", f"{result.top_bm25_score:.2f}")
        with m5:
            st.metric("Latency", f"{elapsed_sec:.2f}s")

        # ── Detailed Tabs ─────────────────────────────────────────────────────
        tab_features, tab_passages, tab_benchmark = st.tabs(
            ["📊 10 Engineered Features", "📚 Retrieved Passages (Top-k)", "📈 Empirical Test Benchmark"]
        )

        with tab_features:
            st.markdown("#### 10-Dimensional Operational Feature Vector")
            st.markdown(
                "Features are strictly inference-available (derived solely from query, BM25 retrieval, and RoBERTa confidence)."
            )

            feat_dict = result.features
            feat_df = pd.DataFrame(
                [
                    {
                        "Feature Name": k,
                        "Value": round(v, 4) if pd.notna(v) else "N/A",
                        "Category": "Retrieval (BM25)" if "bm25" in k else "Reader / Query",
                        "Description": (
                            "Highest BM25 score among top-k" if k == "top_bm25_score" else
                            "Second highest BM25 score" if k == "second_bm25_score" else
                            "Score margin (top - second)" if k == "bm25_score_gap" else
                            "Mean BM25 score across top-k" if k == "mean_topk_bm25" else
                            "RoBERTa confidence on best span" if k == "qa_score" else
                            "RoBERTa probability of impossible answer" if k == "qa_no_answer_probability" else
                            "Word count of predicted span" if k == "predicted_answer_length" else
                            "Indicator whether predicted span is non-empty" if k == "answer_nonempty" else
                            "Word count of question" if k == "question_length" else
                            "Unique alphanumeric token count in question"
                        ),
                    }
                    for k, v in feat_dict.items()
                ]
            )

            col_table, col_chart = st.columns([1.3, 1.7], gap="medium")
            with col_table:
                st.dataframe(feat_df[["Feature Name", "Value", "Category"]], use_container_width=True, hide_index=True)

            with col_chart:
                st.markdown("##### Model Probability Signals")
                prob_chart_data = pd.DataFrame({
                    "Signal": ["RoBERTa QA Score", "RoBERTa No-Answer Prob", "LogReg P(Answerable)"],
                    "Probability": [
                        feat_dict.get("qa_score", 0.0),
                        feat_dict.get("qa_no_answer_probability", 0.0),
                        result.decision_probability,
                    ],
                }).set_index("Signal")
                st.bar_chart(prob_chart_data)

        with tab_passages:
            st.markdown(f"#### Top-{len(result.retrieved_passages)} Retrieved Contexts")
            if not result.retrieved_passages:
                st.info("No passages were retrieved.")
            else:
                for p in result.retrieved_passages:
                    is_best = (
                        result.best_passage is not None
                        and p["paragraph_id"] == result.best_passage["paragraph_id"]
                    )
                    border_color = "#10B981" if is_best else "#3B82F6"
                    best_tag = " ⭐ (Selected by RoBERTa Reader)" if is_best else ""

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
            st.markdown("#### Test Set Comparative Evaluation (Dynamically Loaded)")
            st.markdown(
                "Results loaded from `results/strategy_comparison.csv` on the strictly held-out test split (100 examples: 50 answerable, 50 unanswerable):"
            )

            sc_path = PROJECT_ROOT / "results" / "strategy_comparison.csv"
            if sc_path.exists():
                actual_sc = pd.read_csv(sc_path)
                display_cols = [c for c in ["strategy", "precision", "recall", "f1", "accuracy", "coverage", "false_answer_rate", "false_abstention_rate", "exact_match", "token_f1"] if c in actual_sc.columns]
                st.dataframe(actual_sc[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("Evaluation results file `results/strategy_comparison.csv` not yet generated. Run `scripts/run_experiments.py`.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ENSA Tétouan — Université Abdelmalek Essaâdi | Academic Research Project 2026–2027 | Supervised by Prof. Anas Belcaid</small></center>",
    unsafe_allow_html=True,
)
