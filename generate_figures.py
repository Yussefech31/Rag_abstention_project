"""
generate_figures.py
Generates all scientific figures for the RAG Abstention academic report.
Run from the project root: python generate_figures.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = r"c:\Users\WIN 11 25H2\Desktop\nlp\Rag_ab"
RES    = os.path.join(ROOT, "results")
SPLITS = os.path.join(ROOT, "data", "splits")
FIG    = os.path.join(RES,  "figures")
os.makedirs(FIG, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

PALETTE      = ["#2563EB", "#DC2626"]   # blue = answerable, red = unanswerable
STRAT_COLORS = ["#6B7280", "#2563EB", "#059669"]  # grey, blue, green
FIG_SIZE     = (10, 6)
DPI          = 300

created  = []
skipped  = []


def savefig(name, fig=None):
    path = os.path.join(FIG, name)
    (fig or plt).savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    created.append(name)
    print(f"  ✓ saved {name}")


def skip(name, reason):
    skipped.append((name, reason))
    print(f"  ✗ skipped {name}: {reason}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Dataset Distribution by Split and Answerability
# ══════════════════════════════════════════════════════════════════════════════
print("\n[fig2] Dataset distribution …")
try:
    dfs = {}
    for split in ("train", "validation", "test"):
        path = os.path.join(SPLITS, f"{split}.csv")
        dfs[split] = pd.read_csv(path, usecols=["is_answerable"])

    split_names = ["Train", "Validation", "Test"]
    answerable   = [dfs[s]["is_answerable"].sum() for s in ("train","validation","test")]
    unanswerable = [len(dfs[s]) - a for s, a in zip(("train","validation","test"), answerable)]

    x   = np.arange(len(split_names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    b1 = ax.bar(x - w/2, answerable,   w, label="Answerable",   color=PALETTE[0], alpha=0.85)
    b2 = ax.bar(x + w/2, unanswerable, w, label="Unanswerable", color=PALETTE[1], alpha=0.85)

    # value labels
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.annotate(f"{int(h):,}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(split_names, fontsize=12)
    ax.set_xlabel("Data Split", fontsize=13)
    ax.set_ylabel("Number of Questions", fontsize=13)
    ax.set_title("Figure 2: Dataset Distribution by Split and Answerability", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{int(v):,}"))
    plt.tight_layout()
    savefig("fig2_dataset_distribution.png")
except Exception as e:
    skip("fig2_dataset_distribution.png", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — BM25 Score Distribution by Answerability (validation, KDE overlay)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[fig3] BM25 score distribution by answerability …")
try:
    vrf = pd.read_csv(os.path.join(RES, "validation_retrieval_features.csv"),
                      usecols=["is_answerable", "top_bm25_score"])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for flag, label, color in [
        (True,  "Answerable",   PALETTE[0]),
        (False, "Unanswerable", PALETTE[1]),
    ]:
        subset = vrf[vrf["is_answerable"] == flag]["top_bm25_score"]
        sns.kdeplot(subset, ax=ax, fill=True, alpha=0.4, color=color, label=label, linewidth=2)
        ax.axvline(subset.median(), color=color, linestyle="--", linewidth=1.5,
                   label=f"{label} median ({subset.median():.1f})")

    ax.set_xlabel("Top BM25 Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Figure 3: BM25 Score Distribution by Answerability", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig("fig3_bm25_score_distribution.png")
except Exception as e:
    skip("fig3_bm25_score_distribution.png", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — BM25 Score Distribution Across Data Splits (box plots)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[fig4] BM25 scores per split …")
try:
    frames = []
    for split in ("train", "validation", "test"):
        df = pd.read_csv(os.path.join(RES, f"{split}_retrieval_features.csv"),
                         usecols=["top_bm25_score"])
        df["Split"] = split.capitalize()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["Split"] = pd.Categorical(combined["Split"],
                                       categories=["Train", "Validation", "Test"], ordered=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=combined, x="Split", y="top_bm25_score", ax=ax,
                palette=["#3B82F6", "#8B5CF6", "#10B981"],
                width=0.5, flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_xlabel("Data Split", fontsize=13)
    ax.set_ylabel("Top BM25 Score", fontsize=13)
    ax.set_title("Figure 4: BM25 Score Distribution Across Data Splits", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("fig4_bm25_scores_per_split.png")
except Exception as e:
    skip("fig4_bm25_scores_per_split.png", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Threshold Selection on Validation Data
# ══════════════════════════════════════════════════════════════════════════════
print("\n[fig5] Threshold selection …")
try:
    thdf = pd.read_csv(os.path.join(RES, "classifier_validation_thresholds.csv"))

    # Best threshold = max F1
    best_idx = thdf["f1"].idxmax()
    best_thr = thdf.loc[best_idx, "threshold"]
    best_f1  = thdf.loc[best_idx, "f1"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(thdf["threshold"], thdf["precision"], color="#2563EB", linewidth=2, marker="o",
            markersize=3, label="Precision")
    ax.plot(thdf["threshold"], thdf["recall"],    color="#DC2626", linewidth=2, marker="s",
            markersize=3, label="Recall")
    ax.plot(thdf["threshold"], thdf["f1"],        color="#059669", linewidth=2.5, marker="^",
            markersize=4, label="F1-score")

    ax.axvline(best_thr, color="#6B7280", linestyle="--", linewidth=1.8,
               label=f"Selected threshold = {best_thr:.2f}\n(F1 = {best_f1:.3f})")
    ax.scatter([best_thr], [best_f1], color="#059669", s=100, zorder=5)

    ax.set_xlabel("Answerability Probability Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 5: Threshold Selection on Validation Data", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    savefig("fig5_threshold_selection.png")
except Exception as e:
    skip("fig5_threshold_selection.png", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Strategy Comparison on Test Set
# ══════════════════════════════════════════════════════════════════════════════
print("\n[fig6] Strategy comparison …")
try:
    sc_path = os.path.join(RES, "strategy_comparison.csv")
    if not os.path.exists(sc_path):
        raise FileNotFoundError("strategy_comparison.csv not found")

    sc = pd.read_csv(sc_path)
    strategies = sc["strategy"].tolist()
    metrics    = ["precision", "recall", "f1", "coverage"]

    # Clean labels
    strat_labels = [s.replace("Answerability classifier", "Classifier") for s in strategies]

    n_strats  = len(strategies)
    n_metrics = len(metrics)
    x         = np.arange(n_strats)
    total_w   = 0.7
    w         = total_w / n_metrics

    metric_colors = ["#2563EB", "#DC2626", "#059669", "#D97706"]
    metric_labels = ["Precision", "Recall", "F1-score", "Coverage"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for i, (m, color, mlabel) in enumerate(zip(metrics, metric_colors, metric_labels)):
        offset = (i - n_metrics/2 + 0.5) * w
        vals   = sc[m].values
        bars   = ax.bar(x + offset, vals, w, label=mlabel, color=color, alpha=0.85)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Strategy", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Figure 6: Strategy Comparison on Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    savefig("fig6_strategy_comparison.png")
except Exception as e:
    skip("fig6_strategy_comparison.png", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Classifier Confusion Matrix (Test Set)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[fig8] Confusion matrix …")
try:
    cp = pd.read_csv(os.path.join(RES, "test_classifier_predictions.csv"))
    te = pd.read_csv(os.path.join(SPLITS, "test.csv"), usecols=["question_id", "is_answerable"])

    merged = cp.merge(te, on="question_id", how="inner")

    # Ground truth & prediction — both as bool
    y_true = merged["is_answerable"].astype(bool)
    y_pred = merged["classifier_answer"].astype(bool)

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    # rows: actual Answerable / Unanswerable
    # cols: predicted Answerable / Unanswerable

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = LinearSegmentedColormap.from_list("blue_white", ["#DBEAFE", "#1D4ED8"])
    im   = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Answerable", "Unanswerable"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i,j] > thresh else "black")

    ax.set_ylabel("True Label", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_title("Figure 8: Classifier Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("fig8_confusion_matrix.png")
except Exception as e:
    skip("fig8_confusion_matrix.png", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Error Analysis by Strategy
# Uses test_evaluation_rows.csv which has all per-question columns
# ══════════════════════════════════════════════════════════════════════════════
print("\n[fig9] Error analysis …")
try:
    ev = pd.read_csv(os.path.join(RES, "test_evaluation_rows.csv"))

    # Columns available:
    #  question_id, question, is_answerable, predicted_answer,
    #  answerability_probability, classifier_answer, bm25_threshold_answer,
    #  top_bm25_score, qa_score

    # For each strategy define what "attempt to answer" means:
    #  always_answer       → always True
    #  bm25_threshold      → bm25_threshold_answer
    #  classifier          → classifier_answer

    # An answer is "correct" if is_answerable==True and QA token-F1 > 0 (non-empty answer)
    # We use qa_score > 0 as proxy for getting any answer text.
    # Buckets per question×strategy:
    #  correct_answer     : attempted=True, is_answerable=True, qa_score>0
    #  false_answer       : attempted=True, is_answerable=False
    #  correct_abstention : attempted=False, is_answerable=False
    #  false_abstention   : attempted=False, is_answerable=True

    strategies_map = {
        "Always Answer":    ev["is_answerable"].apply(lambda _: True),  # always attempt
        "BM25 Threshold":   ev["bm25_threshold_answer"].astype(bool),
        "Classifier":       ev["classifier_answer"].astype(bool),
    }

    results = {}
    for name, attempt in strategies_map.items():
        is_ans  = ev["is_answerable"].astype(bool)
        correct_answer     = (attempt  &  is_ans).sum()
        false_answer       = (attempt  & ~is_ans).sum()
        correct_abstention = (~attempt & ~is_ans).sum()
        false_abstention   = (~attempt &  is_ans).sum()
        results[name] = {
            "Correct Answers":      correct_answer,
            "False Answers":        false_answer,
            "Correct Abstentions":  correct_abstention,
            "False Abstentions":    false_abstention,
        }

    df9 = pd.DataFrame(results).T
    categories  = ["Correct Answers", "Correct Abstentions", "False Answers", "False Abstentions"]
    cat_colors  = ["#059669", "#2563EB", "#DC2626", "#F59E0B"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bottoms = np.zeros(len(df9))
    x9 = np.arange(len(df9))

    for cat, color in zip(categories, cat_colors):
        vals = df9[cat].values
        bars = ax.bar(x9, vals, bottom=bottoms, label=cat, color=color, alpha=0.88, width=0.5)
        # label segments with counts
        for b, v, bot in zip(bars, vals, bottoms):
            if v > 5:  # only label visible segments
                ax.text(b.get_x() + b.get_width()/2, bot + v/2,
                        str(int(v)), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        bottoms = bottoms + vals

    ax.set_xticks(x9)
    ax.set_xticklabels(list(results.keys()), fontsize=12)
    ax.set_ylabel("Number of Questions", fontsize=13)
    ax.set_xlabel("Strategy", fontsize=13)
    ax.set_title("Figure 9: Error Analysis by Strategy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    savefig("fig9_error_analysis.png")
except Exception as e:
    skip("fig9_error_analysis.png", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"Created ({len(created)}):")
for f in created:
    print(f"  • {f}")
if skipped:
    print(f"\nSkipped ({len(skipped)}):")
    for name, reason in skipped:
        print(f"  • {name}: {reason}")
print("="*60)
