"""
scripts/run_experiments.py
End-to-End Experiment Runner for RAG Abstention Study.

Implements:
1. Leak-Free Article-Level Splitting (Max 500 examples: 300 train / 100 val / 100 test, seed 42)
2. BM25Okapi Indexing (top-k=5) & Diagnostic Retrieval Feature Extraction
3. Extractive QA Inference (deepset/roberta-base-squad2)
4. 10 Inference-Available Predictive Features (source_passage_retrieved strictly excluded from X)
5. Model Training (LogisticRegression, class_weight='balanced')
6. Validation-Driven Threshold Selection (maximizing validation F1)
7. Final Held-Out Test Evaluation (Answerability metrics + Exact Match + Token F1)
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import joblib
from rank_bm25 import BM25Okapi
from transformers import pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

for d in [SPLITS_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_COUNTS = {
    "train": (150, 150),       # 300 total (50% answerable, 50% unanswerable)
    "validation": (50, 50),    # 100 total
    "test": (50, 50),          # 100 total
}

# The 10 inference-available predictive features (strictly excluding source_passage_retrieved)
PREDICTIVE_FEATURES = [
    "top_bm25_score",
    "second_bm25_score",
    "bm25_score_gap",
    "mean_topk_bm25",
    "qa_score",
    "qa_no_answer_probability",
    "predicted_answer_length",
    "answer_nonempty",
    "question_length",
    "question_token_count",
]

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TOKEN_SET_RE = re.compile(r"[A-Za-z0-9]+")
ARTICLES = {"a", "an", "the"}
PUNCT = re.compile(r"[^\w\s]")


def tokenize(text: str):
    return TOKEN_PATTERN.findall(str(text).lower())


def token_set(text: str):
    return set(TOKEN_SET_RE.findall(str(text).lower()))


def normalize_answer(s: str) -> str:
    """Standard SQuAD answer normalization."""
    s = str(s).lower()
    s = PUNCT.sub(" ", s)
    tokens = [t for t in s.split() if t not in ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2.0 * precision * recall / (precision + recall)


def best_reference_score(pred: str, answers: list, metric_fn) -> float:
    refs = [a.get("text", "") for a in answers if isinstance(a, dict) and "text" in a]
    if not refs:
        return 0.0
    return max(metric_fn(pred, ref) for ref in refs)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Article-Level Group Splitting (Max 500 Examples)
# ══════════════════════════════════════════════════════════════════════════════
def step1_split_dataset():
    print("\n" + "=" * 70)
    print("STEP 1: Article-Level Group Splitting (Target: 500 examples)")
    print("=" * 70)

    clean_path = PROCESSED_DIR / "questions_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}")

    df = pd.read_csv(clean_path)
    df["answers"] = df["answers"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df["is_answerable"] = df["is_answerable"].astype(bool)

    # 1. Hold out ~20% of articles for test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev_idx, test_idx = next(gss_test.split(df, groups=df["article_title"]))
    dev_pool = df.iloc[dev_idx].copy()
    test_pool = df.iloc[test_idx].copy()

    # 2. Hold out ~25% of development pool for validation
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_idx, val_idx = next(gss_val.split(dev_pool, groups=dev_pool["article_title"]))
    train_pool = dev_pool.iloc[train_idx].copy()
    val_pool = dev_pool.iloc[val_idx].copy()

    pools = {"train": train_pool, "validation": val_pool, "test": test_pool}

    def balanced_sample(pool, n_pos, n_neg, seed):
        pos = pool[pool.is_answerable == True]
        neg = pool[pool.is_answerable == False]
        n_p = min(n_pos, len(pos))
        n_n = min(n_neg, len(neg))
        sampled = pd.concat([
            pos.sample(n=n_p, random_state=seed),
            neg.sample(n=n_n, random_state=seed + 1),
        ], ignore_index=True)
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    splits = {}
    for i, (name, (n_pos, n_neg)) in enumerate(TARGET_COUNTS.items()):
        splits[name] = balanced_sample(pools[name], n_pos, n_neg, SEED + i)
        print(f"  {name:10s} -> Total: {len(splits[name]):3d} (Ans: {(splits[name].is_answerable==True).sum():3d}, "
              f"Unans: {(splits[name].is_answerable==False).sum():3d}) | Articles: {splits[name].article_title.nunique():3d}")

    # Verify zero article leakage
    train_arts = set(splits["train"].article_title)
    val_arts = set(splits["validation"].article_title)
    test_arts = set(splits["test"].article_title)

    assert len(train_arts & val_arts) == 0, "Leakage between train and validation!"
    assert len(train_arts & test_arts) == 0, "Leakage between train and test!"
    assert len(val_arts & test_arts) == 0, "Leakage between validation and test!"
    total_samples = sum(len(s) for s in splits.values())
    assert total_samples <= 500, f"Total samples exceed 500 ({total_samples})"

    # Save to disk
    metadata = {
        "seed": SEED,
        "total_examples": total_samples,
        "policy": "GroupShuffleSplit by article_title with balanced sampling per split",
        "splits": {
            k: {
                "total": len(v),
                "answerable": int((v.is_answerable == True).sum()),
                "unanswerable": int((v.is_answerable == False).sum()),
                "unique_articles": int(v.article_title.nunique()),
            }
            for k, v in splits.items()
        },
    }
    with (SPLITS_DIR / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    for name, part in splits.items():
        export = part.copy()
        export["answers"] = export["answers"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        export.to_csv(SPLITS_DIR / f"{name}.csv", index=False, encoding="utf-8")
        with (SPLITS_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in part.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓ Split complete. Verified disjoint articles. Saved to {SPLITS_DIR}.")
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: BM25 Retrieval & Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════
def step2_bm25_retrieval(splits, top_k=5):
    print("\n" + "=" * 70)
    print("STEP 2: BM25 Indexing and Top-k Retrieval")
    print("=" * 70)

    retrieval_results = {}

    for name, df in splits.items():
        print(f"Building BM25 index for {name} split...")
        corpus = (
            df[["paragraph_id", "article_title", "context"]]
            .drop_duplicates("paragraph_id")
            .reset_index(drop=True)
        )
        corpus["tokens"] = corpus["context"].map(tokenize)
        bm25 = BM25Okapi(corpus["tokens"].tolist())

        rows = []
        for row in df.itertuples(index=False):
            q_tokens = tokenize(row.question)
            if q_tokens:
                scores = bm25.get_scores(q_tokens)
                order = np.argsort(scores)[::-1][:top_k]
                hits = corpus.iloc[order].copy()
                scores_k = scores[order]
            else:
                hits = corpus.head(top_k).copy()
                scores_k = np.zeros(len(hits))

            top_score = float(scores_k[0]) if len(scores_k) > 0 else 0.0
            second_score = float(scores_k[1]) if len(scores_k) > 1 else 0.0
            gap = top_score - second_score
            mean_score = float(scores_k.mean()) if len(scores_k) > 0 else 0.0
            retrieved_ids = list(hits["paragraph_id"])
            source_retrieved = int(row.paragraph_id in set(retrieved_ids))

            rows.append({
                "question_id": row.question_id,
                "top_bm25_score": top_score,
                "second_bm25_score": second_score,
                "bm25_score_gap": gap,
                "mean_topk_bm25": mean_score,
                "source_passage_retrieved": source_retrieved,  # Diagnostic only
                "retrieved_passage_ids": json.dumps(retrieved_ids),
            })

        ret_df = pd.DataFrame(rows)
        ret_df.to_csv(RESULTS_DIR / f"{name}_retrieval_features.csv", index=False)
        retrieval_results[name] = ret_df
        print(f"✓ Saved {name}_retrieval_features.csv ({len(ret_df)} rows).")

    return retrieval_results


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Extractive QA with RoBERTa
# ══════════════════════════════════════════════════════════════════════════════
def step3_qa_inference(splits, top_k=5):
    print("\n" + "=" * 70)
    print("STEP 3: RoBERTa QA Inference (deepset/roberta-base-squad2)")
    print("=" * 70)

    print("Loading HuggingFace question-answering pipeline...")
    qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

    qa_results = {}
    for name, df in splits.items():
        print(f"Running QA inference on {name} split ({len(df)} questions)...")
        ret_df = pd.read_csv(RESULTS_DIR / f"{name}_retrieval_features.csv").set_index("question_id")
        corpus = (
            df[["paragraph_id", "article_title", "context"]]
            .drop_duplicates("paragraph_id")
            .set_index("paragraph_id")
        )

        outputs = []
        t0 = time.time()
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            pids = json.loads(ret_df.loc[row.question_id, "retrieved_passage_ids"])[:top_k]
            candidates = []
            for pid in pids:
                if pid not in corpus.index:
                    continue
                ctx = corpus.loc[pid, "context"]
                try:
                    pred = qa_pipe(question=row.question, context=ctx, handle_impossible_answer=True)
                    candidates.append({
                        "passage_id": pid,
                        "answer": str(pred.get("answer", "")).strip(),
                        "qa_score": float(pred.get("score", 0.0)),
                        "qa_no_answer_probability": float(pred.get("no_answer_probability", 0.0)),
                    })
                except Exception as err:
                    candidates.append({
                        "passage_id": pid,
                        "answer": "",
                        "qa_score": 0.0,
                        "qa_no_answer_probability": 1.0,
                    })

            if candidates:
                best = max(candidates, key=lambda c: c["qa_score"])
            else:
                best = {"passage_id": None, "answer": "", "qa_score": 0.0, "qa_no_answer_probability": 1.0}

            outputs.append({
                "question_id": row.question_id,
                "predicted_answer": best["answer"],
                "qa_score": best["qa_score"],
                "qa_no_answer_probability": best["qa_no_answer_probability"],
                "selected_passage_id": best["passage_id"],
            })

            if idx % 50 == 0 or idx == len(df):
                print(f"  [{name}] {idx}/{len(df)} questions processed ({time.time() - t0:.1f}s)")

        qa_df = pd.DataFrame(outputs)
        qa_df.to_csv(RESULTS_DIR / f"{name}_qa_predictions.csv", index=False)
        qa_results[name] = qa_df
        print(f"✓ Saved {name}_qa_predictions.csv.")

    return qa_results


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Feature Assembly & Classifier Training (10 Features)
# ══════════════════════════════════════════════════════════════════════════════
def step4_feature_assembly_and_training(splits):
    print("\n" + "=" * 70)
    print("STEP 4: Assemble 10 Features and Train Classifier")
    print("=" * 70)

    data = {}
    for name in ["train", "validation", "test"]:
        q_df = splits[name].copy()
        r_df = pd.read_csv(RESULTS_DIR / f"{name}_retrieval_features.csv")
        a_df = pd.read_csv(RESULTS_DIR / f"{name}_qa_predictions.csv")

        overlap_r = [c for c in r_df.columns if c in q_df.columns and c != "question_id"]
        overlap_a = [c for c in a_df.columns if c in q_df.columns and c != "question_id"]
        r_df = r_df.drop(columns=overlap_r)
        a_df = a_df.drop(columns=overlap_a)

        merged = q_df.merge(r_df, on="question_id").merge(a_df, on="question_id")

        # Compute query & answer length features
        merged["question_length"] = merged["question"].fillna("").str.split().str.len().astype(float)
        merged["predicted_answer_length"] = merged["predicted_answer"].fillna("").str.split().str.len().astype(float)
        merged["question_token_count"] = merged["question"].map(lambda q: float(len(token_set(q))))
        merged["answer_nonempty"] = (merged["predicted_answer"].fillna("").str.strip() != "").astype(float)

        data[name] = merged

    # Train Split
    X_train = data["train"][PREDICTIVE_FEATURES]
    y_train = data["train"]["is_answerable"].astype(int)

    print("Training Logistic Regression pipeline on", len(X_train), "examples...")
    print("Predictive features (10):", PREDICTIVE_FEATURES)

    classifier_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])
    classifier_pipeline.fit(X_train, y_train)

    joblib.dump(classifier_pipeline, MODELS_DIR / "answerability_logreg.joblib")
    print(f"✓ Model trained and saved to {MODELS_DIR / 'answerability_logreg.joblib'}.")

    # Validation: Tune Decision Threshold
    X_val = data["validation"][PREDICTIVE_FEATURES]
    y_val = data["validation"]["is_answerable"].astype(int)
    val_probs = classifier_pipeline.predict_proba(X_val)[:, 1]

    threshold_grid = np.linspace(0.05, 0.95, 91)
    t_rows = []
    for t in threshold_grid:
        pred = (val_probs >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_val, pred, average="binary", zero_division=0)
        acc = accuracy_score(y_val, pred)
        t_rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": f1, "accuracy": acc})

    val_thresholds_df = pd.DataFrame(t_rows)
    best_row = val_thresholds_df.loc[val_thresholds_df["f1"].idxmax()]
    best_classifier_thresh = float(best_row["threshold"])
    val_thresholds_df.to_csv(RESULTS_DIR / "classifier_validation_thresholds.csv", index=False)
    print(f"✓ Best Validation Classifier Threshold: tau = {best_classifier_thresh:.2f} (F1: {best_row['f1']:.4f})")

    # Tune BM25 threshold on validation
    bm25_val_scores = data["validation"]["top_bm25_score"].to_numpy()
    bm25_cands = np.unique(np.quantile(bm25_val_scores, np.linspace(0.0, 1.0, 101)))
    b_rows = []
    for bt in bm25_cands:
        b_pred = (bm25_val_scores >= bt).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_val, b_pred, average="binary", zero_division=0)
        b_rows.append({"threshold": float(bt), "precision": p, "recall": r, "f1": f1})
    bm25_th_df = pd.DataFrame(b_rows).sort_values("f1", ascending=False)
    best_bm25_thresh = float(bm25_th_df.iloc[0]["threshold"])
    print(f"✓ Best Validation BM25 Threshold: tau = {best_bm25_thresh:.2f}")

    # Save validation decisions
    data["validation"]["classifier_prob"] = val_probs
    data["validation"]["classifier_answer"] = (val_probs >= best_classifier_thresh)
    data["validation"]["bm25_threshold_answer"] = (data["validation"]["top_bm25_score"] >= best_bm25_thresh)
    data["validation"][["question_id", "always_answer", "bm25_threshold_answer"]] = pd.DataFrame({
        "question_id": data["validation"]["question_id"],
        "always_answer": True,
        "bm25_threshold_answer": data["validation"]["bm25_threshold_answer"],
    })
    data["validation"][["question_id", "always_answer", "bm25_threshold_answer"]].to_csv(
        RESULTS_DIR / "validation_threshold_decisions.csv", index=False
    )

    return data, classifier_pipeline, best_classifier_thresh, best_bm25_thresh


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Held-Out Test Evaluation (All 3 Strategies)
# ══════════════════════════════════════════════════════════════════════════════
def step5_evaluate_test(data, classifier, clf_threshold, bm25_threshold):
    print("\n" + "=" * 70)
    print("STEP 5: Held-Out Test Set Evaluation")
    print("=" * 70)

    test = data["test"].copy()
    X_test = test[PREDICTIVE_FEATURES]
    y_test_bool = test["is_answerable"].astype(bool).to_numpy()
    y_test_int = y_test_bool.astype(int)

    # 1. Predictions
    test_probs = classifier.predict_proba(X_test)[:, 1]
    clf_decisions = (test_probs >= clf_threshold)
    bm25_decisions = (test["top_bm25_score"].to_numpy() >= bm25_threshold)
    always_decisions = np.ones(len(test), dtype=bool)

    # Save intermediate decisions
    pd.DataFrame({
        "question_id": test["question_id"],
        "classifier_prob": test_probs,
        "classifier_answer": clf_decisions,
    }).to_csv(RESULTS_DIR / "test_classifier_predictions.csv", index=False)

    pd.DataFrame({
        "question_id": test["question_id"],
        "always_answer": always_decisions,
        "bm25_threshold_answer": bm25_decisions,
    }).to_csv(RESULTS_DIR / "test_threshold_decisions.csv", index=False)

    strategies = {
        "Always answer": always_decisions,
        "BM25 threshold": bm25_decisions,
        "Answerability classifier": clf_decisions,
    }

    # Answer Quality Precomputation (Exact Match and Token F1 on Answerable)
    test["exact_match"] = test.apply(
        lambda r: best_reference_score(r["predicted_answer"], r["answers"], exact_match) if r["is_answerable"] else 0.0,
        axis=1,
    )
    test["token_f1"] = test.apply(
        lambda r: best_reference_score(r["predicted_answer"], r["answers"], token_f1) if r["is_answerable"] else 0.0,
        axis=1,
    )

    comparison_rows = []
    quality_rows = []

    for strat_name, dec in strategies.items():
        # Answerability Classification Metrics
        p, r, f1, _ = precision_recall_fscore_support(y_test_bool, dec, average="binary", zero_division=0)
        acc = accuracy_score(y_test_bool, dec)
        cov = float(dec.mean())

        unans = ~y_test_bool
        ans = y_test_bool

        far = float(dec[unans].mean()) if unans.any() else 0.0   # False-Answer Rate (answered on unanswerable)
        fabr = float((~dec[ans]).mean()) if ans.any() else 0.0  # False-Abstention Rate (abstained on answerable)

        # CM: tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(y_test_bool, dec, labels=[False, True]).ravel()

        # Strict Answer Correctness Metrics
        # Overall EM across all 100 questions (correct abstention on unanswerable = 1.0; wrong answer = 0.0)
        overall_em_list = []
        overall_f1_list = []
        for is_ans, accepted, em_val, f1_val in zip(y_test_bool, dec, test["exact_match"], test["token_f1"]):
            if not is_ans:
                overall_em_list.append(1.0 if not accepted else 0.0)
                overall_f1_list.append(1.0 if not accepted else 0.0)
            else:
                overall_em_list.append(em_val if accepted else 0.0)
                overall_f1_list.append(f1_val if accepted else 0.0)

        overall_em = float(np.mean(overall_em_list))
        overall_token_f1 = float(np.mean(overall_f1_list))

        # Selective Answer Quality on Accepted Subset only:
        accepted_mask = dec
        if accepted_mask.sum() > 0:
            accepted_em = float(test.loc[accepted_mask & ans, "exact_match"].sum() / accepted_mask.sum())
            accepted_token_f1 = float(test.loc[accepted_mask & ans, "token_f1"].sum() / accepted_mask.sum())
        else:
            accepted_em = 0.0
            accepted_token_f1 = 0.0

        comparison_rows.append({
            "strategy": strat_name,
            "precision": p,
            "recall": r,
            "f1": f1,
            "accuracy": acc,
            "coverage": cov,
            "false_answer_rate": far,
            "false_abstention_rate": fabr,
            "exact_match": overall_em,
            "token_f1": overall_token_f1,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

        quality_rows.append({
            "strategy": strat_name,
            "coverage": cov,
            "accepted_exact_match": accepted_em,
            "accepted_token_f1": accepted_token_f1,
            "overall_exact_match": overall_em,
            "overall_token_f1": overall_token_f1,
        })

    comp_df = pd.DataFrame(comparison_rows)
    qual_df = pd.DataFrame(quality_rows)

    comp_df.to_csv(RESULTS_DIR / "strategy_comparison.csv", index=False)
    qual_df.to_csv(RESULTS_DIR / "answer_quality_by_strategy.csv", index=False)

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST RESULTS (N = 100):")
    print("=" * 70)
    print(comp_df[["strategy", "precision", "recall", "f1", "accuracy", "coverage", "false_answer_rate", "false_abstention_rate", "exact_match", "token_f1"]].to_string(index=False))
    print("=" * 70)

    # Save detailed evaluation rows for manual error review
    test["classifier_answer"] = clf_decisions
    test["bm25_threshold_answer"] = bm25_decisions
    test["classifier_prob"] = test_probs

    def assign_outcome(row):
        if row.is_answerable and row.classifier_answer:
            return "Correct answer decision"
        if (not row.is_answerable) and (not row.classifier_answer):
            return "Correct abstention"
        if (not row.is_answerable) and row.classifier_answer:
            return "False answer"
        return "Unnecessary refusal"

    test["classifier_outcome"] = test.apply(assign_outcome, axis=1)
    test.to_csv(RESULTS_DIR / "test_evaluation_rows.csv", index=False)
    test.to_csv(RESULTS_DIR / "manual_error_review.csv", index=False)
    print("✓ Saved results/strategy_comparison.csv and results/test_evaluation_rows.csv.")

    return comp_df, qual_df


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()
    splits = step1_split_dataset()
    retrieval_res = step2_bm25_retrieval(splits, top_k=5)
    qa_res = step3_qa_inference(splits, top_k=5)
    data, clf, clf_th, bm25_th = step4_feature_assembly_and_training(splits)
    comp_df, qual_df = step5_evaluate_test(data, clf, clf_th, bm25_th)
    print(f"\nAll experimental steps completed successfully in {time.time() - t_start:.1f}s.")
