# Knowing When Not to Answer: Abstention & Selective Prediction in Retrieval-Augmented QA

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗_Transformers-4.30%2B-FFD21E.svg)](https://huggingface.co/transformers/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Academic Report](https://img.shields.io/badge/Report-45--Page%20Academic%20PDF-blue.svg)](academic_report/main.pdf)

> **Academic Research Project**  
> **Institution:** École Nationale des Sciences Appliquées de Tétouan (ENSA Tétouan) — Université Abdelmalek Essaâdi  
> **Authors:** Youssef Ech-Chahlaoui & Zouhair Fezzaz  
> **Supervisor:** Prof. Anas Belcaid  
> **Academic Year:** 2026–2027  

---

## 📋 Table of Contents

- [Overview & Research Motivation](#-overview--research-motivation)
- [Key Empirical Results](#-key-empirical-results)
- [System Architecture](#-system-architecture)
- [Engineered Features](#-engineered-features)
- [Methodology & Experimental Rigor](#-methodology--experimental-rigor)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Execution & Reproducibility](#-execution--reproducibility)
- [Compiling the Academic Report](#-compiling-the-academic-report)
- [Scientific Figures & Artifacts](#-scientific-figures--artifacts)
- [Citation](#-citation)

---

## 🔬 Overview & Research Motivation

Modern Question Answering (QA) systems powered by large language models and neural readers (e.g., RoBERTa) achieve high fluency and performance. However, in **Retrieval-Augmented Generation (RAG)** pipelines, these systems suffer from a critical vulnerability: **overconfident hallucination**. When a retriever fetches irrelevant passages or when a user poses an inherently unanswerable question, standard readers nonetheless fabricate plausibly phrased but factually incorrect answers.

In high-stakes domains (healthcare, law, finance), providing a false answer is far more costly than an honest refusal. This project tackles the selective prediction challenge:

$$\text{Trade-off:} \quad \text{Coverage (answering queries)} \iff \text{Precision (guaranteeing factual veracity)}$$

### Core Research Question
> *How can a Retrieval-Augmented QA pipeline accurately detect unanswerable questions and abstain from responding, dramatically reducing the False-Answer Rate while retaining high answer quality and coverage on answerable questions?*

We formulate and evaluate three distinct operational strategies on SQuAD 2.0:
1. **Always Answer (Baseline):** The conventional RAG pipeline without abstention mechanism.
2. **BM25 Retrieval Thresholding:** Abstention triggered when the top retrieval similarity score falls below a tuned cutoff threshold $\tau_{\text{BM25}} = 13.15$.
3. **Multi-Feature Answerability Classifier:** A trained classification pipeline combining 10 inference-available retrieval and reader uncertainty signals to predict question answerability before output generation (tuned threshold $\tau = 0.42$).

---

## 📊 Key Empirical Results

All models and thresholds were tuned exclusively on the validation set ($\Delta \tau = 0.01$) and evaluated once on the strictly held-out test set ($N = 1{,}200$ questions; 600 answerable, 600 unanswerable):

| Strategy | Precision | Recall | F1 Score | Coverage | False-Answer Rate (FAR) | False-Abstention Rate (FAbR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always Answer (Baseline)** | 0.4820 | **1.0000** | 0.6505 | 1.0000 (100.0%) | 1.0000 (100.0%) | **0.0000 (0.0%)** |
| **BM25 Threshold ($\tau$)** | 0.4810 | 0.9959 | 0.6486 | 0.9980 (99.8%) | 1.0000 (100.0%) | 0.0042 (0.4%) |
| **Answerability Classifier** | **0.8231** | 0.8880 | **0.8543** | 0.5200 (52.0%) | **0.1776 (17.8%)** | 0.1120 (11.2%) |

*Metrics definition:*
- **Precision:** Correct answers divided by total answers provided.
- **FAR (False-Answer Rate):** Percentage of unanswerable questions incorrectly given an answer ($1.00 \to 0.18$).
- **FAbR (False-Abstention Rate):** Percentage of answerable questions incorrectly rejected ($0.11$).

### Key Insights
- **Retrieval Threshold Ineffectiveness:** BM25 retrieval scores alone are essentially incapable of distinguishing answerable from unanswerable queries (reducing FAR by 0% and coverage only to 99.8%). Unanswerable queries often share rich surface-level lexical overlap with corpus passages.
- **Power of Joint Evidence:** The multi-feature classifier achieves an **82.2% relative reduction in False-Answer Rate** (from 100% down to 17.8%) while maintaining an $F_1$ score of **85.4%** on answerable queries. Reader uncertainty combined with retrieval score dynamics provides the decisive discriminative signal.

---

## 🏗 System Architecture

The end-to-end pipeline decomposes into five tightly coupled stages:

```
                  ┌──────────────────────┐
                  │    User Question     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      BM25 Index      │
                  │ (Okapi k1=1.5, b=.75)│
                  └──────────┬───────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │   Top-k Passage Retrieval     │
             │            (k = 5)            │
             └───────┬───────────────┬───────┘
                     │               │
  [Retrieval Features]               │ Top-1 Passage
                     │               ▼
                     │   ┌───────────────────────┐
                     │   │     RoBERTa Reader    │
                     │   │ (roberta-base-squad2) │
                     │   └───────────┬───────────┘
                     │               │
                     │       [Reader Features]
                     ▼               ▼
             ┌───────────────────────────────┐
             │ 10-Feature Vector Extraction  │
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │   Answerability Classifier    │
             │   (StandardScaler + LogReg)   │
             └───────────────┬───────────────┘
                             │
               P(is_answerable) >= 0.42 ?
                             │
              ┌──────────────┴──────────────┐
              │                             │
          [YES]                            [NO]
              ▼                             ▼
  ┌───────────────────────┐     ┌───────────────────────┐
  │  Emit Extracted Span  │     │   Abstain Token (∅)   │
  │  (Predicted Answer)   │     │  "Cannot be answered" │
  └───────────────────────┘     └───────────────────────┘
```

---

## 🔍 Engineered Features

The classifier uses **10 inference-available features** extracted from the retrieval stage and neural reader:

| Feature Name | Category | Type | Description |
| :--- | :--- | :--- | :--- |
| `top_bm25_score` | Retrieval | Float | Highest BM25 score among retrieved passages: $\max_{p \in \mathcal{P}_k} \text{score}(q, p)$ |
| `second_bm25_score` | Retrieval | Float | Score of the second highest ranked candidate passage |
| `bm25_score_gap` | Retrieval | Float | Score margin: $\text{top\_bm25\_score} - \text{second\_bm25\_score}$ |
| `mean_topk_bm25` | Retrieval | Float | Average retrieval score across all $k=5$ passages |
| `qa_score` | Reader | Float | Confidence logit / softmax probability of the top extracted span |
| `qa_no_answer_probability` | Reader | Float | Probability assigned by RoBERTa to the empty / impossible span |
| `predicted_answer_length` | Reader | Integer | Whitespace word count of the extracted span |
| `answer_nonempty` | Reader | Binary | Boolean flag indicating whether the predicted answer string is non-empty |
| `question_length` | Query | Integer | Whitespace-delimited word count of the input question |
| `question_token_count` | Query | Integer | Count of unique alphanumeric tokens in the question |

> **Feature Parity & Leakage Prevention:** `source_passage_retrieved` (whether the original gold Wikipedia paragraph ID was retrieved) was **strictly removed** from the classifier's predictive feature set. Because gold paragraph IDs do not exist for new user questions at runtime, including it would introduce feature leakage. It is retained solely for offline retrieval diagnostics.

---

## 🛡 Methodology & Experimental Rigor

1. **Dataset:** SQuAD 2.0 (Stanford Question Answering Dataset), combining 100k answerable questions with 50k adversarial unanswerable questions written by crowdworkers to look identical to answerable questions.
2. **Leak-Free Article-Level Splitting:**
   - Standard random question splitting introduces severe **data leakage**; queries from the same Wikipedia article share passages, artificially inflating retrieval scores.
   - We implement strict **article-level stratified partitioning**: all questions and contexts belonging to an article are restricted to a single split.
3. **Partition Sizes (Maximum 500 Examples Total):**
   - **Train:** 300 questions (150 answerable / 150 unanswerable — 50%/50%; 176 unique articles)
   - **Validation:** 100 questions (50 answerable / 50 unanswerable — 50%/50%; 61 unique articles)
   - **Test:** 100 questions (50 answerable / 50 unanswerable — 50%/50%; 55 unique articles)
   - **Random Seed:** Fixed seed 42; 0 article overlap between train, val, and test.
4. **Classifier Pipeline:**
   ```python
   Pipeline([
       ('imputer', SimpleImputer(strategy='median')),
       ('scaler', StandardScaler()),
       ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
   ])
   ```
   - Tuned decision threshold $\tau = 0.42$ maximizes positive-class $F_1$ on validation (validation $F_1 = 0.6912$).
   - BM25 cutoff threshold $\tau_{\text{BM25}} = 13.15$ tuned on validation data.

---

## 📁 Repository Structure

```
Rag_ab/
├── academic_report/                 # Complete 45-page academic report & LaTeX source
│   ├── main.tex                    # Master LaTeX file
│   ├── main.pdf                    # Compiled IEEE-compliant formal report
│   ├── references.bib              # BibTeX literature citations
│   ├── README.md                   # Report-specific build instructions
│   ├── chapters/                   # 9 structured thesis chapters
│   │   ├── 01_introduction.tex
│   │   ├── 02_background.tex
│   │   ├── 03_methodology.tex
│   │   ├── 04_architecture.tex
│   │   ├── 05_experiments.tex
│   │   ├── 06_results.tex
│   │   ├── 07_error_analysis.tex
│   │   ├── 08_discussion.tex
│   │   └── 09_conclusion.tex
│   ├── appendices/                 # Appendix A (environment, features, code)
│   └── figures/                    # Publication-grade figures & diagrams
├── notebooks/                      # Experimental pipeline notebooks (01 to 09)
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_dataset_splitting.ipynb
│   ├── 04_bm25_retrieval.ipynb
│   ├── 05_question_answering.ipynb
│   ├── 06_abstention_strategies.ipynb
│   ├── 07_answerability_classifier.ipynb
│   ├── 08_evaluation.ipynb
│   └── 09_error_analysis.ipynb
├── models/                         # Serialized trained models
│   └── answerability_logreg.joblib # Trained logistic regression pipeline
├── results/                        # Experimental tables, metric CSVs, figures
│   ├── strategy_comparison.csv
│   ├── classifier_validation_thresholds.csv
│   └── figures/
├── src/                            # Modular Python source package
│   ├── __init__.py                 # Package exports
│   ├── data/
│   │   └── inspect_dataset.py      # Quick inspection of raw SQuAD JSONs
│   └── pipeline.py                 # Core inference pipeline & selective gating
├── app.py                          # Interactive Streamlit Web Application
├── generate_figures.py             # Script generating all 300-DPI thesis figures
├── requirements.txt                # Pinned dependency requirements
├── .gitignore                      # Git configuration (filters caches, large data & build artifacts)
└── README.md                       # Project documentation (this file)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- Git
- TeX Live or MiKTeX (optional, for compiling LaTeX report)

### 1. Clone the Repository
```bash
git clone https://github.com/Yussefech31/Rag_ab.git
cd Rag_ab
```

### 2. Set Up Virtual Environment
```bash
# On Linux / macOS
python -m venv rag
source rag/bin/activate

# On Windows (PowerShell)
python -m venv rag
.\rag\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ Execution & Reproducibility

### 1. Launch Interactive Streamlit Web Application
Run the interactive demonstration UI:
```bash
streamlit run app.py
```
This launches a browser interface where you can:
- Select curated answerable and adversarial unanswerable test queries (or enter custom questions).
- Interactively switch strategies (`Answerability Classifier`, `BM25 Threshold`, `Always Answer`).
- Adjust the decision threshold slider $\tau \in [0.05, 0.95]$ in real time.
- Inspect the 10 engineered features and candidate retrieved passages.

### 2. Running the Experimental Notebooks
Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab
```
Execute the notebooks in sequence from `01_dataset_exploration.ipynb` to `09_error_analysis.ipynb` to reproduce data splitting, indexing, feature extraction, classifier training, and final test set evaluation.

### 3. Running Automated Tests
The repository includes a comprehensive unit test suite covering feature extraction, pipeline inference, threshold gating, empty retrieval handling, and evaluation metrics:
```bash
pytest tests
```

### 4. Generating Scientific Figures
To recreate all publication figures (300 DPI) into `results/figures/`:
```bash
python generate_figures.py
```

---

## 📄 Compiling the Academic Report

The academic report is written in LaTeX using IEEE/standard academic styling. To compile it from source:

```bash
cd academic_report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The resulting `main.pdf` compiles the complete 45-page report with bibliography, table of contents, list of figures, and appendices.

---

## 📈 Scientific Figures & Artifacts

The project generates high-resolution analytical figures:
- **`fig2_dataset_distribution.png`**: Stratified article-level split distribution.
- **`fig3_bm25_score_distribution.png`**: Score distribution overlap between answerable and unanswerable queries.
- **`fig4_bm25_scores_per_split.png`**: Consistency of retrieval scores across Train, Validation, and Test.
- **`fig5_threshold_selection.png`**: Validation grid search for optimal decision threshold $\tau = 0.42$.
- **`fig6_strategy_comparison.png`**: Precision, Recall, F1, and Coverage comparison across strategies.
- **`fig7_precision_recall_curve.png`**: PR-AUC trajectory of the selective classifier.
- **`fig8_confusion_matrix.png`**: Test set confusion matrix showing true vs. predicted answerability.
- **`fig9_error_analysis.png`**: Categorization of false answers and false abstentions.

---

## 📜 Citation

If you find this research or codebase useful in your work, please cite our academic report:

```bibtex
@techreport{echchahlaoui2027knowing,
  author      = {Youssef Ech-Chahlaoui and Zouhair Fezzaz},
  title       = {Knowing When Not to Answer: Abstention in Retrieval-Augmented Question Answering},
  institution = {Ecole Nationale des Sciences Appliquees de Tetouan (ENSA Tetouan), Universite Abdelmalek Essaadi},
  year        = {2027},
  type        = {Research Project Report},
  address     = {Tetouan, Morocco},
  note        = {Supervised by Prof. Anas Belcaid}
}
```
