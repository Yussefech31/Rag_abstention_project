# Knowing When Not to Answer: Abstention in Retrieval-Augmented Question Answering
## Academic Research Report — ENSA Tétouan (Université Abdelmalek Essaâdi)

**Authors:** Youssef Ech-Chahlaoui, Zouhair Fezzaz  
**Supervisor:** Prof. Anas Belcaid  
**Academic Year:** 2026–2027  

---

## Directory Overview

```
academic_report/
│── main.tex                 # Master document (compiles entire report)
│── main.pdf                 # Compiled 45-page formal academic report
│── references.bib           # IEEE-compliant BibTeX bibliography
│── README.md                # Compilation instructions & overview
│── chapters/
│   ├── 01_introduction.tex  # Problem definition & RAG motivation
│   ├── 02_background.tex    # SQuAD 2.0, BM25 theory, RoBERTa & selective prediction
│   ├── 03_methodology.tex   # Leak-free splitting, features & 3 abstention strategies
│   ├── 04_architecture.tex  # TikZ end-to-end system flowchart & descriptions
│   ├── 05_experiments.tex   # Experimental setup, hardware & metric definitions
│   ├── 06_results.tex       # Comparative tables, statistical & curve evaluations
│   ├── 07_error_analysis.tex# Error taxonomy & representative qualitative cases
│   ├── 08_discussion.tex    # In-depth discussion on retrieval vs. reader signals
│   └── 09_conclusion.tex    # Contributions summary, limitations & future work
│── appendices/
│   └── appendix_a.tex       # Environment specs, feature table & pipeline code
└── figures/
    ├── ensa_logo.png
    ├── fig2_dataset_distribution.png
    ├── fig3_bm25_score_distribution.png
    ├── fig4_bm25_scores_per_split.png
    ├── fig5_threshold_selection.png
    ├── fig6_strategy_comparison.png
    ├── fig7_precision_recall_curve.png
    ├── fig8_confusion_matrix.png
    └── fig9_error_analysis.png
```

---

## How to Compile from Source

Ensure you have a modern TeX distribution installed (MiKTeX or TeX Live).

Run the standard multi-pass build:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The resulting `main.pdf` contains the complete report with rendered cross-references, list of figures, list of tables, and IEEE-styled citations.
