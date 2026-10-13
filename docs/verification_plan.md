# LAPS Verification and Testing Plan

This document details the verification protocols used to validate the **LAPS (Linguistic Adaptive Preservation Scoring)** system.

---

## 1. Unit and Functional Verification

We use automated unit tests to verify modular component logic:

| Component | Target File | Verification Criteria | Status |
| :--- | :--- | :--- | :--- |
| **CMI Tagger** | `cmi_detector.py` | Validates CMI calculation and monolingual English classification. | **Passed** |
| **Tokenizer** | `tokenizer.py` | Asserts Hindi grammar markers vs English loanword tagging. | **Passed** |
| **Feature Extractor** | `feature_extractor.py` | Verifies neural vector similarity bounds in [0.0, 1.0]. | **Passed** |
| **Graph Engine** | `graph_engine.py` | Asserts O(1) SQLite database queries on 2,907 nodes. | **Passed** |
| **Sigmoid Engine** | `threshold_engine.py` | Verifies logistic preservation probability and decision zones. | **Passed** |
| **LLM Rewriter** | `rewriter.py` | Verifies fluent Hinglish-to-English translation and XML constraints. | **Passed** |
| **Optimizer** | `optimize_weights.py` | Asserts simplex constraint satisfaction (sum w_j = 1, w_j >= 0). | **Passed** |
| **Benchmark Suite** | `benchmark.py` | Validates OTER, BLEU, and Token F1 computation over gold corpus. | **Passed** |
| **Human Eval Export** | `human_eval_export.py` | Asserts CSV structure, token counts, and blank rater columns. | **Passed** |
| **Streamlit App** | `app.py` | Verifies web app loads at `http://localhost:8501`. | **Passed** |

---

## 2. Automated Benchmark Metrics

LAPS output rewrites are benchmarked over the 100-sentence evaluation corpus against baseline systems:

* **Token F1:** Measures accuracy of token preservation classification against gold human labels.
* **Over-Translation Error Rate (OTER):**
  $$\text{OTER} = \frac{\text{Number of valid loanwords incorrectly translated}}{\text{Total loanwords in source sentence}}$$
* **BLEU:** Measures n-gram precision against human reference rewrites.

---

## 3. Human Evaluation Protocol

A double-blind survey with N = 25 bilingual annotators assessing 100 code-mixed evaluation sentences across 4 qualitative axes (Likert scale 1-5):
1. **Readability and Fluency**
2. **Semantic Retention**
3. **Over-Translation Avoidance**
4. **Explanation Trustworthiness**

Inter-Annotator Agreement is computed using **Fleiss' Kappa**.
