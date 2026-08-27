# LAPS Project Roadmap

This document outlines the complete 12-week timeline and milestone strategy for **LAPS (Linguistic Adaptive Preservation Scoring)**.

---

## Academic Milestone Alignment and Current Progress

```mermaid
gantt
    title LAPS Development and Review Timeline
    dateFormat  YYYY-MM-DD
    section Academic Reviews
    Review 1 (Concept and Architecture)    :milestone, done, 2026-09-01, 1d
    Review 2 (Core Pipeline and Alg)       :milestone, done, 2026-10-01, 1d
    Review 3 (Evaluation and Metrics)      :milestone, active, 2026-11-01, 1d
    Final Review and Demo                  :milestone, 2026-11-20, 1d
    section Development Status
    Phase 1: Preprocessing and LID         :done, 2026-08-28, 7d
    Phase 2: Neural Scorer and Features    :done, 2026-09-04, 14d
    Phase 3: 2907-Node Graph DB            :done, 2026-09-18, 14d
    Phase 4: Web Demo Dashboard            :done, 2026-10-02, 7d
    Phase 5: Sigmoid and XML Prompter      :done, 2026-10-09, 14d
    Phase 6: Weight Optimization and Bench :done, 2026-10-23, 14d
    Phase 7: Gold Corpus and Human Eval    :done, 2026-11-01, 7d
```

---

## Weekly Timeline Breakdown

### Week 1-2: Preprocessing, LID and Baselines [COMPLETED]
* **Status:** Completed.
* **Deliverables:** Built `tokenizer.py` with subword/phonetic Hindi language identification (LID) and sentence-level CMI classifier (`cmi_detector.py`).

### Week 3-4: Feature Extraction Engine and Neural Embeddings [COMPLETED]
* **Status:** Completed.
* **Deliverables:** Built `feature_extractor.py` integrating `sentence-transformers` (`all-MiniLM-L6-v2`) zero-shot vector embeddings and fuzzy typo matching (`difflib`).

### Week 5-6: Production Knowledge Graph and Adaptive Thresholds [COMPLETED]
* **Status:** Completed.
* **Deliverables:** Ingested `wordfreq` Zipf frequencies into **2,907-Node, 2,940-Edge SQLite Knowledge Graph (`laps_knowledge_graph.db`)**. Built dynamic threshold engine (`threshold_engine.py`) and Streamlit web dashboard (`app.py`).

### Week 7-8: Sigmoid Probability Engine and XML Prompter [COMPLETED]
* **Status:** Completed.
* **Deliverables:** Implemented Soft Sigmoid Logistic Preservation Probability engine ($P(\text{Preserve} \mid t)$), KG ego-density scoring, and Graph-Enriched XML Prompt Generator.

### Week 9-10: Automated Weight Optimization and Benchmarking [COMPLETED]
* **Status:** Completed.
* **Deliverables:** Implemented `optimize_weights.py` for weight vector tuning ($\mathbf{w}^*$). Built automated benchmark harness (`benchmark.py`) calculating OTER, BLEU, and Token F1.

### Week 11-12: Gold Corpus Expansion and Human Evaluation Suite [COMPLETED]
* **Status:** Completed.
* **Deliverables:** Expanded `gold_corpus.json` to 100 balanced sentences with negative labels, and built `human_eval_export.py` for 3-rater double-blind evaluation studies.
