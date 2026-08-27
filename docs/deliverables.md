# LAPS Project Deliverables

This document details the concrete, verifiable software, algorithmic, and experimental deliverables produced by the **LAPS (Linguistic Adaptive Preservation Scoring)** project.

---

## 1. Core Software Artifacts

### LAPS Core Engine (`src/laps/`)
* **`cmi_detector.py`**: Sentence-level Code-Mixing Index (Gamback and Das, 2014) detector distinguishing monolingual English versus code-mixed Hinglish.
* **`tokenizer.py`**: Subword/phonetic Language Identifier (LID) tagging Latin/Devanagari scripts and transliterated Hindi grammar markers.
* **`feature_extractor.py`**: Neural feature calculator utilizing `sentence-transformers` (`all-MiniLM-L6-v2`) zero-shot embeddings and fuzzy typo alignment (`difflib`).
* **`graph_engine.py`**: High-performance SQLite query API for `laps_knowledge_graph.db`.
* **`scorer.py`**: Multi-factor LAPS weighted scoring calculation over 6 linguistic feature axes.
* **`threshold_engine.py`**: Soft Sigmoid logistic probability boundary engine with three decision zones (PRESERVE, UNCERTAIN, REWRITE) and KG ego-density modulation.

### Production Knowledge Graph Database (`data/vocabulary/`)
* **`laps_knowledge_graph.db`**: SQLite Graph Database containing **2,907 Nodes and 2,940 Multi-Relational Edges** storing Zipf frequencies, Devanagari script mappings, naturalness penalties, and reverse-borrowing indices.
* **`laps_knowledge_graph.json`**: Node-link JSON graph representation.

### Evaluation and Experimental Harnesses (`scripts/` and `data/evaluation/`)
* **`data/evaluation/gold_corpus.json`**: 100-sentence gold standard corpus annotated with human token preservation labels across 4 communicative registers (Casual, Business, Academic, Social).
* **`scripts/optimize_weights.py`**: Simplex grid search and SLSQP local optimizer generating calibrated weight vectors $\mathbf{w}^*$.
* **`scripts/benchmark.py`**: Automated evaluation harness measuring Over-Translation Error Rate (OTER), BLEU, and Token F1.
* **`scripts/human_eval_export.py`**: Automated exporter generating double-blind CSV sheets for 3-rater Fleiss' Kappa agreement studies.

### LLM Guiding Engine (`src/llm/`)
* **`rewriter.py`**: Graph-enriched XML prompt builder supporting OpenAI (GPT-4o-mini), local Ollama (Llama 3), and an offline fluent Hinglish-to-English natural translator.
* **`explainer.py`**: Rationale generator producing structured, transparent token-level justifications.

### Interactive Frontend (`app.py`)
* **Streamlit Web Dashboard (http://localhost:8501):**
  * Token decision maps (PRESERVE / UNCERTAIN / REWRITE / NATIVE pills).
  * 6-axis hexagonal radar charts and probability bar plots with threshold overlays.
  * Constrained rewrite output box with preserved token list.
  * Graph-enriched XML prompt inspector.
  * Token explanation tables with CSV export functionality.

---

## 2. Academic Review 2 Deliverables (`Review-2/`)

* **`Review-2/LAPS_Review_2_Draft_Report.md`**: Full report draft matching Chapters 1-4 of the official curriculum template.
* **`Review-2/LAPS_Review_2_Presentation.pptx`**: Generated 13-slide PowerPoint deck covering all evaluation rubrics.
* **`Review-2/LAPS_Review_2_PPT_Content.md`**: Markdown version of PPT slides with speaking script.
