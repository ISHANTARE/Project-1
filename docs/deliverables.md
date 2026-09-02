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

### Production Lexical & Corpus Artifacts (`data/vocabulary/`)
* **`empirical_loanword_index.json`**: Empirical borrowing index compiled from **16,024 LinCE and PHINC code-mixed sentences**, indexing **9,879 unique English loanword tokens** with occurrence count, document frequency, and continuous borrowing propensity scores ($\text{BPI}$).
* **`laps_knowledge_graph.db`**: High-performance SQLite Database containing **2,907 Nodes and 2,940 Multi-Relational Edges** storing Zipf frequencies, Devanagari script mappings, naturalness penalties, and reverse-borrowing indices.

### Evaluation and Experimental Harnesses (`scripts/` and `data/evaluation/`)
* **`data/evaluation/gold_corpus.json`**: 100-sentence gold standard corpus annotated with human token preservation labels across 4 communicative registers (Casual, Business, Academic, Social).
* **`data/evaluation/gold_corpus_real_50.json`**: 50 authentic evaluation sentences extracted directly from LinCE and PHINC benchmarks.
* **`scripts/build_empirical_loanword_index.py`**: Automated pipeline downloading and parsing LinCE and PHINC to compute statistical borrowing propensities.
* **`scripts/benchmark.py`**: Automated multi-baseline evaluation harness measuring Over-Translation Error Rate (OTER), BLEU, chrF++, and Token F1.
* **`scripts/run_ablation_study.py`**: Automated ablation study runner quantifying individual feature contributions across all 6 axes.
* **`scripts/compute_agreement.py`**: Inter-annotator agreement calculator computing Fleiss' Kappa from human annotations.
* **`scripts/optimize_weights.py`**: Simplex grid search and SLSQP local optimizer generating calibrated weight vectors $\mathbf{w}^*$.

### LLM Guiding Engine (`src/llm/`)
* **`rewriter.py`**: Graph-enriched XML prompt builder supporting OpenAI (GPT-4o-mini), local Ollama (Llama 3), local MarianMT NMT, and an offline rule-based translator.
* **`nmt_backend.py`**: Local neural translation engine using HuggingFace Helsinki-NLP Opus-MT models.
* **`explainer.py`**: Rationale generator producing structured, transparent token-level justifications.

### Interactive Frontend (`app.py`)
* **Streamlit Web Dashboard (http://localhost:8501) with 4 Comprehensive Tabs:**
  1. **Live Interactive Rewriter:** Token decision maps, 6-axis hexagonal radar charts, score distribution bar plots, and structured audit logs.
  2. **LinCE & PHINC Empirical Corpus Explorer:** Searchable, filterable table of 9,879 loanwords with dynamic frequencies and borrowing scores.
  3. **Comparative Benchmark & Evaluation:** Side-by-side performance cards, bar plots, and statistical findings comparing LAPS vs Static Dictionary vs Unconstrained NMT.
  4. **LAPS Mathematical Formulation:** Full interactive LaTeX equations detailing the multidimensional scoring formula and Soft Sigmoid probability function.

---

## 2. Academic Review 2 Deliverables (`Review-2/`)

* **`Review-2/LAPS_Review_2_Draft_Report.md`**: Full report draft matching Chapters 1-4 of the official curriculum template.
* **`Review-2/LAPS_Review_2_Presentation.pptx`**: Generated 13-slide PowerPoint deck covering all evaluation rubrics.
* **`Review-2/LAPS_Review_2_PPT_Content.md`**: Markdown version of PPT slides with speaking script.
