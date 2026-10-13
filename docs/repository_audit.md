# LAPS Repository Audit & Complete Inventory
**Project Domain:** Natural Language Processing, Code-Mixed Text Rewriting, Neuro-Symbolic AI, Explainable AI  
**Course:** BCSE497J — Project-I, Vellore Institute of Technology, Vellore  
**Audit Date:** October 2026 / Local Workspace Audit  

---

## 1. Executive Summary

This document provides a comprehensive inventory and source-level audit of the **LAPS (Linguistic Adaptive Preservation Scoring)** research software repository. LAPS is an explainable, pre-generation neuro-symbolic guardrail system designed to prevent lexical over-translation during code-mixed (Hinglish/Indic) text translation and rewriting into English.

The audit verified all source files, modules under `src/laps/` and `src/llm/`, Streamlit application (`app.py`), datasets, models, databases, scripts, results, and documentation.

---

## 2. Repository Inventory

### 2.1 Python Source Code (`src/`)

| Module Path | Size | Implementation Status | Purpose & Role |
| :--- | :--- | :--- | :--- |
| `src/__init__.py` | 21 B | Verified | Package root initialization |
| `src/pipeline.py` | 4.7 KB | Verified & Functional | End-to-end orchestrator coordinating CMI detection, tokenization, feature extraction, scoring, thresholding, XML compilation, rewriting, and explanation |
| `src/laps/__init__.py` | 26 B | Verified | Subpackage root initialization |
| `src/laps/cmi_detector.py` | 3.8 KB | Verified & Functional | Sentence-level Code-Mixing Index (Gambäck & Das, 2014) and language mode detection (`MONOLINGUAL_EN`, `MONOLINGUAL_HI`, `CODE_MIXED`) |
| `src/laps/tokenizer.py` | 7.3 KB | Verified & Functional | Subword/word tokenization, script tagging (Latin/Devanagari/punct), lexical Hindi markers, and homograph collision handling |
| `src/laps/neural_lid.py` | 15.1 KB | Verified & Functional | Character-CNN (`CharCNN`) PyTorch model inference (`char_lid.pt`), character n-gram profiles, phonetic normalizer, continuous posterior probability calibration |
| `src/laps/register_detector.py` | 5.0 KB | Verified & Functional | Automated register prediction (`casual`, `social`, `business`, `academic`) with dynamic formality threshold assignment ($\theta_R \in [0.35, 0.65]$) |
| `src/laps/feature_extractor.py` | 13.9 KB | Verified & Functional | 6-feature extractor (BF, DS, SI, CC, AF, TB) combining MiniLM contextual vector embeddings, empirical Zipf indices, and Knowledge Graph attributes |
| `src/laps/graph_engine.py` | 5.9 KB | Verified & Functional | SQLite engine querying `laps_knowledge_graph.db` for bilingual translation penalties, domain categories, and context co-occurrence PMI |
| `src/laps/scorer.py` | 2.4 KB | Verified & Functional | Computes weighted preservation score $\text{Score}(t) \in [0.0, 1.0]$ with default simplex weights and register overrides |
| `src/laps/threshold_engine.py` | 12.8 KB | Verified & Functional | Soft Sigmoid logistic preservation probability boundary $\sigma(\lambda \cdot (\text{Score} - \theta_R))$, KG ego-density modulation, 3-zone decision mapping (`PRESERVE`, `UNCERTAIN`, `REWRITE`) |
| `src/llm/__init__.py` | 25 B | Verified | Subpackage root initialization |
| `src/llm/nmt_backend.py` | 11.2 KB | Verified & Functional | Real Hugging Face MarianMT (`Helsinki-NLP/opus-mt-hi-en`) with forced lexical decoding constraints (`force_words_ids`), and deterministic rule-based surrogate fallback |
| `src/llm/rewriter.py` | 13.8 KB | Verified & Functional | Graph-enriched XML prompt builder (`<preservation_constraints>`), Few-Shot ICL prompt builder, Zero-Shot baseline, and model dispatch (NMT, OpenAI, Ollama, Offline) |
| `src/llm/explainer.py` | 7.3 KB | Verified & Functional | Token-level explanation generator producing plain-English linguistic justifications and structured markdown audit tables |

### 2.2 User Interface

| File | Size | Implementation Status | Purpose & Role |
| :--- | :--- | :--- | :--- |
| `app.py` | 29.8 KB | Verified & Functional | Interactive Streamlit research platform with 4 tabs: Live Interactive Rewriter, LinCE & PHINC Empirical Corpus Explorer, Comparative Benchmark & Evaluation, and Mathematical Specification |

### 2.3 Evaluation Datasets & Vocabulary Assets (`data/`)

| Path | Size / Count | Status | Description |
| :--- | :--- | :--- | :--- |
| `data/vocabulary/laps_knowledge_graph.db` | 491 KB (SQLite) | Verified | SQLite 3 database: `nodes` table (2,907 entries) and `edges` table (2,940 entries) linking English loanwords and Hindi equivalents with domain, frequency, and penalty attributes |
| `data/vocabulary/empirical_loanword_index.json` | 1.21 MB (59.3k lines) | Verified | 9,879 unique English loanwords extracted from 16,024 Hinglish sentences (LinCE + PHINC) with counts, document frequencies, and empirical BPI scores |
| `data/vocabulary/borrowing_frequency.json` | 5.0 KB | Verified | Auxiliary lexical borrowing index for common vocabulary items |
| `data/vocabulary/domain_lexicon.json` | 2.5 KB | Verified | Domain-anchored vocabulary seeds for Tech, Business, Academic, Medical, Finance |
| `data/models/char_lid.pt` | 101 KB (PyTorch) | Verified | Trained Character-CNN weights for subword Latin-script language identification |
| `data/weights/w_star_academic.json` | 588 B | Verified | SLSQP-optimized weights for academic register ($\mathbf{w}^* = [0.0, 0.2, 0.2, 0.4, 0.2, 0.0]$) |
| `data/weights/w_star_business.json` | 524 B | Verified | SLSQP-optimized weights for business register |
| `data/weights/w_star_casual.json` | 586 B | Verified | SLSQP-optimized weights for casual register |
| `data/weights/w_star_social.json` | 586 B | Verified | SLSQP-optimized weights for social register |
| `data/evaluation/gold_corpus.json` | 66.3 KB | Verified | 100 code-mixed sentences across 4 registers with token-level binary preservation gold labels |
| `data/evaluation/gold_corpus_real_50.json` | 13.0 KB | Verified | 50 real-world sentences extracted directly from LinCE and PHINC |
| `data/evaluation/train_split.json` | 74.0 KB | Verified | Development split for cross-validation weight optimization |
| `data/evaluation/val_split.json` | 16.6 KB | Verified | Validation split |
| `data/evaluation/test_split.json` | 19.8 KB | Verified | Held-out test split with OOV test cases |
| `data/evaluation/annotations.csv` | 5.7 KB | Verified | Multi-annotator evaluation dataset: 90 candidate tokens across 20 sentences annotated independently by 2 raters |

### 2.4 Research Scripts (`scripts/`)

| Script | Purpose | Status |
| :--- | :--- | :--- |
| `scripts/benchmark.py` | Multi-baseline evaluation runner comparing LAPS, Few-Shot ICL, Static Dictionary, and Unconstrained NMT across OTER, UTER, BLEU, chrF++, Token F1, and paired bootstrap significance | Audited (Identified UTER surrogate factor) |
| `scripts/run_ablation_study.py` | Feature ablation runner measuring F1 impact of removing BF, DS, SI, CC, AF, TB, and Soft Sigmoid | Audited (Identified weight indexing bug) |
| `scripts/compute_agreement.py` | Computes Fleiss' Kappa, Gwet's AC1, observed and expected agreement from multi-rater CSV | Verified & Reproducible (Fleiss $\kappa = 0.8739$) |
| `scripts/optimize_weights.py` | SLSQP and grid search optimizer finding register-specific weight vectors $\mathbf{w}^*$ | Verified |
| `scripts/build_empirical_loanword_index.py` | Ingests LinCE and PHINC datasets and computes Borrowing Preservation Index | Verified |
| `scripts/human_eval_export.py` | Exports structured CSV for Likert-scale human annotation | Verified |
| `scripts/ingest_real_datasets.py` | Populates SQLite Knowledge Graph with verified nodes and edges | Verified |
| `scripts/train_char_lid.py` | Trains PyTorch CharCNN model on Latin-script tokens | Verified |
| `scripts/collect_real_data.py` | Prepares annotation templates for rater studies | Verified |
| `scripts/prepare_public_benchmark.py` | Splits datasets into train, validation, and held-out test sets | Verified |
| `scripts/generate_vit_report_docx.py` | Automated institutional Project-I report generator (.docx) | Verified |
| `scripts/generate_vit_ppt.py` | Automated presentation deck generator (.pptx) | Verified |

### 2.5 Tests (`tests/`)

| File | Status | Coverage |
| :--- | :--- | :--- |
| `tests/test_laps.py` | **18/18 Tests Passing** | CMI detection, Tokenization & LID, Char-CNN inference, Feature extraction & TB differentiation, KG engine, Scorer & Sigmoid boundary, Rewriter XML construction, End-to-end pipeline, Benchmark splits integrity, Homograph disambiguation, Register detection |

---

## 3. Discrepancies & Issues Identified During Audit

1. **Dependency Omission (`requirements.txt`):**
   - `wordfreq`, `fpdf2`, and `sentencepiece` were missing from `requirements.txt`.
   - When `wordfreq` was missing, dynamic Zipf calculations failed gracefully to default constants, which caused `TestEndToEndPipeline.test_pipeline_execution` to fail under academic register threshold ($\theta_R = 0.65$).
   - *Remediation:* Installed `wordfreq`, `fpdf2`, and `sentencepiece`, and updated `requirements.txt`. All 18 tests pass immediately.

2. **Database Schema Discrepancy:**
   - In `LAPS.md` (§3), the edge schema is documented with columns `source_id` and `target_id`.
   - In `laps_knowledge_graph.db` and `graph_engine.py`, the actual column names are `source`, `target`, `relation`, `weight`, `naturalness_penalty`.
   - *Status:* Recorded in mathematical audit; verified that `graph_engine.py` uses the actual schema correctly.

3. **Annotator Count Inconsistency:**
   - `data/evaluation/gold_corpus.json` header specifies `"annotators": 3`.
   - `data/evaluation/annotations.csv` contains annotations from 2 raters (`rater_1`, `rater_2`) for 90 candidate tokens across 20 sentences.
   - *Status:* Verified that Fleiss' Kappa of $0.8739$ and observed agreement of $94.44\%$ are reproduced exactly on the 2-rater annotation file.

4. **Under-Translation Error Rate (UTER) Shared Value ($\approx 0.8442$):**
   - In `results/benchmark_results.json`, LAPS, Few-Shot ICL, and Static Dictionary all reported identical UTER of $0.8442$.
   - *Investigation Finding:* When running offline benchmarking without HuggingFace model weights loaded, all three systems routed their output generation through `_generate_realistic_nmt_translation()` in `nmt_backend.py`. Because that surrogate only translates a small set of predefined grammatical markers (`kal` $\to$ `tomorrow`, `pe` $\to$ `on`, `nahi` $\to$ `not`), approximately 84.4% of non-loanword Hindi words remained untranslated in the output text across all three systems.
   - *Status:* Thoroughly documented as a benchmark surrogate phenomenon rather than a mathematical metric error.

5. **Ablation Weight Indexing Bug:**
   - In `scripts/run_ablation_study.py`, `w_active = {k: weights.get(f"w{i+1}", 0.20) for i, k in enumerate(active_keys)}` indexed weights by enumeration order ($i=0 \to w_1$) rather than key name, shifting weight semantics when removing features.
   - Additionally, comparing `prob >= 0.50` is mathematically equivalent to `score >= 0.50` under the logistic sigmoid, causing the "Hard Step Threshold" ablation to yield identical F1 ($0.9603$).
   - *Status:* Documented and slated for code remediation.

6. **Missing User-Facing Report Exports:**
   - Streamlit `app.py` previously lacked export buttons for downloading complete rewriting run audits as JSON, PDF, or CSV.
   - *Status:* Slated for implementation in `src/laps/exporter.py` and `app.py`.
