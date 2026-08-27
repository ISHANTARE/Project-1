# LAPS Project Deliverables

This document details the concrete, verifiable deliverables produced by the **LAPS (Linguistic Adaptive Preservation Scoring)** project.

---

## 💻 1. Core Software Artifacts

The system is delivered as a modular Python package with a web interface:

### **LAPS Core Engine (`src/laps/`)**
* **`tokenizer.py`**: A bilingual, regex/subword tokenizer with integrated token-level Language Identification (LID).
* **`feature_extractor.py`**: Pre-generation vector calculator producing scores for BF, DS, SI, CC, AF, and TB.
* **`scorer.py`**: Mathematical weighted scoring loop using optimized weight vectors.
* **`threshold_engine.py`**: Context-register boundary manager containing adaptive configurations (Casual, Business, Academic, Social).

### **LLM Guiding Engine (`src/llm/`)**
* **`prompt_builder.py`**: Templates converting scoring decisions into hard XML-tagged constraint boundaries.
* **`rewriter.py`**: API caller (supporting OpenAI, Ollama, and Qwen) executing constrained decoding instructions.
* **`explainer.py`**: Explanatory logic compiler that formats token-level scoring metadata into clean justifications.

### **Interactive Frontend (`app.py`)**
* A **Streamlit-based dashboard** offering:
  * **Input Arena:** Simple interface for pasting code-mixed source sentences.
  * **Scoring Visualizer:** Interactive radar and bar charts detailing feature scores per token.
  * **Decision Log:** Explanatory panels detailing why words were preserved or rewritten.
  * **Evaluation Sandbox:** Comparative interface presenting output rewrites side-by-side with baselines.

---

## 📊 2. Datasets & Configurations

* **`data/vocabulary/`**:
  * **`borrowing_frequency.json`**: Empirical Hinglish word frequency database.
  * **`domain_lexicon.json`**: Registries of software, finance, and medical terms.
* **`data/eval_sets/`**:
  * **`laps_hinglish_benchmark.json`**: Pre-curated sentences with ground truth manual annotations for threshold boundary tuning.

---

## 📝 3. Academic & Patent Submissions

### **Research Paper: LaTeX Manuscript**
* **Scope:** Suitable for Scopus/IEEE venue submission.
* **Core sections:**
  * **Abstract & Introduction:** Framing the over-translation problem in code-mixing.
  * **Related Work:** Analyzing existing neural machine translation constraints.
  * **Methodology:** Complete mathematical details of the LAPS formula.
  * **Experimental Setup:** Baselines and dataset metrics.
  * **Results & Discussion:** Graphing quantitative metrics (BERTScore, ChrF++, OTER) and human evaluation scores.
  * **Conclusion & Future Directions.**

### **Patent Specification Draft**
* **Draft Title:** *A Computer-Implemented Adaptive Token Preservation Framework for Explainable Code-Mixed Multilingual Text Rewriting*
* **Includes:**
  * **Detailed Description:** Explaining the system architecture, scoring steps, and dynamic thresholds.
  * **Claims:** Structuring independent method, system, and apparatus claims.
  * **Flowcharts:** High-quality diagrams illustrating system pipelines and decision loops.
