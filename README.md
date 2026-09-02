# LAPS: Linguistic Adaptive Preservation Scoring

> **Explainable Token-Level Decision Making for Code-Mixed Text Rewriting**  
> **Course:** BCSE497J - Project-I | School of Computer Science and Engineering, VIT | September 2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-orange.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/UI-Streamlit-blue.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Graph_DB-SQLite-green.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-15%20Passed-brightgreen.svg)](tests/test_laps.py)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Executive Summary

When Large Language Models (LLMs) and Neural Machine Translation (NMT) engines process code-mixed text (such as Hinglish, Tanglish, or Benglish), they frequently suffer from **lexical over-translation** - converting naturally borrowed loanwords (*meeting*, *client*, *server*, *viva*, *calendar*, *link*) into archaic, unnatural monolingual translations (*baithak*, *grahak*, *kadi*).

**LAPS (Linguistic Adaptive Preservation Scoring)** introduces an explicit, explainable linguistic decision layer **prior to text generation**. Positioned as an auditable, deterministic enterprise governance guardrail, LAPS evaluates candidate loanwords across six corpus-grounded feature vectors, maps composite scores through a Soft Sigmoid Logistic Probability Boundary, and compiles machine-parseable XML constraints to strictly guide downstream generative models without stochastic prompt drift.

> **Detailed Mathematical Specification:** For the complete mathematical formulation, feature derivation, Knowledge Graph schema, and optimization algorithms, refer to [LAPS.md](LAPS.md).

---

## Key Framework Capabilities

* **PyTorch Subword Character-CNN Language Identifier:** Employs an authentic 1D-CNN over character embeddings (`char_lid.pt`) with Bayesian subword n-gram log-likelihood fallback to classify Romanized Indic vs English tokens.
* **Corpus-Grounded 6-Feature Derivations:**
  - **BF (Borrowing Frequency):** Continuous Zipf distribution over 150k+ words with Laplace smoothing.
  - **DS (Domain Specificity):** Context-aware dense neural embeddings (`all-MiniLM-L6-v2`) measured against domain prototype centroids.
  - **SI (Semantic Importance):** Information-theoretic entropy and Inverse Document Frequency (IDF).
  - **CC (Context Consistency):** Sub-graph co-occurrence density & bilingual context window PMI.
  - **AF (Audience Familiarity):** Universal cross-lingual frequency index.
  - **TB (Translation Benefit):** Dynamic syntactic functional analysis distinguishing nominal entities from functional action modifiers.
* **Symmetric Bidirectional Knowledge Graph:** Powered by **2,907 Nodes and 2,940 Edges** in a high-performance SQLite database (`laps_knowledge_graph.db`).
* **Sentence-Level Code-Mixing Index (CMI) Classifier:** Automatically calculates CMI to bypass monolingual sentences with transparent `PASSTHROUGH` audit logging.
* **Soft Sigmoid Decision Boundary Engine:** Logistic preservation probabilities $P(\text{Preserve} \mid t_i, C, \theta_R) = \sigma(\lambda \cdot (\text{Score} - \theta_R))$ across Casual, Social, Business, and Academic registers.
* **Multi-Baseline Benchmark Harness:** Evaluates against Zero-Shot LLM, Few-Shot ICL, Static Dictionary, and Unconstrained NMT on held-out test splits with out-of-vocabulary (OOV) tokens.
* **Interactive Streamlit Web Dashboard:** Professional web interface featuring token decision maps, 6-axis hexagonal polar radar plots, probability distribution bar charts, and exportable audit reports.

---

## System Architecture

```
                                  [Code-Mixed Input Sentence]
                                               |
                                               v
                                  +--------------------------+
                                  |   CMI Language Tagger    |
                                  +------------+-------------+
                                               | Candidate Loanwords
                                               v
                                  +--------------------------+
                                  |  Feature Extractor & KG  |<-- SQLite Graph DB
                                  +------------+-------------+   (2,907 Nodes, 2,940 Edges)
                                               | 6-Feature Vectors (BF, DS, SI, CC, AF, TB)
                                               v
                                  +--------------------------+
                                  |   LAPS Scorer Engine     |
                                  +------------+-------------+
                                               | LAPS Score in [0, 1]
                                               v
                                  +--------------------------+
                                  | Sigmoid Threshold Engine | (Casual: 0.35, Business: 0.50, Academic: 0.65)
                                  +------------+-------------+
                                               | P(Preserve) & Decision Zones
                         +---------------------+---------------------+
                         |                                           |
                         v                                           v
           +--------------------------+                +--------------------------+
           |    XML Prompt Builder    |                |  Explanation Generator   |
           |     + LLM Rewriter       |                |    (Structured Table)    |
           +-------------+------------+                +-------------+------------+
                         |                                           |
                         v                                           v
              [Rewritten Output Text]                     [Token Justifications]
```

---

## The LAPS Mathematical Formula

The Preservation Score for a candidate loanword token $t_i$ in context window $C$ is computed via:

$$\text{LAPSScore}(t_i) = w_1 \text{BF}(t_i) + w_2 \text{DS}(t_i) + w_3 \text{SI}(t_i) + w_4 \text{CC}(t_i, C) + w_5 \text{AF}(t_i) - w_6 \text{TB}(t_i)$$

Where:
* **$\text{BF}$ (Borrowing Frequency):** Queried from the SQLite Knowledge Graph or `wordfreq` Zipf frequency ratios.
* **$\text{DS}$ (Domain Specificity):** Zero-shot neural cosine similarity against domain vector anchors.
* **$\text{SI}$ (Semantic Importance):** Information density and semantic drift risk under translation.
* **$\text{CC}$ (Context Consistency):** Sub-graph co-occurrence density ($\text{CC}_{\text{ego}}$) within Knowledge Graph $G$.
* **$\text{AF}$ (Audience Familiarity):** Conversational recognition score across bilingual audiences.
* **$\text{TB}$ (Translation Benefit):** Syntactic benefit of translating $t_i$ to improve target language fluency (subtracted).

### Soft Sigmoid Logistic Preservation Probability

$$P(\text{Preserve} \mid t_i, C, \theta_R) = \frac{1}{1 + e^{-\lambda (\text{LAPSScore}(t_i) - \theta_R)}}$$

With $\lambda = 10.0$ and dynamic register thresholds $\theta_R \in \{\text{Casual: 0.35}, \text{Social: 0.40}, \text{Business: 0.50}, \text{Academic: 0.65}\}$.

Three operational decision zones:
* **$P \ge 0.70 \implies \text{PRESERVE}$:** High-confidence preservation.
* **$0.50 \le P < 0.70 \implies \text{UNCERTAIN}$:** Borderline candidate; retained with advisory notice.
* **$P < 0.50 \implies \text{REWRITE}$:** Eligible for fluent, natural translation.

---

## Algorithmic Feasibility & Benchmark Framework

The primary objective of this project is to **demonstrate that an explicit, explainable token-preservation algorithm is feasible and technically viable** for code-mixed text rewriting, rather than competing against large-scale frontier LLMs.

### Experimental Setup & Baselines
The LAPS pipeline is evaluated against reproducible, transparent baselines:
1. **LAPS (Proposed)**: Dynamic linguistic scoring + sigmoid thresholding + constrained generation.
2. **Static Dictionary Filter**: Rule-based lexical constraint using a fixed glossary of frequent loanwords.
3. **Unconstrained NMT Baseline**: Raw neural translation without preservation constraints (subject to over-translation errors).

### Evaluation Harness
```bash
# Generate annotation template for human evaluators
python scripts/collect_real_data.py

# Calculate inter-rater agreement from real judgements
python scripts/compute_agreement.py --annotations data/evaluation/annotations.csv

# Run multi-baseline evaluation harness
python scripts/benchmark.py --corpus data/evaluation/gold_corpus.json
```

> **Note on Evaluation:** Benchmark results are computed over controlled test sets evaluating Over-Translation Error Rate (OTER), Token-level F1, BLEU, and chrF++. The framework includes paired bootstrap significance testing to validate whether LAPS preserves loanwords significantly better than unconstrained baselines.

---

## Quickstart

### 1. Installation

Clone the repository and install required dependencies:
```bash
git clone https://github.com/ISHANTARE/Project-1.git
cd Project-1
pip install -r requirements.txt
```

### 2. Launch Streamlit Web App

Start the interactive dashboard locally:
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### 3. Run Automated Tests

Execute the full unit and integration test suite:
```bash
pytest tests/
```

### 4. Run Benchmark Harness

Run automated benchmark experiments against baseline systems:
```bash
python scripts/benchmark.py
```

### 5. Export Human Evaluation Dataset

Export the 100-sentence dataset for double-blind human annotation:
```bash
python scripts/human_eval_export.py
```

### 6. Calibrate Optimal Weights

Compute optimal weights $\mathbf{w}^*$ for any communicative register:
```bash
python scripts/optimize_weights.py --corpus data/evaluation/gold_corpus.json --register business
```

---

## Python API Usage

```python
from src.pipeline import run

# Run LAPS on a Hinglish sentence under the Academic register
result = run(
    sentence="hi! Kisi ke pass viva calender ki link h kya?",
    register="academic",
    llm_backend="mock"  # Options: 'mock', 'openai', 'ollama'
)

print("Preserved Tokens:", result["preserve_list"])
# Output: ['viva', 'calendar', 'link']

print("Rewritten Sentence:", result["rewrite_result"]["rewritten"])
# Output: "Hi! Does anyone have the link for the viva calendar?"

print("Summary:", result["summary"])
# Output: {'total_candidates': 3, 'preserve_count': 3, 'uncertain_count': 0, 'rewrite_count': 0, 'avg_probability': 0.884}
```

---

## Repository Structure

```
PROJECT-1/
|-- app.py                             # Interactive Streamlit Web Dashboard
|-- requirements.txt                   # Dependency specifications
|-- README.md                          # Comprehensive project documentation
|-- LAPS.md                            # Complete LAPS algorithm mathematical specification
|-- data/
|   |-- vocabulary/
|   |   |-- laps_knowledge_graph.db    # SQLite Production Graph Database (2,907 Nodes)
|   |   `-- laps_knowledge_graph.json  # Serialized Node-Link JSON Graph
|   |-- evaluation/
|   |   `-- gold_corpus.json           # 100-Sentence Gold Standard Corpus
|   `-- weights/
|       |-- w_star_academic.json       # Calibrated Weights (Academic)
|       |-- w_star_business.json       # Calibrated Weights (Business)
|       |-- w_star_casual.json         # Calibrated Weights (Casual)
|       `-- w_star_social.json         # Calibrated Weights (Social)
|-- src/
|   |-- laps/
|   |   |-- cmi_detector.py            # Sentence-Level CMI Tagger
|   |   |-- tokenizer.py               # Bilingual Tokenizer & Language ID
|   |   |-- feature_extractor.py       # Neural Feature Extractor & KG Queries
|   |   |-- scorer.py                  # LAPS Weighted Score Calculator
|   |   |-- threshold_engine.py        # Sigmoid Soft-Boundary Threshold Engine
|   |   `-- graph_engine.py            # High-Performance SQLite Graph Query API
|   |-- llm/
|   |   |-- rewriter.py                # Graph-Enriched XML Prompt Builder & Translator
|   |   `-- explainer.py               # Token-Level Explanation Generator
|   `-- pipeline.py                    # Master End-to-End Orchestrator
|-- scripts/
|   |-- benchmark.py                   # OTER, BLEU, and Token F1 Benchmark Harness
|   |-- optimize_weights.py            # Simplex Grid Search & SLSQP Optimizer
|   |-- human_eval_export.py           # Double-Blind Annotation CSV Exporter
|   `-- ingest_real_datasets.py        # Knowledge Graph Ingestion Builder
|-- tests/
|   `-- test_laps.py                   # Pytest Unit & Integration Test Suite (12 Tests)
|-- results/
|   |-- benchmark_results.json         # Automated Evaluation Metrics
|   `-- human_eval_export.csv          # Human Evaluation Annotation Sheet
`-- docs/                              # Engineering and Research Specifications
    |-- roadmap.md                     # 12-Week Academic Milestone Calendar
    |-- deliverables.md                # Deliverable Specifications
    |-- algorithm_improvements.md      # LAPS Mathematical Formulation
    |-- implementation_plan.md         # Phase-Wise Engineering Plan
    |-- verification_plan.md           # Metrics and Verification Strategy
    `-- final_phase_research_plan.md   # Final Research Blueprint
```

---

## Citation and License

If you use LAPS or the Bilingual Code-Mixed Knowledge Graph in your research, please cite:

```bibtex
@article{antare2026laps,
  title={LAPS: Linguistic Adaptive Preservation Scoring for Explainable Token-Level Decision Making in Code-Mixed Text Rewriting},
  author={Antare, Ishan and Research Team},
  journal={VIT School of Computer Science and Engineering},
  year={2026}
}
```

Distributed under the **MIT License**.
