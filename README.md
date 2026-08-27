# LAPS: Linguistic Adaptive Preservation Scoring

> **Explainable Token-Level Decision Making for Code-Mixed Text Rewriting**  
> **Course Project:** BCSE497J Project-I | School of Computer Science and Engineering, VIT | September 2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-orange.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/UI-Streamlit-blue.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Graph_DB-SQLite-green.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Project Overview

When modern Large Language Models (LLMs) rewrite code-mixed text (such as Hinglish, Tanglish, or Benglish), they frequently suffer from **lexical over-translation** - translating naturally borrowed loanwords (*meeting*, *client*, *server*, *viva*, *calendar*, *link*) into rigid, unnatural monolingual equivalents.

**LAPS (Linguistic Adaptive Preservation Scoring)** introduces an explicit, explainable linguistic decision layer **before** text generation. Rather than treating text rewriting as a black-box end-to-end task, LAPS evaluates candidate loanword tokens across six distinct linguistic and contextual feature vectors against dynamic context thresholds. It guides downstream LLMs with structured XML constraints while generating transparent, human-auditable rationales for every preservation decision.

---

## Key Features

* **Symmetric Bidirectional Knowledge Graph:** Powered by **2,907 Nodes and 2,940 Edges** in a high-performance SQLite database (`laps_knowledge_graph.db`) storing English Zipf frequencies, Devanagari script mappings, naturalness replacement penalties, and reverse-borrowing indices.
* **Sentence-Level Code-Mixing Index (CMI) Tagger:** Automatically measures the Code-Mixing Index (Gamback and Das, 2014) to distinguish pure monolingual English/Hindi text from code-mixed sentences.
* **Soft Sigmoid Decision Engine:** Replaces hard binary cutoffs with continuous logistic preservation probabilities $P(\text{Preserve} \mid t_i, C, \theta_R) = \sigma(\lambda \cdot (\text{Score} - \theta_R))$ with boundary classification (Preserve, Uncertain, Rewrite).
* **Graph-Enriched XML Prompt Builder:** Injects machine-parseable constraints with domain metadata and naturalness penalties to eliminate LLM translation hallucinations.
* **Weight Optimization Harness:** Computes optimal 6-feature weight vectors ($\mathbf{w}^*$) using Grid Search and SLSQP optimization over human-labeled gold standard corpora.
* **Human Evaluation and Benchmark Suite:** Supports double-blind Likert-scale human annotation exports, OTER (Over-Translation Error Rate), Token F1, and BLEU benchmarking.
* **Interactive Streamlit Web Dashboard:** Professional UI featuring token decision maps, hexagonal radar charts, threshold overlays, XML prompt inspect, and CSV report downloads.

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
                                               | P(Preserve) & Decisions: PRESERVE / UNCERTAIN / REWRITE
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

## Quickstart

### 1. Prerequisites and Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/ISHANTARE/Project-1.git
cd Project-1
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard
Run the Streamlit application locally:
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser to test code-mixed input sentences.

### 3. Run Automated Benchmark
Execute the automated evaluation suite:
```bash
python scripts/benchmark.py
```

### 4. Export Human Evaluation Dataset
Export the 100-sentence gold standard corpus for double-blind human annotation:
```bash
python scripts/human_eval_export.py
```

---

## Python API Usage

Execute the LAPS pipeline directly in your Python code:

```python
from src.pipeline import run

# Run LAPS on a Hinglish sentence under the Business register
result = run(
    sentence="hi! Kisi ke pass viva calender ki link h kya?",
    register="academic",
    llm_backend="mock"  # Options: 'mock', 'openai', 'ollama'
)

print("Preserved Tokens:", result["preserve_list"])
# Output: ['viva', 'calendar', 'link']

print("Rewritten Sentence:", result["rewrite_result"]["rewritten"])
# Output: "Hi! Does anyone have the link for the viva calendar?"
```

---

## Repository Structure

```
PROJECT-1/
|-- app.py                             # Interactive Streamlit Web Dashboard
|-- requirements.txt                   # Dependency specifications
|-- README.md                          # Project documentation and guide
|-- data/
|   |-- vocabulary/
|   |   |-- laps_knowledge_graph.db    # SQLite Production Graph Database (2,907 Nodes)
|   |   `-- laps_knowledge_graph.json  # Serialized Node-Link JSON Graph
|   |-- evaluation/
|   |   `-- gold_corpus.json           # 100-Sentence Gold Standard Corpus
|   `-- weights/
|       `-- w_star_business.json       # Mathematically Optimized Weights
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
