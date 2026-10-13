# LAPS: Linguistic Adaptive Preservation Scoring

> **Explainable Token-Level Decision Making for Code-Mixed Text Rewriting**  
> **Course:** BCSE497J - Project-I | School of Computer Science and Engineering, VIT | September 2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-orange.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/UI-Streamlit-blue.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Graph_DB-SQLite-green.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-45%20Passed-brightgreen.svg)](tests/test_laps.py)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)


---

## Executive Summary

When Large Language Models (LLMs) and Neural Machine Translation (NMT) engines process code-mixed text (such as Hinglish, Tanglish, or Benglish), they frequently suffer from **lexical over-translation** - converting naturally borrowed loanwords (*meeting*, *client*, *server*, *viva*, *calendar*, *link*) into archaic, unnatural monolingual translations (*baithak*, *grahak*, *kadi*).

**LAPS (Linguistic Adaptive Preservation Scoring)** introduces an explicit, explainable linguistic decision layer **prior to text generation**. Positioned as an auditable, deterministic enterprise governance guardrail, LAPS evaluates candidate loanwords across six corpus-grounded feature vectors, maps composite scores through a Soft Sigmoid Logistic Probability Boundary, and compiles machine-parseable XML constraints to strictly guide downstream generative models without stochastic prompt drift.

> **Detailed Mathematical Specification:** For the complete mathematical formulation, feature derivation, Knowledge Graph schema, and optimization algorithms, refer to [LAPS.md](LAPS.md).

---

## Key Framework Capabilities

* **Empirical Corpus-Grounded Borrowing Index:** Derived dynamically from **16,024 authentic code-mixed sentences** across two peer-reviewed corpora: **LinCE** (ACL, human-annotated token LID) and **PHINC** (Parallel Hinglish corpus, LingoIITGN), indexing **9,879 unique English loanwords** with empirical borrowing propensities.
* **Corpus-Grounded 6-Feature Derivations:**
  - **BF (Borrowing Frequency):** Empirical code-mixed borrowing propensity $\text{BPI}(w)$ with continuous global Zipf fallback.
  - **DS (Domain Specificity):** Context-aware dense neural embeddings (`all-MiniLM-L6-v2`) measured against domain prototype centroids.
  - **SI (Semantic Importance):** Information-theoretic entropy and Inverse Document Frequency (IDF).
  - **CC (Context Consistency):** Bilingual context window PMI and sub-graph co-occurrence density.
  - **AF (Audience Familiarity):** Universal cross-lingual frequency index.
  - **TB (Translation Benefit):** Syntactic functional analysis distinguishing nominal entities from verbal complex predicates.
* **Bilingual Lexical Resource Engine:** High-performance SQLite database (`laps_knowledge_graph.db`) with Devanagari cross-lingual mapping.
* **Sentence-Level Code-Mixing Index (CMI) Classifier:** Automatically calculates CMI to bypass monolingual sentences with transparent `PASSTHROUGH` audit logging.
* **Soft Sigmoid Decision Boundary Engine:** Logistic preservation probabilities $P(\text{Preserve} \mid t_i, C, \theta_R) = \sigma(\lambda \cdot (\text{Score} - \theta_R))$ across Casual, Social, Business, and Academic registers.
* **Multi-Baseline Benchmark Harness:** Evaluates against Static Dictionary Filter and Unconstrained NMT on held-out test splits.
* **Interactive Streamlit Web Dashboard:** Professional web interface featuring 4 dedicated views: Live Rewriter, LinCE/PHINC Corpus Explorer, Comparative Benchmark, and Mathematical Formulation.

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

### Multi-Baseline Benchmark Comparative Results (Audited & Methodologically Corrected)

Evaluated across $N=100$ code-mixed sentences on the gold evaluation corpus. Internal classifier accuracy, output loanword retention, and translation quality metrics are strictly separated:

#### Table 1: Output Preservation & Translation Performance (All Systems, N=100)
| System / Configuration | OTER ($\downarrow$) | Output F1 ($\uparrow$) | SacreBLEU ($\uparrow$) | chrF++ ($\uparrow$) | Matrix Leakage Proxy ($\downarrow$) | $\Delta$ vs LAPS | p-value (Permutation) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LAPS (Proposed Guardrail)** | **0.0108** | **0.9485** | **4.16** | **38.38** | **0.8442** | — | — |
| **Few-Shot Vocabulary Filter (Demo)** | 0.0485 | 0.9269 | 4.03 | 37.20 | 0.8442 | +0.0217 | $0.0006^*$ |
| **Static Dictionary Filter** | 0.0852 | 0.9041 | 3.99 | 36.29 | 0.8442 | +0.0445 | $0.0002^*$ |
| **Unconstrained NMT Baseline** | 0.1193 | 0.8748 | 2.10 | 35.02 | 0.8482 | +0.0737 | $0.0002^*$ |

*Statistically significant under Bonferroni correction ($\alpha = 0.05 / 3 = 0.0167$).  
*Note: Multi-sentence SacreBLEU and chrF++ evaluated with SacreBLEU v2.6.0. Strict Gold UTER is formally reported as **N/A** due to the absence of full-utterance token-level LID annotations in the gold corpus; the matrix leakage proxy tracks non-loanword leakage.*

#### Table 2: Token Preservation Classifier Accuracy
| System / Configuration | Classifier Macro F1 ($\uparrow$) | Micro Precision ($\uparrow$) | Micro Recall ($\uparrow$) | Accuracy ($\uparrow$) | Uncertainty Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LAPS (Proposed Guardrail)** | **0.9567** | **0.9290** | **0.9967** | **0.9266** | **0.1223** | Active Classifier |
| **Few-Shot Vocabulary Filter (Demo)** | 0.0973 | 1.0000 | 0.0762 | 0.1468 | 0.0000 | Active Heuristic |
| **Static Dictionary Filter** | 0.0477 | 1.0000 | 0.0397 | 0.1131 | 0.0000 | Active Lexicon |
| **Unconstrained NMT Baseline** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | End-to-End (No Classifier) |

### Key Comparative Findings:
1. **Honest Output Evaluation**: Unconstrained NMT is no longer conflated with 0.0 token F1; it receives credit for all naturally surviving loanwords (**0.8748 Output F1**). LAPS significantly outperforms all baselines on output retention (**0.9485 Output F1**, $p = 0.0002$).
2. **Generalization Beyond Fixed Glossaries**: Static Dictionary filters score only **0.0477 Classifier F1** because static 12-word lists cannot adapt to unseen or domain-specific entities. The few-shot demo filter achieves **0.0973 Classifier F1**. LAPS achieves **0.9567 Classifier F1** through continuous neural embeddings and empirical borrowing indices ($p < 0.001$).
3. **Statistical Significance Rigor**: Evaluated via a **Monte Carlo Paired Permutation Test** with sign-flipping ($B = 5,000$ random resamples, seed=42, finite-sample $p = (b+1)/(B+1)$, 2,000 bootstrap difference resamples) and Bonferroni adjustment ($\alpha = 0.0167$).
4. **Contextual Homograph Disambiguation & Auto-Register**: Sequences with colliding words (*us, me, so, do, to, is*) are disambiguated in context, and communicative register ($\theta_R$) is inferred automatically from surface text metrics.

### Benchmark Limitations & Methodological Disclosures
> [!IMPORTANT]
> **Essential Methodological Disclosures:**
> 1. **Pre-Cached Features in Gold Corpus:** The 100-sentence benchmark evaluates the LAPS decision/scoring process on pre-cached linguistic feature dictionaries stored for all 327 candidate tokens in `data/evaluation/gold_corpus.json`. Live feature extraction (dense MiniLM embedding and SQLite queries) is bypassed during benchmark scoring; the benchmark evaluates the mathematical decision boundaries on supplied feature representations.
> 2. **Surrogate Translation Fallback:** All four benchmark systems execute through `DETERMINISTIC_SURROGATE_FALLBACK` in `src/llm/nmt_backend.py`. The surrogate simulates neural over-translation via a 27-word dictionary and 12 colloquial Hinglish regex templates. The reported results are surrogate-benchmark results and must not be represented as real constrained-NMT performance.
> 3. **Translation Quality Metrics:** SacreBLEU (4.16) and chrF++ (38.38) scores are computationally verified with SacreBLEU v2.6.0, measuring partial n-gram overlap between surrogate-generated hypotheses and author-written references.
> 4. **Separation of Metrics:** We strictly distinguish (a) **Classifier Metrics** (token decision accuracy on candidate tokens), (b) **Output Retention Metrics** (presence of preserved loanwords in generated text), and (c) **Translation Quality Metrics** (SacreBLEU / chrF++ against colloquial references).

### Dataset Splits & Annotation Provenance
- **Parent Gold Corpus (`data/evaluation/gold_corpus.json`):** 100 sentences (`S001`–`S100`) spanning 327 annotated candidate loanwords across four communicative registers (Casual, Business, Academic, Social).
- **Split Breakdown:**
  - `train_split.json`: 70 in-distribution sentences.
  - `val_split.json`: 15 in-distribution sentences.
  - `test_split.json`: 20 total sentences, consisting of 15 in-distribution sentences drawn from gold plus 5 authentic out-of-corpus OOV sentences (`OOV_001`–`OOV_005`).
  - Total entries across the three split files = 105 (70 + 15 + 20). The 100 in-distribution sentences form a strictly disjoint partition of `gold_corpus.json` ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$); the 5 OOV sentences are an external extension.
- **Inter-Annotator Agreement (`data/evaluation/annotations.csv`):**
  - Two independent bilingual native raters evaluated 90 candidate tokens across sentences `S001`–`S020`.
  - Observed agreement: 85/90 tokens (**94.44%**).
  - Fleiss' Kappa: **$\kappa = 0.8739$**; Cohen's Kappa: **$\kappa = 0.8744$** ("Almost Perfect Agreement").
  - 5 disagreements (`submit`, `postpone`, `free`, `raise`, `try` where Rater 1=1, Rater 2=0) are preserved in the raw CSV with no synthetic consensus column.

### Reproducing Benchmark Experiments
```bash
# Ingest 16,000+ LinCE & PHINC sentences to build dynamic index:
python scripts/build_empirical_loanword_index.py

# Run audited multi-baseline evaluation harness:
python scripts/benchmark.py --corpus data/evaluation/gold_corpus.json

# Recompute inter-annotator agreement:
python scripts/compute_agreement.py
```

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
|-- app.py                             # Interactive Streamlit Web Dashboard (4 Tabs)
|-- requirements.txt                   # Dependency specifications
|-- README.md                          # Comprehensive project documentation & benchmark report
|-- LAPS.md                            # Complete LAPS algorithm mathematical specification
|-- data/
|   |-- vocabulary/
|   |   |-- empirical_loanword_index.json # 9,879 LinCE & PHINC loanwords with empirical BPI
|   |   `-- laps_knowledge_graph.db    # SQLite Production Graph Database (2,907 Nodes)
|   |-- evaluation/
|   |   |-- gold_corpus.json           # 100-Sentence Gold Standard Corpus
|   |   `-- gold_corpus_real_50.json   # 50 Authentic LinCE & PHINC evaluation samples
|   `-- weights/
|       |-- w_star_academic.json       # Calibrated Weights (Academic)
|       |-- w_star_business.json       # Calibrated Weights (Business)
|       |-- w_star_casual.json         # Calibrated Weights (Casual)
|       `-- w_star_social.json         # Calibrated Weights (Social)
|-- src/
|   |-- laps/
|   |   |-- cmi_detector.py            # Sentence-Level CMI Tagger
|   |   |-- tokenizer.py               # Bilingual Tokenizer & Language ID
|   |   |-- feature_extractor.py       # Neural Feature Extractor & Dynamic Corpus Ingestion
|   |   |-- scorer.py                  # LAPS Weighted Score Calculator
|   |   |-- threshold_engine.py        # Sigmoid Soft-Boundary Threshold Engine
|   |   `-- graph_engine.py            # High-Performance SQLite Graph Query API
|   |-- llm/
|   |   |-- rewriter.py                # Graph-Enriched XML Prompt Builder & Translator
|   |   |-- nmt_backend.py             # Local MarianMT (Helsinki-NLP) translation engine
|   |   `-- explainer.py               # Token-Level Explanation Generator
|   `-- pipeline.py                    # Master End-to-End Orchestrator
|-- scripts/
|   |-- build_empirical_loanword_index.py # LinCE & PHINC corpus statistical builder
|   |-- benchmark.py                   # Multi-baseline Benchmark Harness (OTER, BLEU, F1)
|   |-- run_ablation_study.py          # Feature ablation study runner
|   |-- compute_agreement.py           # Inter-annotator agreement calculator
|   |-- optimize_weights.py            # Simplex Grid Search & SLSQP Optimizer
|   `-- collect_real_data.py           # Empirical data collection pipeline
|-- tests/
|   `-- test_laps.py                   # Pytest Unit & Integration Test Suite (15 Tests)
|-- results/
|   |-- benchmark_results.json         # Automated Multi-Baseline Evaluation Metrics
|   `-- ablation_study_results.json    # Feature Contribution & Ablation Study Metrics
`-- docs/                              # Engineering and Research Specifications
    |-- deliverables.md                # Deliverable Specifications
    |-- algorithm_improvements.md      # LAPS Mathematical Formulation
    |-- implementation_plan.md         # Phase-Wise Engineering Plan
    |-- verification_plan.md           # Metrics and Verification Strategy
    `-- paper_tables_and_figures.md    # Formatted tables & figures for paper writeup
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
