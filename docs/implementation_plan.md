# Implementation Plan: Rigorous Scientific Remediation & Research Publication Roadmap for LAPS

This implementation plan provides a systematic, sprint-by-sprint engineering and research roadmap to remediate every critical flaw, unverified assumption, and methodological gap identified in [adversarial_research_critique.md](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/docs/adversarial_research_critique.md). The objective is to transition LAPS from a simulated prototype into a validated, reproducible, and publishable paper for a top-tier NLP/AI conference (ACL, EMNLP, COLING, or IEEE Transactions).

---

## 1. Problem-to-Remediation Mapping Matrix

| Critique Vulnerability | Underlying Defect | Target Remediation | Sprint & Phase |
| :--- | :--- | :--- | :--- |
| **1. Circular Mock Evaluation** | `benchmark.py` routes all baselines through mock regex script; identical metrics across baselines | Connect real NMT (`IndicTrans2`, `NLLB-200`) and real LLM backends (`Llama-3.3-70B`, `GPT-4o-mini`); enforce genuine constrained vs unconstrained execution | **Phase 1 (Sprint 1.1 & 1.2)** |
| **2. Missing Essential Baselines** | No comparison against In-Context Learning (ICL) or standard NMT | Implement 4 control baselines: Zero-Shot LLM, Few-Shot ICL with Loanword Rules, IndicTrans2, and Static Lexicon Constraint | **Phase 1 (Sprint 1.3)** |
| **3. Brittle Token-Level LID** | `tokenizer.py` relies on static exact sets; misspelling/slang defaults to English candidates | Replace heuristic set matching with neural subword / character-n-gram Romanized Indic LID (e.g. `fastText-Indic` / fine-tuned Indic RoBERTa) | **Phase 2 (Sprint 2.1 & 2.2)** |
| **4. Hardcoded Heuristic Features** | $BF, SI, AF, TB$ are hardcoded lists; $DS$ relies on static word bags | Re-derive $BF/AF$ from real corpus Zipf distributions (IndicCorp/LinCE); replace $SI/DS$ with contextual token embeddings (`xlm-roberta-base`); derive $TB$ from dependency parse roles | **Phase 3 (Sprint 3.1 & 3.2)** |
| **5. Terminology Inflation** | SQLite key-value store labeled as a "Knowledge Graph" | Reframe as "Bilingual Lexical Graph & Co-occurrence Index" and implement true graph neighborhood/subgraph density calculations | **Phase 3 (Sprint 3.3)** |
| **6. Dataset Scale & Agreement** | $N=100$ hand-crafted sentences without inter-annotator agreement | Expand evaluation corpus to $N \ge 1,000$ sentences across 4 domains; conduct 3-rater annotation with Fleiss' $\kappa \ge 0.75$ | **Phase 4 (Sprint 4.1)** |
| **7. SLSQP Weight Overfitting** | 6 continuous weights optimized on 100 sentences without data splits | Implement 5-fold cross-validation with stratified 70/15/15 train/dev/test splits in `optimize_weights.py` | **Phase 4 (Sprint 4.2)** |
| **8. Multi-Language Scope** | Hardcoded Hindi markers make Tanglish/Benglish claims invalid | Abstract LID and lexical indices into modular language packs (Hindi, Tamil, Bengali, Telugu) | **Phase 4 (Sprint 4.3)** |
| **9. Lack of Human Evaluation** | No qualitative human validation on Fluency and Adequacy | Execute double-blind human evaluation on 100 stratified test samples across 3 bilingual raters with paired bootstrap significance testing ($p < 0.01$) | **Phase 5 (Sprint 5.1)** |
| **10. Latency & Prompt Overhead** | Verbose XML prompt increases token costs and adds pre-processing lag | Implement compact JSON/tag constraints mode; benchmark latency vs Few-Shot ICL | **Phase 5 (Sprint 5.2)** |

---

## 2. Phased Sprint Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAPS RESEARCH IMPLEMENTATION ROADMAP                     │
└─────────────────────────────────────────────────────────────────────────────┘
  Phase 1: Empirical Integrity & Real Baselines           (Sprints 1.1 - 1.3)
    │
    ▼
  Phase 2: Subword Neural Language ID & Noise Resilience  (Sprints 2.1 - 2.2)
    │
    ▼
  Phase 3: Context-Aware Feature Engine & Dynamic Graph   (Sprints 3.1 - 3.3)
    │
    ▼
  Phase 4: Scaled Multi-Annotator Corpus & Validation     (Sprints 4.1 - 4.3)
    │
    ▼
  Phase 5: Human Evaluation, Significance & Submission   (Sprints 5.1 - 5.2)
```

---

### **Phase 1: Empirical Evaluation Integrity & Real Baselines**
*Objective: Completely eliminate mock evaluation and establish genuine, reproducible model comparisons.*

#### **Sprint 1.1: Real NMT Engine Integration**
- Integrate HuggingFace inference for **AI4Bharat IndicTrans2** (`ai4bharat/indictrans2-indic-en-1B` / `indictrans2-en-indic-1B`) and **Meta NLLB-200** (`facebook/nllb-200-distilled-600M`).
- Implement constrained decoding wrapper utilizing lexically constrained beam search with HuggingFace `ForceTokensLogitsProcessor` or grid beam search.
- **Files:**
  - `[NEW]` [src/llm/nmt_backend.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/llm/nmt_backend.py)
  - `[MODIFY]` [src/llm/rewriter.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/llm/rewriter.py)

#### **Sprint 1.2: Real LLM Backend Adapter**
- Connect OpenAI API (`gpt-4o-mini`, `gpt-4o`) and local Ollama/vLLM endpoints (`llama3.3:70b`, `qwen2.5:32b`).
- Ensure unconstrained ablation actually executes without any preservation constraints, producing real, unconstrained translation outputs.
- **Files:**
  - `[MODIFY]` [src/llm/rewriter.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/llm/rewriter.py)

#### **Sprint 1.3: Multi-Baseline Benchmark Harness**
- Refactor `scripts/benchmark.py` to evaluate 5 distinct systems:
  1. **Zero-Shot LLM** (Unconstrained)
  2. **Few-Shot In-Context Learning (ICL)** (Standard prompt with loanword retention guidelines)
  3. **IndicTrans2** (Raw NMT)
  4. **IndicTrans2 + Lexical Constraints** (Static dictionary)
  5. **LAPS + LLM / IndicTrans2** (Proposed neuro-symbolic framework)
- Compute real **OTER**, **sacreBLEU**, **chrF++**, **BERTScore F1**, and **Token F1**.
- **Files:**
  - `[MODIFY]` [scripts/benchmark.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/scripts/benchmark.py)

---

### **Phase 2: Subword Neural Language ID & Noise Resilience**
*Objective: Prevent upstream LID collapse on informal, uncurated Romanized Indic text.*

#### **Sprint 2.1: Character-n-gram / Subword Neural LID**
- Implement a character-n-gram and subword neural classifier using `fasttext` or fine-tuned `mBERT`/`IndicBERT` on Romanized code-mixed datasets (LinCE / SAIL).
- Eliminate hardcoded `HINDI_MARKERS` exact dictionary checks.
- Add phonetic normalization step (Soundex/Metaphone adapted for Romanized Indic: *chahiye = chahye = chaiye*).
- **Files:**
  - `[MODIFY]` [src/laps/tokenizer.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/laps/tokenizer.py)
  - `[NEW]` [src/laps/neural_lid.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/laps/neural_lid.py)

#### **Sprint 2.2: CMI Tagger Refactoring**
- Update `cmi_detector.py` to use probabilistic language tags ($P(\text{HI})$, $P(\text{EN})$) rather than discrete binary string lookups.
- Validate on the LinCE Code-Mixing Benchmark.
- **Files:**
  - `[MODIFY]` [src/laps/cmi_detector.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/laps/cmi_detector.py)

---

### **Phase 3: Context-Aware Feature Engine & Dynamic Graph**
*Objective: Replace static hardcoded heuristics with corpus-derived statistics and contextual representations.*

#### **Sprint 3.1: Contextual Semantic Drift ($SI$) and Domain Specificity ($DS$)**
- Replace single-word `SentenceTransformer` cosine similarity with contextual token representations from `xlm-roberta-base`.
- Compute $SI$ as the contextual representation shift under target language projection (measuring semantic information loss if translated).
- Compute $DS$ by measuring attention weight / similarity against domain prototype contexts rather than static word bags.
- **Files:**
  - `[MODIFY]` [src/laps/feature_extractor.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/laps/feature_extractor.py)

#### **Sprint 3.2: Dynamic Corpus-Driven Statistics ($BF, AF, TB$)**
- Derive Borrowing Frequency ($BF$) and Audience Familiarity ($AF$) from empirical Zipf distributions computed over the IITB Parallel Corpus and IndicCorp Hinglish subsets.
- Compute Translation Benefit ($TB$) dynamically using dependency parsing: nouns and core entity arguments receive low $TB$, while auxiliary verbs and functional modifiers receive high $TB$.
- **Files:**
  - `[MODIFY]` [src/laps/feature_extractor.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/laps/feature_extractor.py)
  - `[NEW]` [scripts/build_corpus_frequencies.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/scripts/build_corpus_frequencies.py)

#### **Sprint 3.3: Graph Engine Refinement & Subgraph Co-occurrence**
- Update SQLite database schema and query API to compute genuine graph-theoretic metric: Subgraph Co-occurrence Density ($\text{CC}_{\text{ego}}$) and Pointwise Mutual Information (PMI) across bilingual co-occurrence windows.
- Update documentation to transparently define the graph structure.
- **Files:**
  - `[MODIFY]` [src/laps/graph_engine.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/src/laps/graph_engine.py)

---

### **Phase 4: Scaled Multi-Annotator Corpus & Cross-Validation**
*Objective: Build a solid, statistically sound empirical foundation with $N \ge 1,000$ sentences.*

#### **Sprint 4.1: Dataset Expansion & Multi-Rater Annotation Protocol**
- Curate $N=1,000$ code-mixed sentences balanced across 4 domains: Tech (250), Academic (250), Business (250), and Social/Casual (250).
- Deploy annotation interface (`scripts/human_eval_export.py`) for 3 independent bilingual annotators.
- Compute and report **Fleiss' $\kappa$** and **Cohen's $\kappa$** ($ target \ge 0.75$) to guarantee ground truth reliability.
- **Files:**
  - `[NEW]` [data/evaluation/gold_corpus_v2_1k.json](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/data/evaluation/gold_corpus_v2_1k.json)
  - `[MODIFY]` [scripts/human_eval_export.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/scripts/human_eval_export.py)
  - `[NEW]` [scripts/compute_agreement.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/scripts/compute_agreement.py)

#### **Sprint 4.2: 5-Fold Cross-Validation for Weight Optimization ($\mathbf{w}^*$)**
- Modify `scripts/optimize_weights.py` to enforce stratified 5-fold cross-validation (70% train, 15% dev, 15% test).
- Optimize $\mathbf{w}^*$ exclusively on training folds; evaluate and report out-of-fold generalization performance to prevent overfitting claims.
- **Files:**
  - `[MODIFY]` [scripts/optimize_weights.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/scripts/optimize_weights.py)

#### **Sprint 4.3: Multilingual Generalization Packs**
- Abstract language-specific parameters into modular language packs: Hinglish (`hi-en`), Tanglish (`ta-en`), and Benglish (`bn-en`).
- **Files:**
  - `[NEW]` [data/vocabulary/language_packs/](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/data/vocabulary/language_packs/)

---

### **Phase 5: Human Evaluation, Significance & Publication Artifacts**
*Objective: Complete full paper requirements including double-blind human evaluation and statistical testing.*

#### **Sprint 5.1: Double-Blind Human Evaluation & Statistical Significance**
- Conduct double-blind human preference evaluation ($N=100$ stratified samples) rated by 3 independent evaluators on 1-5 Likert scales:
  - **Adequacy** (Meaning preservation)
  - **Fluency** (Grammatical naturalness)
  - **Preservation Precision** (Appropriate loanword retention without archaic over-translation)
- Compute **Paired Bootstrap Resampling** ($10^4$ iterations) and **McNemar's test** reporting $p$-values ($p < 0.01$).
- **Files:**
  - `[NEW]` [scripts/run_human_evaluation_study.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/scripts/run_human_evaluation_study.py)
  - `[NEW]` [results/human_eval_significance_results.json](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/results/human_eval_significance_results.json)

#### **Sprint 5.2: Feature Ablation Study & Latency Benchmarks**
- Execute comprehensive ablation matrix:
  - Full LAPS vs $(-\text{BF})$ vs $(-\text{DS})$ vs $(-\text{SI})$ vs $(-\text{CC})$ vs $(-\text{AF})$ vs $(-\text{TB})$ vs Single-Threshold (Hard cutoff).
- Profile latency and token overhead comparing XML constraints vs compact JSON tags vs Few-Shot prompting.
- Compile complete camera-ready LaTeX tables for the paper.
- **Files:**
  - `[NEW]` [scripts/run_ablation_study.py](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/scripts/run_ablation_study.py)
  - `[NEW]` [docs/paper_tables_and_figures.md](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/docs/paper_tables_and_figures.md)

---

## 3. Verification Plan

### **Automated Testing Suite**
- Run unit and integration tests across new neural modules:
  ```bash
  pytest tests/ -v
  ```
- Run automated cross-validation and baseline benchmark:
  ```bash
  python scripts/benchmark.py --corpus data/evaluation/gold_corpus_v2_1k.json --models indictrans2,gpt-4o-mini,llama3
  ```
- Run ablation experiment suite:
  ```bash
  python scripts/run_ablation_study.py
  ```

### **Manual & Qualitative Verification**
- Verify that OTER, BLEU, and Token F1 scores across systems are distinct, non-trivial, and statistically sound.
- Verify Fleiss' $\kappa \ge 0.75$ across human annotators on the expanded dataset.
- Confirm that informal transliteration inputs (*"smjh ni aya cl bhej do"*) are cleanly handled by the neural LID without false-positive English candidate tagging.
