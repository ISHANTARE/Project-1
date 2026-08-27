# LAPS: Linguistic Adaptive Preservation Scoring
## Adversarial Research Paper Critique & Publishability Roadmap

> **Target Assessment:** Tier-1/Tier-2 NLP & AI Venues (*ACL, EMNLP, COLING, NAACL, IEEE Transactions on Audio, Speech, and Language Processing, IEEE Access*)  
> **Evaluation Mode:** Peer-Reviewer Stress-Test & Adversarial Gap Analysis  
> **Document Location:** `docs/adversarial_research_critique.md`

---

## 1. Executive Research Verdict

### **Verdict: 🔴 High Risk / Reject in Current Form — Major Methodological & Empirical Vulnerabilities**

**Summary Assessment:**  
LAPS addresses an authentic and impactful NLP challenge: **lexical over-translation and unnatural loanword substitution in code-mixed (Hinglish/Indic) text processing**. The motivation is sound, but from a publishable research paper perspective, the manuscript and implementation in their current state would receive an immediate **Desk Reject or Strong Reject** from peer reviewers at ACL, EMNLP, or IEEE venues.

### **The Core Scientific Deficiencies:**
1. **Benchmark Circularity & Invalidation:** The empirical comparisons reported in the project do not evaluate actual Neural Machine Translation (NMT) engines or Large Language Models (LLMs). The benchmark harness (`scripts/benchmark.py`) passes all systems through a deterministic Python regex script (`_call_fluent_translator`), rendering all reported Over-Translation Error Rates ($0.0192$) and BLEU scores ($0.3904$) identical and scientifically invalid.
2. **Missing Essential Baselines:** The project fails to benchmark against state-of-the-art multilingual Indic models (such as **AI4Bharat IndicTrans2**, **mBART-50**, **NLLB-200**) and frontier LLM in-context prompting baselines (**GPT-4o**, **Llama-3.3-70B** with standard few-shot preservation prompts).
3. **Corpus Scale & Inter-Annotator Agreement:** The evaluation relies on a toy dataset of $N=100$ hand-crafted sentences without documented multi-annotator agreement metrics (e.g., Cohen’s or Fleiss’ $\kappa$).
4. **Heuristic-Driven Language Identification (LID):** The upstream pipeline uses static word dictionaries and regex suffixes rather than robust character/subword neural classifiers, causing catastrophic error propagation on uncurated, phonetically varied Romanized Indic text.

---

## 2. Critical Scientific & Methodological Flaws

```
                     ┌─────────────────────────────────────────────────────────┐
                     │          THE CRITICAL METHODOLOGICAL BOTTLENECK         │
                     └─────────────────────────────────────────────────────────┘

   [Real User Input] ──► [Static Tokenizer/LID] ──► [Hardcoded Dictionaries] ──► [Mock Regex Script]
          │                        │                              │                       │
          ▼                        ▼                              ▼                       ▼
   Phonetic Variations     Misclassifies noisy            Collinear features        Identical metrics
   (e.g., "smjh ni aya")  Hindi tokens as English       (BF/AF/SI/TB are static)   across all baselines
```

### **Flaw 1: Evaluation Invalidation Due to Mock Engine Execution**
* **Why Reviewers Will Reject:** Reviewers evaluate empirical claims based on experimental validity. Claiming that LAPS outperforms baselines when all baselines share a mock regex function constitutes benchmark fabrication.
* **Evidence in Code:** In `scripts/benchmark.py` (lines 144, 173, 191), `run_laps`, `run_static_dict_filter`, and `run_unconstrained` all call `_call_fluent_translator(raw, preserve_list)`. Because `_call_fluent_translator` relies on rigid regex templates (e.g., `Kisi ke pass (.+?) ki link h kya?`), it returns identical text regardless of the passed constraint list.
* **Reviewer Severity:** 10/10 | **Likelihood of Detection:** 10/10 | **Difficulty to Fix:** Moderate (requires integrating real model APIs/checkpoints).

---

### **Flaw 2: Absence of In-Context Learning (ICL) Control Baselines**
* **Why Reviewers Will Reject:** Any paper proposing a complex pre-generation neuro-symbolic pipeline must prove that a standard prompt on modern LLMs cannot achieve the exact same result with zero pre-processing overhead.
* **Missing Baseline:** `LLM(Prompt="Translate the following Hinglish sentence to natural English, preserving all commonly used English technical loanwords: {sentence}")`. If zero-shot or 2-shot prompting matches LAPS's token retention without an external SQLite graph, the paper's primary contribution is nullified.
* **Reviewer Severity:** 9/10 | **Likelihood of Detection:** 9/10 | **Difficulty to Fix:** Low (add prompt ablation experiments).

---

### **Flaw 3: Static Rule Dependency Masquerading as Feature Extraction**
* **Why Reviewers Will Reject:** The 6-feature formulation ($BF, DS, SI, CC, AF, TB$) claims to provide multi-dimensional linguistic scoring, but the underlying implementation in `src/laps/feature_extractor.py` reduces to hardcoded tables:
  * $BF$ and $AF$ query the same static dictionary (`WELL_KNOWN_LOANWORDS`).
  * $SI$ returns $0.90$ for an arbitrary 20-word list, and $0.50$ for all other words.
  * $TB$ returns $0.85$ for 6 specific verbs (`postpone`, `reschedule`, `arrange`, etc.) and $0.15$ otherwise.
  * $DS$ compares tokens against static word-bag strings via `SentenceTransformer`, vulnerable to polysemy errors.
* **Reviewer Severity:** 8/10 | **Likelihood of Detection:** 8/10 | **Difficulty to Fix:** High (requires dynamic corpus-driven statistical indices).

---

### **Flaw 4: Insufficient Dataset Scale & Lack of Agreement Metrics**
* **Why Reviewers Will Reject:** A 100-sentence evaluation set is insufficient for statistical significance testing (such as paired bootstrap resampling or McNemar's test). Without Fleiss' $\kappa$ across at least 3 independent native annotators, the gold labels reflect individual subjective bias.
* **Reviewer Severity:** 8/10 | **Likelihood of Detection:** 9/10 | **Difficulty to Fix:** Moderate (expand to 1,000+ sentences with multi-annotator agreement).

---

## 3. Hidden Linguistic & Engineering Assumptions

| Assumption | Why It Might Be Wrong in Practice | Scientific Consequence if Violated | Severity |
| :--- | :--- | :--- | :---: |
| **Romanized Indic follows clean standard orthography** | Users write phonetically (*kal / cl / kall*, *bhejo / snd krdo*, *chahiye / chahye / chaiye*). | Non-standard Hindi words default to `EN` candidate tokens and are preserved as un-translated noise. | **Critical** |
| **Loanword preservation is token-invariant** | Whether a term like *meeting* should be preserved depends on communicative register and pragmatic intent (*"office meeting"* vs *"pehli meeting / mulaqat"*). | Static per-token mathematical scoring fails to account for semantic and pragmatic context. | **High** |
| **SLSQP weight optimization ($\mathbf{w}^*$) on 100 sentences generalizes** | Optimizing 6 continuous parameters on 100 sentences without a cross-validation split leads to extreme empirical overfitting. | Calibrated weights fail when tested on out-of-domain or conversational datasets. | **High** |
| **Cosine similarity to domain anchor strings captures true domain specificity** | Word-level embeddings ignore context (*bank* in financial tech vs river bank; *class* in OOP vs university lecture). | Polysemous loanwords receive incorrect domain weights and inappropriate preservation tags. | **Medium** |

---

## 4. Technical Failure Points in Linguistic Pipelines

```
[Input: "bhai kal ka ppt aur docx link share kr dena plzz"]
               │
               ▼
   [src/laps/tokenizer.py]
   • "kr" not in HINDI_MARKERS   ──► Misclassified as EN candidate!
   • "plzz" not in HINDI_MARKERS ──► Misclassified as EN candidate!
               │
               ▼
   [src/laps/feature_extractor.py]
   • Neural DS on "plzz" ────────► Default fallback (0.40)
   • Neural DS on "kr"   ────────► Default fallback (0.40)
               │
               ▼
   [src/laps/scorer.py + threshold_engine.py]
   • False positive candidate tokens get preserved
               │
               ▼
   [src/llm/rewriter.py (XML Injection)]
   • <token name="kr" action="PRESERVE">
   • <token name="plzz" action="PRESERVE">
               │
               ▼
[Output: "Brother please share yesterday's ppt and docx link kr plzz"] (Degraded Output)
```

### **1. Upstream LID Error Cascade**
Because `tokenize()` classifies any Latin-script token not explicitly found in `HINDI_MARKERS` as an English candidate (`is_candidate = True`), every informal transliteration spelling error instantly cascades into a false-positive loanword candidate.

### **2. Out-of-Vocabulary (OOV) Collapse**
Any loanword not present in `laps_knowledge_graph.db` or `WELL_KNOWN_LOANWORDS` falls back to default constants ($BF=0.40, SI=0.50, TB=0.15$). The mathematical formula effectively collapses to a constant baseline for all genuine open-vocabulary test cases.

### **3. Latency & Token Overhead**
For each sentence, the pipeline runs:
$$\text{Tokenization} \longrightarrow \text{CMI Tagger} \longrightarrow \text{Neural Embeddings} \longrightarrow \text{SQLite Lookups} \longrightarrow \text{XML Compilation} \longrightarrow \text{LLM Call}$$
This adds significant pre-processing latency ($100\text{--}300\text{ ms}$) and expands LLM prompt token counts by $300\text{--}500\%$, which reviewers will question in production efficiency evaluations.

---

## 5. Academic Prior-Art & Competitor Landscape

To establish publishable novelty, LAPS must clearly differentiate itself from existing literature:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PRIOR ART LANDSCAPE                             │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ Existing Paradigm             │ How LAPS Must Scientifically Differentiate  │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 1. Constrained NMT Decoding   │ Grid Beam Search (Post & Vilar, 2018)       │
│    (Lexical Constraints)      │ requires static user glossaries. LAPS       │
│                               │ dynamically determines WHAT to constrain.   │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 2. Multilingual Indic Models  │ IndicTrans2 (AI4Bharat, 2023) handles raw   │
│    (IndicTrans2, NLLB-200)    │ text but still over-translates colloquial   │
│                               │ tech/business loanwords in formal modes.    │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 3. Frontier LLM In-Context    │ LLMs lack deterministic explainability.     │
│    Learning (GPT-4o, Claude)  │ LAPS provides audit-ready token-level       │
│                               │ mathematical justifications and thresholds. │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 6. Real-World Failure Scenarios

* **Scenario 1 — Synthetic / Curated Corpora (The Demo Trap):**  
  * *Input:* `"hi! Kisi ke pass viva calender ki link h kya?"`  
  * *Behavior:* Exact dictionary hits, matched regex templates, pristine output. Gives a false impression of system maturity.
* **Scenario 2 — Noisy Real-World Transliteration:**  
  * *Input:* `"bhai cl ka meeting postpone hogya h kya update h"`  
  * *Behavior:* `cl` (tomorrow/call) and `hogya` (happened) trigger LID misclassification; fuzzy matching confuses `cl` with `call`, distorting sentence semantics.
* **Scenario 3 — Polysemous Loanword Ambiguity:**  
  * *Input:* `"Unhone naye bank ke paas project shuru kiya"` (*bank* = river bank).  
  * *Behavior:* Static domain anchor matches `"finance"`, artificially boosting $DS$, preventing appropriate context-aware translation.
* **Scenario 4 — Dialect & Code-Mixing Switches (Tanglish / Benglish):**  
  * *Input:* `"Intha code review mudinjatha? Link send pannunga."` (Tanglish)  
  * *Behavior:* Hindi marker sets fail completely; Tanglish words (`mudinjatha`, `pannunga`) default to English candidates, corrupting the entire constraint map.

---

## 7. Reviewer Blind Spots & Methodological Gaps

When submitted to an NLP conference, peer reviewers will scrutinize the following blind spots:

1. **"Where is the Human Evaluation?"**  
   Automatic metrics (BLEU, chrF) correlate poorly with human judgment on code-mixed text. A paper without a double-blind, 3-rater human evaluation measuring **Fluency**, **Adequacy**, and **Loanword Naturalness** will be rejected.
2. **"Why is the baseline performance identical?"**  
   Reviewers will immediately catch identical metric columns in the benchmark table and flag the evaluation as fabricated or defective.
3. **"Is the Knowledge Graph actually a Graph?"**  
   Reviewers will check whether graph algorithms (PageRank, node2vec, shortest path) are utilized. If the SQLite DB is used only for key-value attribute lookups, calling it a "Knowledge Graph" will be flagged as terminology inflation.
4. **"What is the ablation contribution of each of the 6 features?"**  
   Without an ablation table isolating the marginal impact of $BF$, $DS$, $SI$, $CC$, $AF$, and $TB$, the formula appears ad-hoc.

---

## 8. Top 10 Ways a Peer-Review Committee Will Reject This Paper

```
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                  TOP 10 PEER-REVIEW REJECTION ARGUMENTS                   │
   └───────────────────────────────────────────────────────────────────────────┘
```

| Rank | Fatal Rejection Reason | Likelihood | Impact |
| :---: | :--- | :---: | :---: |
| **1** | **Circular / Simulated Evaluation:** Benchmark numbers are generated by a mock script with zero real model evaluation. | **100%** | **Desk Reject** |
| **2** | **Missing Standard Baselines:** No comparison against IndicTrans2, NLLB-200, GPT-4o, or few-shot ICL prompting. | **95%** | **Strong Reject** |
| **3** | **Unvalidated Toy Dataset:** Evaluation conducted on only 100 self-authored sentences without inter-annotator agreement ($\kappa$). | **90%** | **Strong Reject** |
| **4** | **Dictionary-Disguised Features:** Core features rely on arbitrary binary lookups rather than empirical corpus statistics. | **85%** | **Reject** |
| **5** | **Over-Engineered Architecture:** Standard few-shot LLM prompts achieve comparable preservation without pre-generation pipeline latency. | **80%** | **Reject** |
| **6** | **Brittle Token-Level LID:** Rule-based LID collapses on noisy, informal, or non-standard Romanized Indic text. | **80%** | **Reject** |
| **7** | **Terminology Inflation:** Key-value SQL tables labeled as a "Knowledge Graph" without true graph-theoretic operations. | **75%** | **Borderline Reject** |
| **8** | **Lack of Human Evaluation:** No 3-rater double-blind evaluation of Naturalness, Fluency, and Preservation Accuracy. | **85%** | **Reject** |
| **9** | **Overfitted SLSQP Optimization:** 6 continuous weights fitted to 100 sentences without cross-validation test splits. | **70%** | **Borderline Reject** |
| **10** | **Lack of Multi-Language Generalization:** Claims support for Tanglish/Benglish but hardcodes Hindi grammatical markers. | **75%** | **Borderline Reject** |

---

## 9. The Strongest Argument Against Publication
*(Written from the perspective of an ACL/EMNLP Area Chair)*

> **Meta-Review:**  
> *"This paper introduces LAPS, a pre-generation scoring framework for loanword preservation in code-mixed text rewriting. While the research question is well-motivated, the submission suffers from fatal experimental and methodological flaws. The empirical evaluation relies on a mock rule-based translation script where baseline comparisons exhibit identical metric scores, indicating a lack of genuine model evaluation. Furthermore, the system lacks comparisons against standard open-source translation models (e.g., IndicTrans2) and modern in-context learning baselines. The feature extraction layer relies heavily on hardcoded word lists, and the 100-sentence test set lacks inter-annotator agreement validation. In its current form, the paper does not meet the standards for publication in an ACL/EMNLP venue. Rejection is strongly recommended."*

---

## 10. Scientific Recovery Roadmap: How to Make LAPS Genuinely Publishable

To transform LAPS into an accepted paper at a top-tier venue, execute the following 5-phase scientific recovery plan:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       5-PHASE RESEARCH RECOVERY ROADMAP                     │
└─────────────────────────────────────────────────────────────────────────────┘

 Phase 1: Real Model Integration & Baseline Harness (Weeks 1-2)
    ├── Connect real NMT engines: IndicTrans2, NLLB-200, mBART-50
    ├── Connect real LLM backends: Llama-3.3-70B, GPT-4o-mini, Qwen-2.5
    └── Implement 3 distinct prompt baselines (Zero-Shot, Few-Shot, Chain-of-Thought)

 Phase 2: Neural Character/Subword Language ID (Weeks 3-4)
    ├── Replace regex dictionary with fine-tuned Romanized Indic RoBERTa / fastText
    └── Handle out-of-vocabulary and phonetic spelling variations dynamically

 Phase 3: Scaled Multi-Annotator Corpus & Inter-Rater Agreement (Weeks 5-6)
    ├── Curate 1,000+ code-mixed sentences across 4 domains (Tech, Academics, Business, Social)
    ├── Collect independent annotations from 3 bilingual native speakers
    └── Compute and report Fleiss' Kappa (κ ≥ 0.75) for preservation ground truth

 Phase 4: Rigorous Feature Reformulation & Dynamic Statistics (Weeks 7-8)
    ├── Replace static dictionary lists with dynamic Zipf frequencies from IndicCorp/DoCo
    ├── Derive Semantic Drift (SI) via multilingual contextual embeddings
    └── Perform exhaustive ablation study: LAPS vs (LAPS - BF) vs (LAPS - DS), etc.

 Phase 5: Comprehensive Human Evaluation & Statistical Significance (Weeks 9-10)
    ├── Conduct double-blind human evaluation (N=100 samples, 3 raters) on:
    │   • Naturalness (1-5 Likert scale)
    │   • Fluency (1-5 Likert scale)
    │   • Over-Translation Error Rate (OTER)
    └── Report paired bootstrap resampling p-values (p < 0.01) against all baselines
```

---

### **Summary Checklist for Paper Submission**

- [ ] Real translation models executed in benchmark harness (IndicTrans2, NLLB-200, GPT-4o).
- [ ] Direct comparison against standard Few-Shot In-Context Prompting baseline.
- [ ] Evaluation dataset expanded to $N \ge 1,000$ sentences with Fleiss' $\kappa \ge 0.75$.
- [ ] Subword Neural LID replacing static regex wordlists.
- [ ] Full feature ablation table ($-\text{BF}, -\text{DS}, -\text{SI}, -\text{CC}, -\text{AF}, -\text{TB}$).
- [ ] Double-blind human evaluation with statistical significance testing ($p < 0.01$).
- [ ] Transparent error analysis section detailing failure modes and polysemy limits.
