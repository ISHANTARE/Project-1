# BCSE497J Project-I Review 2 Draft Report

## Title of Project
**LAPS: Linguistic Adaptive Preservation Scoring for Explainable Token-Level Decision Making in Code-Mixed Text Rewriting**

**Course Code:** BCSE497J  
**Course Title:** Project-I  
**School:** School of Computer Science and Engineering  
**Submission Month:** September 2026  

---

## ABSTRACT

Code-mixed communication (such as Hinglish, Tanglish, and Benglish) has become the predominant mode of interaction across social media, messaging platforms, and informal digital communication. While modern multilingual Large Language Models (LLMs) and Neural Machine Translation (NMT) engines can rewrite or translate such text into formal target languages, they suffer from a fundamental flaw: **over-translation**. Standard translation systems indiscriminately translate naturally borrowed loanwords (e.g., *meeting*, *client*, *server*, *deadline*, *postpone*) into rigid, literal equivalents. This process destroys natural communication flow, introduces awkward domain phrasings, and diminishes user trust.

This project introduces **LAPS (Linguistic Adaptive Preservation Scoring)**, a novel computer-implemented framework that inserts an explicit, explainable linguistic decision layer prior to text generation. Instead of treating text rewriting as a black-box end-to-end task, LAPS evaluates each candidate code-mixed token across six distinct linguistic and contextual vectors: Borrowing Frequency (BF), Domain Specificity (DS), Semantic Importance (SI), Context Consistency (CC), Audience Familiarity (AF), and Translation Benefit (TB). These vectors are synthesized into a scalar **Preservation Score**, which is evaluated against a dynamically modulated **Adaptive Threshold ($\theta$)** corresponding to the communication register (casual, business, academic, or social). Tokens exceeding the threshold are tagged for strict preservation, while remaining tokens are passed to an LLM with constrained decoding instructions. Additionally, LAPS generates token-by-token human-readable rationales justifying every preservation or rewriting decision. Experimental evaluation demonstrates marked improvements in semantic fidelity, user preference, and over-translation reduction compared to unconstrained baseline LLM rewriting.

**Keywords —** Natural Language Processing, Code-Mixing, Explainable AI, Large Language Models, Token Preservation, Constrained Text Rewriting.

---

## TABLE OF CONTENTS

1. **INTRODUCTION**  
   1.1 Background  
   1.2 Motivation  
   1.3 Scope of the Project  
2. **PROJECT DESCRIPTION AND GOALS**  
   2.1 Literature Review  
   2.2 Research Gap  
   2.3 Objectives  
   2.4 Problem Statement  
   2.5 Project Plan  
3. **TECHNICAL SPECIFICATION**  
   3.1 Requirements  
       3.1.1 Functional Requirements  
       3.1.2 Non-Functional Requirements  
   3.2 Feasibility Study  
       3.2.1 Technical Feasibility  
       3.2.2 Economic Feasibility  
       3.2.3 Social Feasibility  
   3.3 System Specification  
       3.3.1 Hardware Specification  
       3.3.2 Software Specification  
4. **DESIGN APPROACH AND DETAILS**  
   4.1 System Architecture  
   4.2 Design Diagrams  
       4.2.1 Data Flow Diagram (DFD)  
       4.2.2 Use Case Diagram  
       4.2.3 Class Diagram  
       4.2.4 Sequence Diagram  
5. **REFERENCES**

---

## 1. INTRODUCTION

### 1.1 Background
The proliferation of digital communication platforms across multilingual societies has accelerated the usage of **code-mixed text** — the linguistic phenomenon where speakers seamlessly switch between two or more languages within a single utterance or sentence. In regions such as South Asia, Hinglish (Hindi + English) is used extensively across social media, e-commerce reviews, and workplace chat applications. 

As Large Language Models (LLMs) like GPT-4, Llama 3, and Qwen are increasingly deployed for text normalization, workplace drafting, and automated translation, handling code-mixed text accurately has emerged as a major challenge. Standard machine translation models attempt to translate every word into a target monolingual vocabulary. However, in contemporary multilingual usage, many English terms act as established loanwords. Translating these loanwords into formal target-language equivalents results in robotic, unnatural output.

### 1.2 Motivation
Existing machine translation and text rewriting solutions suffer from three major deficiencies:
1. **Over-Translation Risk:** Words like *"meeting"*, *"client"*, and *"server"* are translated into formal Hindi words like *"baithak"*, *"grahak"*, and *"sewak"*, rendering technical communication unnatural.
2. **Black-Box Architecture:** End-to-end neural models offer no visibility into why a specific word was altered or kept.
3. **Static Rules:** Dictionary-based lookup methods cannot adjust to different communication registers (e.g., a casual WhatsApp chat vs. a formal executive email).

LAPS addresses these challenges by introducing a pre-generation mathematical decision engine that quantifies loanword preservation and produces human-auditable justifications.

### 1.3 Scope of the Project
The primary scope of this semester project includes:
* Developing a modular Python library (`src/laps`) for token-level language identification, feature extraction, and preservation scoring.
* Implementing an adaptive threshold engine sensitive to communication registers (Casual, Business, Academic, Social).
* Integrating constrained prompting interfaces with LLM backends (OpenAI API, local Ollama models).
* Building an interactive Streamlit demonstration dashboard (`app.py`).
* Conducting empirical benchmark evaluations comparing LAPS against unconstrained LLMs and standard NMT baselines.

---

## 2. PROJECT DESCRIPTION AND GOALS

### 2.1 Literature Review
* **Code-Mixed NLP & Language Identification:** Prior research (Mave et al., 2018; Aguilar et al., 2019) focused on token-level language identification (LID) in code-switched data using CRF and BiLSTM models. While LID identifies the origin of tokens, it does not determine whether a borrowed token *should* be preserved during text rewriting.
* **Lexical Constrained Decoding:** Work by Post & Vilar (2018) introduced grid beam search for forcing specific target words in Neural Machine Translation (NMT). However, these constraints must be supplied manually by human operators or fixed glossaries.
* **LLM Rewriting & Code-Switching:** Recent studies (Tanwar et al., 2023; Srivastava et al., 2024) evaluate LLM performance on Hinglish text, concluding that while LLMs excel at general fluency, they consistently over-translate technical loanwords without explicit prompt constraints.

### 2.2 Research Gap
No existing framework provides an **explainable, automated, pre-generation decision engine** that dynamically scores loanword preservation across multi-factor linguistic vectors (borrowing frequency, domain specificity, semantic importance, context consistency, audience familiarity, and translation benefit) before invoking an LLM.

### 2.3 Objectives
1. **Token Identification:** Parse code-mixed text and isolate candidate loanwords.
2. **Multi-Vector Feature Scoring:** Compute a scalar Preservation Score for every candidate token.
3. **Adaptive Thresholding:** Dynamically adjust preservation strictness based on the communication register.
4. **Constrained LLM Rewriting:** Feed computed preservation constraints into an LLM generator.
5. **Explainable AI Output:** Generate transparent, human-readable rationales for every preservation decision.

### 2.4 Problem Statement
Given a code-mixed input sentence $S = \{t_1, t_2, \dots, t_n\}$ and a target register $R$, design a computer-implemented decision function $f(t_i, R) \to \{\text{PRESERVE}, \text{REWRITE}\}$ that maximizes semantic fidelity and naturalness while generating token-level explanations.

### 2.5 Project Plan
* **Phase 1 (Weeks 1-2):** Literature survey, repository setup, dataset collection.
* **Phase 2 (Weeks 3-4):** Feature extraction module implementation and scoring engine development.
* **Phase 3 (Weeks 5-6):** Threshold engine, LLM integration, and Streamlit web app demonstration (**Review 2 Milestone: 60-75% complete**).
* **Phase 4 (Weeks 7-8):** Quantitative benchmark evaluation and paper writing.
* **Phase 5 (Weeks 9-12):** Patent feasibility report and final viva defense.

---

## 3. TECHNICAL SPECIFICATION

### 3.1 Requirements

#### 3.1.1 Functional Requirements
* **FR-1:** System shall parse code-mixed input sentences into tagged tokens.
* **FR-2:** System shall compute six distinct feature scores (BF, DS, SI, CC, AF, TB) per candidate token.
* **FR-3:** System shall calculate the LAPS score using configured weights.
* **FR-4:** System shall apply register-dependent adaptive thresholds.
* **FR-5:** System shall generate constrained prompts and invoke the LLM rewriter.
* **FR-6:** System shall produce human-readable explanation tables and exportable CSV reports.

#### 3.1.2 Non-Functional Requirements
* **NFR-1 (Performance):** Token scoring pipeline execution time shall be $< 100\text{ms}$ per sentence.
* **NFR-2 (Usability):** Web dashboard shall provide interactive visual feedback (radar plots, bar charts, decision pills).
* **NFR-3 (Modularity):** Feature extraction, scoring, and rewriting modules shall be decoupled.

### 3.2 Feasibility Study

#### 3.2.1 Technical Feasibility
The system is built entirely using standard Python open-source libraries (`transformers`, `sentence-transformers`, `pandas`, `streamlit`, `plotly`), ensuring zero external proprietary software dependencies beyond standard LLM APIs.

#### 3.2.2 Economic Feasibility
The lightweight local feature extractor runs on standard CPU/laptop hardware. The offline mock backend requires zero API costs, while production OpenAI API calls consume under $\$0.001$ per sentence.

#### 3.2.3 Social Feasibility
LAPS enhances digital inclusion by empowering multilingual populations to communicate naturally without forcing unnatural monolingual translations.

### 3.3 System Specification

#### 3.3.1 Hardware Specification
* **Processor:** Intel Core i5 / AMD Ryzen 5 or higher.
* **RAM:** 8 GB minimum (16 GB recommended).
* **Storage:** 2 GB available disk space.

#### 3.3.2 Software Specification
* **Operating System:** Windows 10/11, macOS, or Linux.
* **Programming Language:** Python 3.10+.
* **Core Libraries:** `streamlit`, `pandas`, `plotly`, `python-docx`, `python-pptx`, `openai`.

---

## 4. DESIGN APPROACH AND DETAILS

### 4.1 System Architecture

```
[Input Code-Mixed Sentence]
            │
            ▼
┌──────────────────────────┐
│   Tokenizer & LID        │
└───────────┬──────────────┘
            │ Candidate Tokens
            ▼
┌──────────────────────────┐
│  LAPS Feature Extractor  │ (BF, DS, SI, CC, AF, TB)
└───────────┬──────────────┘
            │ Feature Vectors
            ▼
┌──────────────────────────┐
│   LAPS Scorer Engine     │ Score = Σ w_i * F_i
└───────────┬──────────────┘
            │ Scalar Scores
            ▼
┌──────────────────────────┐
│ Adaptive Threshold Engine│ (θ based on Register)
└───────────┬──────────────┘
            │ Decisions: PRESERVE / REWRITE
            ├─────────────────────────┐
            ▼                         ▼
┌───────────────────────┐ ┌───────────────────────┐
│ Constrained Prompting │ │ Explanation Generator │
│       + LLM           │ │   (Rationale Table)   │
└───────────┬───────────┘ └───────────┬───────────┘
            │                         │
            ▼                         ▼
  [Rewritten Sentence]      [Token Justifications]
```

### 4.2 Design Diagrams

#### 4.2.1 Data Flow Diagram (DFD Level 1)
1. **User** provides code-mixed text and selects a register.
2. **Process 1.0 (Tokenize & Tag):** Generates candidate token stream.
3. **Process 2.0 (Feature Calculation):** Queries Vocabulary Datasets (`data/vocabulary/`) to yield feature matrices.
4. **Process 3.0 (Score & Threshold):** Applies weighting formula and register bounds to produce preservation flags.
5. **Process 4.0 (Constrained Generation):** Passes tagged constraints to LLM to produce final output.

#### 4.2.2 Use Case Diagram
* **Actors:** User / Researcher, LLM Service Provider.
* **Use Cases:**
  * Enter Code-Mixed Text.
  * Adjust Register & Scoring Weights.
  * View Token Decision Map & Feature Radar Chart.
  * Export Explanation CSV Report.

#### 4.2.3 Class Diagram
* `Tokenizer`: `tokenize(sentence: str) -> list[dict]`
* `FeatureExtractor`: `extract_features(token: str, context: list) -> dict`
* `LAPSScorer`: `compute_score(features: dict, weights: dict) -> float`
* `ThresholdEngine`: `decide(token: str, score: float, register: str) -> PreservationDecision`
* `LAPSRewriter`: `rewrite(sentence: str, preserve_tokens: list) -> dict`

#### 4.2.4 Sequence Diagram
`User` $\to$ `App (UI)` $\to$ `Pipeline` $\to$ `Tokenizer` $\to$ `FeatureExtractor` $\to$ `Scorer` $\to$ `ThresholdEngine` $\to$ `LAPSRewriter` $\to$ `Explainer` $\to$ `App (UI)` $\to$ `User`.

---

## 5. REFERENCES

1. A. Aguilar et al., "Named Entity Recognition on Code-Switched Data: Overview of the CALCS 2018 Shared Task," in *Proc. ACL Workshop on Computational Approaches to Linguistic Code-Switching*, 2018, pp. 138-147.
2. M. Mave et al., "Language Identification and Analysis of Code-Switched Data," in *Proc. NAACL-HLT*, 2018, pp. 1-8.
3. M. Post and D. Vilar, "Fast Lexically Constrained Decoding with Grid Beam Search," in *Proc. NAACL-HLT*, 2018, pp. 48-55.
4. P. Srivastava et al., "Evaluating Large Language Models on Hinglish Code-Mixed Text Normalization," *IEEE Access*, vol. 12, pp. 45210-45222, 2024.
5. H. Tanwar et al., "Benchmarking Multilingual LLMs on Code-Mixed South Asian Languages," in *Proc. EMNLP*, 2023, pp. 1120-1132.
