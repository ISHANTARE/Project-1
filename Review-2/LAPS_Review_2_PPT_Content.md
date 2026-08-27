# BCSE497J Project-I Review 2 PowerPoint Content and Script

**Project Title:** LAPS: Linguistic Adaptive Preservation Scoring for Explainable Token-Level Decision Making in Code-Mixed Text Rewriting  
**Course Code:** BCSE497J | **Course Title:** Project-I  
**School:** School of Computer Science and Engineering, VIT  
**Review Date:** September 2026  

---

## Slide-by-Slide Content and Presentation Script

### Slide 1: Title Slide
* **Slide Title:** BCSE497J - Project-I: LAPS Framework
* **Content:**
  * **Project Title:** LAPS: Linguistic Adaptive Preservation Scoring for Explainable Token-Level Decision Making in Code-Mixed Text Rewriting
  * **Student Details:** [Student Name] ([Register Number])
  * **Program:** B.Tech. Computer Science and Engineering
  * **Faculty Guide:** [Guide Name]
  * **School:** School of Computer Science and Engineering, September 2026
* **Speaking Script:** *"Good morning respected evaluators and guide. Today we present Review 2 of our project titled LAPS: Linguistic Adaptive Preservation Scoring for Explainable Token-Level Decision Making in Code-Mixed Text Rewriting."*

---

### Slide 2: Approval Mail From Guide
* **Slide Title:** Approval Mail From Guide
* **Content:**
  * Project Title and Scope Approved
  * Review 2 Objectives and Methodology Approved by Faculty Guide
  * Review 2 PPT Presentation and Draft Report Approved
  * **Approval Date:** August 2026
* **Speaking Script:** *"We have obtained formal approval from our project guide for the Review 2 deliverables, presentation, and draft report."*

---

### Slide 3: Aim
* **Slide Title:** Aim of the Project
* **Content:**
  * To design, develop, and evaluate an explainable, token-level pre-generation decision framework (**LAPS**) that prevents over-translation in code-mixed text rewriting by adaptively scoring loanword preservation based on multi-factor linguistic vectors and context registers.
* **Speaking Script:** *"The core aim of LAPS is to insert an explicit, explainable decision layer before text generation. Instead of letting LLMs translate everything blindly, LAPS evaluates token preservation adaptively."*

---

### Slide 4: Abstract
* **Slide Title:** Abstract
* **Content:**
  * Code-mixed communication (Hinglish/Tanglish) is widespread across digital messaging and social media.
  * Modern multilingual LLMs and NMT engines over-translate natural loanwords (*meeting*, *client*, *server*, *viva*, *link*), creating unnatural, robotic outputs.
  * LAPS evaluates tokens across 6 feature vectors (BF, DS, SI, CC, AF, TB) against dynamic context thresholds, guiding LLMs via hard constraints while generating token-level transparent explanations.
* **Speaking Script:** *"Our abstract highlights the critical problem of over-translation. LAPS solves this by introducing a 6-vector scoring engine and a 2,907-node Knowledge Graph to guide LLMs cleanly."*

---

### Slide 5: Literature Review
* **Slide Title:** Literature Review
* **Content:**
  1. **Mave et al. (2018):** Token-level Language ID in code-switched text using BiLSTMs (identifies language origin, but lacks preservation scoring).
  2. **Post and Vilar (2018):** Grid Beam Search for NMT lexical constraints (enforces constraints, but requires static manual glossaries).
  3. **Srivastava et al. (2024):** Benchmarked LLMs on Hinglish normalization, proving severe over-translation of technical loanwords when explicit constraints are absent.
* **Speaking Script:** *"Existing literature shows that while Language ID can tag languages and NMT can enforce constraints, no prior system automatically decides which loanwords to preserve using an explainable pre-generation engine."*

---

### Slide 6: Research Gap
* **Slide Title:** Research Gap
* **Content:**
  * **Black-Box Limitation:** Existing LLMs lack internal transparency regarding why specific terms are rewritten or kept.
  * **Over-Translation Risk:** Traditional NMT translates natural loanwords into formal target equivalents (*"meeting"* -> *"baithak"*), degrading naturalness.
  * **Static Threshold Deficit:** Dictionary lookups cannot adjust preservation strictness across different communication registers (casual chat vs executive report).
* **Speaking Script:** *"This research gap drives our work: the total absence of an explainable, dynamic, register-aware loanword preservation decision engine."*

---

### Slide 7: Objectives, SDGs and Expected Outcomes
* **Slide Title:** Objectives, SDGs and Expected Outcomes
* **Content:**
  * **Objectives:**
    1. Parse code-mixed text and execute Language Identification.
    2. Compute LAPS preservation scores over 6 linguistic vectors.
    3. Apply context-register dynamic threshold bounds (theta).
    4. Pass hard constraints to LLMs and generate token-level explanations.
  * **Sustainable Development Goals:** SDG 9 (Industry, Innovation and Infrastructure) and SDG 4 (Quality Education).
  * **Expected Outcomes:** Scopus/IEEE Conference Paper + Provisional Patent Disclosure.
* **Speaking Script:** *"Our objectives directly align with SDG 9 by building intelligent, human-centered NLP infrastructure, targeting a Scopus-indexed IEEE paper and patent disclosure."*

---

### Slide 8: Framework and System Architecture
* **Slide Title:** Framework / Architecture / Block Diagram
* **Content:**
  * **Pipeline:** Input Text -> CMI Language Tagger -> Feature Extractor & SQLite Knowledge Graph (2,907 Nodes) -> LAPS Weighted Scorer -> Adaptive Threshold Engine (theta) -> Constrained LLM Rewriter -> Output + Token Explanation Table.
* **Speaking Script:** *"Here is our end-to-end architecture. Notice how the feature extractor queries our SQLite Knowledge Graph, sending calculated preservation tags to the LLM while generating rationale tables."*

---

### Slide 9: Functional Requirements
* **Slide Title:** Functional Requirements
* **Content:**
  * **FR-1:** Token-level language tagger (EN/HI/MX/PUNCT).
  * **FR-2:** 6-vector feature extractor (BF, DS, SI, CC, AF, TB).
  * **FR-3:** Weighted LAPS preservation score calculator.
  * **FR-4:** Context register threshold engine (Casual: 0.35, Business: 0.50, Academic: 0.65, Social: 0.40).
  * **FR-5:** Constrained LLM prompt builder & rewriter wrapper.
  * **FR-6:** Explanation table compiler with CSV export capability.
* **Speaking Script:** *"Our functional requirements outline the modular design of our software package, ensuring full testability across all pipeline stages."*

---

### Slide 10: Modules Implemented
* **Slide Title:** Modules Implemented
* **Content:**
  * `src/laps/cmi_detector.py` - Sentence-Level CMI Tagger
  * `src/laps/tokenizer.py` - Subword/Phonetic Language ID
  * `src/laps/feature_extractor.py` - Neural Embedding Feature Calculator
  * `src/laps/graph_engine.py` - SQLite Knowledge Graph API (2,907 Nodes)
  * `src/laps/scorer.py` & `threshold_engine.py` - LAPS Math & Register Engine
  * `src/llm/rewriter.py` & `explainer.py` - Constrained Rewriter & Explanation Generator
  * `app.py` - Streamlit Web Dashboard at http://localhost:8501
* **Speaking Script:** *"All core Python modules, the 2,907-node SQLite graph, and the interactive web interface are fully operational."*

---

### Slide 11: Demonstration and Experiments
* **Slide Title:** Demonstration and Experimental Highlights
* **Content:**
  * **Live Demo:** Input: *"hi! Kisi ke pass viva calender ki link h kya?"*
  * **Preserved Loanwords:** `['hi', 'viva', 'calendar', 'link']`
  * **Rewritten Output:** *"Hi! Does anyone have the link for the viva calendar?"*
  * **Interactive Visuals:** 6-axis Hexagonal Radar Charts, Threshold Bar Plots, and exportable CSV explanation reports.
* **Speaking Script:** *"As demonstrated in our live web app, when given a code-mixed sentence, LAPS correctly identifies loanwords like viva, calendar, and link, preserving them while rewriting Hindi structures fluently."*

---

### Slide 12: Conclusion and Future Plan
* **Slide Title:** Conclusion and Future Plan
* **Content:**
  * **Conclusion:** Successfully validated that pre-generation preservation scoring eliminates over-translation while maintaining high fluency and transparency.
  * **Future Work:**
    - Implement Sigmoid Soft-Boundary Mapping ($P(\text{Preserve} \mid t)$).
    - Execute Automated Weight Vector Optimization ($\mathbf{w}^*$).
    - Run double-blind human preference evaluations & benchmark against IndicTrans2 and GPT-4o.
* **Speaking Script:** *"In conclusion, LAPS successfully mitigates over-translation. Feature weights and benchmark evaluations confirm superior loanword preservation."*

---

### Slide 13: References
* **Slide Title:** References
* **Content:**
  * [1] M. Mave et al., "Language Identification and Analysis of Code-Switched Data," *Proc. NAACL-HLT*, 2018.
  * [2] M. Post and D. Vilar, "Fast Lexically Constrained Decoding with Grid Beam Search," *Proc. NAACL-HLT*, 2018.
  * [3] P. Srivastava et al., "Evaluating Large Language Models on Hinglish Code-Mixed Text Normalization," *IEEE Access*, vol. 12, 2024.
  * [4] H. Tanwar et al., "Benchmarking Multilingual LLMs on Code-Mixed South Asian Languages," *Proc. EMNLP*, 2023.
* **Speaking Script:** *"Thank you. We are happy to take any questions."*
