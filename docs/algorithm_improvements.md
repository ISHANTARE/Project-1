# LAPS Algorithmic and Functional Improvements

This document outlines the mathematical foundation of **LAPS (Linguistic Adaptive Preservation Scoring)** and details the algorithmic upgrades implemented across the codebase.

---

## 1. Algorithmic Foundation

The Preservation Score for a candidate code-mixed token $t$ in sentence context $C$ is computed via:

$$\text{LAPSScore}(t) = w_1 \text{BF}(t) + w_2 \text{DS}(t) + w_3 \text{SI}(t) + w_4 \text{CC}(t, C) + w_5 \text{AF}(t) - w_6 \text{TB}(t)$$

Where:
* **$\text{BF}(t)$ (Borrowing Frequency):** Queried from the 2,907-node Knowledge Graph or `wordfreq` Zipf frequency ratios.
* **$\text{DS}(t)$ (Domain Specificity):** Computed via `sentence-transformers` (`all-MiniLM-L6-v2`) zero-shot vector cosine similarity against Tech, Business, Academic, Medical, and Finance domain anchors.
* **$\text{SI}(t)$ (Semantic Importance):** Quantifies semantic drift when translated into target language.
* **$\text{CC}(t, C)$ (Context Consistency):** Sub-graph co-occurrence density of token $t$ within context window $C$.
* **$\text{AF}(t)$ (Audience Familiarity):** Global conversational recognition score.
* **$\text{TB}(t)$ (Translation Benefit):** Syntactic benefit of translating $t$ to preserve target language grammar.

---

## 2. Implemented Algorithmic Upgrades

### A. Neural Vector Embeddings and Fuzzy Typo Alignment [IMPLEMENTED]
* **Method:** Replaced static word lists with `SentenceTransformer("all-MiniLM-L6-v2")` embeddings. Typos such as `calender` or `scedule` automatically map to canonical forms via `difflib` edit-distance matching.

### B. Sentence-Level Code-Mixing Index (CMI) Detection [IMPLEMENTED]
* **Method:** Built `cmi_detector.py` to calculate CMI (Gamback and Das, 2014):
  $$\text{CMI} = 100 \cdot \left[1 - \frac{\max(N_{\text{lang}})}{N_{\text{total}}}\right]$$
  Monolingual English sentences ($\text{CMI} = 0.0\%$) are cleanly detected and retained without unnecessary code-mixed rewriting.

### C. Soft Sigmoid Probability Decision Boundaries [IMPLEMENTED]
* **Method:** Replaced hard binary cutoffs ($\text{Score} \ge \theta$) with smooth Sigmoid logistic probabilities:
  $$P(\text{Preserve} \mid t) = \frac{1}{1 + e^{-\lambda (\text{LAPSScore}(t) - \theta_R)}}$$
  Where $\lambda = 10.0$ governs transition sharpness. Three distinct decision zones are identified:
  * $P \ge 0.70 \rightarrow$ PRESERVE (high confidence retention)
  * $0.50 \le P < 0.70 \rightarrow$ UNCERTAIN (borderline candidate, preserved with advisory flag)
  * $P < 0.50 \rightarrow$ REWRITE (candidate suitable for natural translation)

### D. Graph-Enriched XML Prompt Engineering [IMPLEMENTED]
* **Method:** Structured XML constraint injection with anti-hallucination domain rationales and edge naturalness penalties ($<token\ name="..."\ naturalness\_penalty="...">$), strictly guiding downstream LLMs during sentence rewriting.

### E. Dynamic Empirical Borrowing Index from LinCE & PHINC [IMPLEMENTED]
* **Method:** Replaced static whitelists with dynamic empirical borrowing propensities derived directly from **16,024 code-mixed sentences** (LinCE and PHINC corpora), indexing **9,879 unique English loanwords** with occurrence counts, document frequencies, and continuous borrowing propensity scores:
  $$\text{BPI}(t) = \min\left(1.0, \frac{\log_{10}(c(t) + 1)}{\log_{10}(1000)} \times \left(0.5 + 0.5 \cdot \frac{\text{df}(t)}{N_{\text{total}}}\right)\right)$$

### F. Automated Weight Vector Optimization ($\mathbf{w}^*$) [IMPLEMENTED]
* **Method:** Grid search and SLSQP optimization over 6-dimensional simplex constraints ($\sum w_j = 1, w_j \ge 0$) targeting maximum Token F1 and minimum Cross-Entropy loss over gold standard datasets.
