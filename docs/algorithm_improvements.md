# LAPS Algorithmic & Functional Improvements

This document outlines the mathematical foundation of **LAPS (Linguistic Adaptive Preservation Scoring)** and reviews the programmatic enhancements needed to turn the initial heuristic framework into a production-grade NLP algorithm.

---

## 📐 1. Algorithmic Foundation

The base score for a candidate code-mixed token $t$ in context $C$ is defined by:

$$\text{Score}(t) = w_1 \text{BF}(t) + w_2 \text{DS}(t) + w_3 \text{SI}(t, C) + w_4 \text{CC}(t, C) + w_5 \text{AF}(t) - w_6 \text{TB}(t)$$

Where:
* **$\text{BF}(t)$ (Borrowing Frequency):** Empirical probability of word borrowing in general code-mixed corpora.
* **$\text{DS}(t)$ (Domain Specificity):** Multi-class category match vector mapped to technical domain weights.
* **$\text{SI}(t, C)$ (Semantic Importance):** Semantic change or information loss if translated.
* **$\text{CC}(t, C)$ (Context Consistency):** Probability of surrounding words sharing similar language borrows.
* **$\text{AF}(t)$ (Audience Familiarity):** Lexical frequency ranking in standard spoken language dictionaries.
* **$\text{TB}(t)$ (Translation Benefit):** Syntactic necessity of translating $t$ to maintain grammatical coherence in the target language.

---

## 🚀 2. Proposed Improvements

To make LAPS robust and peer-reviewed publication ready, the following updates are planned:

### **A. Moving from Static Lexicons to Sentence-Transformers**
* **Current Issue:** Checking semantic importance and domain specificity using static lookup files leads to out-of-vocabulary errors.
* **Solution:** 
  * Use a small, lightweight sentence-transformer model (e.g., `MiniLM-L6-v2`) to compute domain alignment.
  * Define domain anchor vectors (e.g., Software, Finance, Medical) and compute cosine similarity:
    $$\text{DS}(t) = \max_{d \in D} \cos(\mathbf{e}_t, \mathbf{e}_d)$$
  * This ensures semantic generalizations for words not in the hardcoded lexicons.

### **B. Sub-word/Morpheme-Level Language Identification (LID)**
* **Current Issue:** Standard language identifiers classify words on a whole-word basis, which struggles with compounded code-mixed terms (e.g., *"postpone-ment"* or Hinglish pluralizations like *"meetingon"*).
* **Solution:** Implement a sub-word character LSTM or token classifier (e.g., fine-tuned `mT5` or custom character $n$-gram models) to tag language boundaries at the character level.

### **C. Automated Weight Optimization ($\mathbf{w}$)**
* **Current Issue:** Initial weights are chosen based on intuition, which makes objective evaluation difficult to defend in peer reviews.
* **Solution:**
  * Curate a tuning dataset of 100 benchmark sentences with manual preservation labels.
  * Frame weight learning as a constrained optimization problem.
  * Use **Bayesian Optimization** or **Grid Search** to find the weight vector $\mathbf{w}$ that maximizes the F1-score of preservation decisions against human annotations:
    $$\mathbf{w}^* = \arg\max_{\mathbf{w}} \text{F1}(\text{LAPS}(\mathbf{w}), \mathbf{Y}_{\text{human}})$$

### **D. Soft Threshold Sigmoid Mapping**
* **Current Issue:** Hard thresholds ($Score > \theta$) cause binary boundary errors, where a small weight change radically flips the rewrite structure.
* **Solution:** Map the score output to a probability distribution using a Sigmoid function:
  $$P(\text{Preserve}) = \sigma(\lambda(\text{Score}(t) - \theta)) = \frac{1}{1 + e^{-\lambda(\text{Score}(t) - \theta)}}$$
  This probability can feed a stochastic prompt structure or guide sampling-based decoding in LLMs.
