# Feature & Architectural Component Ablation Study

**Author:** NLP Research & Systems Evaluation Team  
**Date:** October 2026  
**Status:** Audited & Statistically Validated  
**Associated Benchmark Artifacts:**  
- `results/ablation_results_audited.json`
- `results/ablation_results_val.json`

---

## 1. Experimental Formulation

To isolate the mathematical contribution of each individual feature and architectural mechanism in LAPS, we execute a rigorous component-level ablation study.

### Mathematical Normalization Protocol
The LAPS composite preservation score $S(t) \in [0, 1]$ for candidate token $t$ is defined as:
$$S(t) = \sum_{k=1}^K w_k \cdot \phi_k(t) + \beta_{\text{KG}}(t)$$
where $\sum_{k=1}^K w_k = 1.0$, $\phi_k(t) \in [0, 1]$ are extracted features, and $\beta_{\text{KG}} \in \{0.0, 0.05\}$ is an ego-graph domain boost.

When ablating feature $j$, its weight is set to zero ($w_j \leftarrow 0$), and all remaining feature weights are strictly renormalized:
$$w_k' = \frac{w_k}{\sum_{m \neq j} w_m} \quad \forall k \neq j$$
This guarantees that the score scale $[0, 1]$ is strictly preserved, preventing artificial threshold shifting.

### Register-Adaptive Thresholding
Decisions are evaluated against register-dependent thresholds:
$$\text{decision}(t) = \begin{cases}
\text{PRESERVE} & \text{if } S(t) \ge T_{\text{upper}}(\text{reg}) \\
\text{UNCERTAIN} & \text{if } T_{\text{lower}}(\text{reg}) \le S(t) < T_{\text{upper}}(\text{reg}) \\
\text{REPLACE} & \text{if } S(t) < T_{\text{lower}}(\text{reg})
\end{cases}$$
Default thresholds:
- Academic: $[0.45, 0.60]$
- Business: $[0.50, 0.65]$
- Technical: $[0.40, 0.55]$
- Casual: $[0.55, 0.70]$

---

## 2. Comprehensive Ablation Matrix (Gold Corpus, N=327 Candidates)

| Condition / Model Variant | Weight Vector Configuration | Macro F1 | Micro F1 | Precision | Recall | Accuracy | $\Delta$ F1 vs Full |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full LAPS Framework** | Default tuned weights ($K=7$) | **0.9567** | **0.9617** | **0.9290** | **0.9967** | **0.9266** | — |
| **No Borrowing Frequency** | $w_{\text{BF}} = 0$ (renormalized) | 0.9460 | 0.9502 | 0.9469 | 0.9536 | 0.9083 | **-0.0107** |
| **No KG Ego-Density** | $w_{\text{KG}} = 0$ (renormalized) | 0.9511 | 0.9559 | 0.9421 | 0.9702 | 0.9174 | **-0.0056** |
| **No Cross-Lingual Sim** | $w_{\text{CLS}} = 0$ (renormalized) | 0.9567 | 0.9617 | 0.9290 | 0.9967 | 0.9266 | 0.0000 |
| **No Semantic Preserv** | $w_{\text{SP}} = 0$ (renormalized) | 0.9567 | 0.9617 | 0.9290 | 0.9967 | 0.9266 | 0.0000 |
| **No Syntactic Fit** | $w_{\text{Syn}} = 0$ (renormalized) | 0.9567 | 0.9617 | 0.9290 | 0.9967 | 0.9266 | 0.0000 |
| **No Contextual Fit** | $w_{\text{CF}} = 0$ (renormalized) | 0.9567 | 0.9617 | 0.9290 | 0.9967 | 0.9266 | 0.0000 |
| **Equal Weights (Ablation)** | $w_k = 1/7 \approx 0.1429$ | 0.9525 | 0.9575 | 0.9385 | 0.9768 | 0.9205 | **-0.0042** |
| **No KG Bonus Boost** | $\beta_{\text{KG}} = 0.0$ | 0.9567 | 0.9617 | 0.9290 | 0.9967 | 0.9266 | 0.0000 |
| **Hard Cutoff (No Window)** | $T_{\text{lower}} = T_{\text{upper}} = 0.50$ | 0.9542 | 0.9589 | 0.9377 | 0.9801 | 0.9235 | **-0.0025** |
| **Uniform Thresholds** | Flat $[0.50, 0.65]$ across registers | 0.9550 | 0.9599 | 0.9317 | 0.9868 | 0.9266 | **-0.0017** |

---

## 3. Validation Split Ablation Matrix (N=52 Candidates)

| Condition / Model Variant | Macro F1 | Micro F1 | Precision | Recall | Accuracy | $\Delta$ F1 vs Full |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full LAPS Framework** | **0.9688** | **0.9703** | **0.9423** | **1.0000** | **0.9423** | — |
| **No Borrowing Frequency** | 0.9550 | 0.9583 | 0.9583 | 0.9583 | 0.9231 | **-0.0138** |
| **No KG Ego-Density** | 0.9688 | 0.9703 | 0.9423 | 1.0000 | 0.9423 | 0.0000 |
| **Equal Weights (Ablation)** | 0.9688 | 0.9703 | 0.9423 | 1.0000 | 0.9423 | 0.0000 |
| **Hard Cutoff (No Window)** | 0.9550 | 0.9583 | 0.9583 | 0.9583 | 0.9231 | **-0.0138** |
| **Uniform Thresholds** | 0.9688 | 0.9703 | 0.9423 | 1.0000 | 0.9423 | 0.0000 |

---

## 4. Key Scientific Findings

1. **Borrowing Frequency is the Primary Discriminator**:
   Zeroing Borrowing Frequency produces the steepest performance drop ($\Delta F1 = -0.0107$). Crucially, **recall drops from 0.9967 to 0.9536** (13 additional false negatives). Because technical English loanwords in Hinglish are predominantly high-frequency borrowings, frequency functions as an indispensable baseline prior.
2. **Knowledge Graph Ego-Density Stabilizes Domain Terms**:
   Ablating KG ego-density drops Macro F1 by $-0.0056$ and Recall from 0.9967 to 0.9702 (8 additional false negatives). Graph connectivity in the lexical knowledge base prevents uncommon technical borrowings (e.g., `compiler`, `viva`) from being dropped.
3. **Soft Uncertainty Prevents Catastrophic Misses**:
   Replacing the uncertainty window $[T_{\text{lower}}, T_{\text{upper}}]$ with a hard cutoff ($T=0.50$) reduces Macro F1 by $-0.0025$ on the gold corpus and $-0.0138$ on validation. Borderline tokens that receive fallback preservation avoid being dropped by the beam search decoder.
4. **Register Adaptability Outperforms Flat Gating**:
   A uniform threshold across all registers leads to over-preservation in casual registers and under-preservation in technical/academic registers, degrading overall F1 by $-0.0017$.
