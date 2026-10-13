# LAPS Mathematical Specification Audit & Formulation Verification

**Document:** `docs/mathematical_specification_audit.md`  
**Course:** BCSE497J — Project-I, Vellore Institute of Technology, Vellore  
**Audit Scope:** Verification of formulas, weights, boundaries, normalization, and code-versus-documentation alignment.

---

## 1. Verified Mathematical Formulation

### 1.1 Sentence-Level Code-Mixing Index (CMI)
Gambäck & Das (2014) index for sentence-level bilingual mixing:

$$\text{CMI} = 100 \cdot \left[1 - \frac{\max(N_{\text{EN}}, N_{\text{HI}})}{N_{\text{total}}}\right]$$

- **Implementation Location:** `src/laps/cmi_detector.py:82`
- **Verification:** Verified in `tests/test_laps.py::TestCMIDetector`.
- **Special Boundary:**
  - $\text{CMI} = 0.0\%$ for pure English $\implies$ `mode = "MONOLINGUAL_EN"`.
  - In `pipeline.py`, pure English utterances bypass rewriting and candidate feature extraction, setting all token decisions to `PASSTHROUGH` and avoiding fake scores.

---

### 1.2 The Six LAPS Features

For each candidate loanword token $t_i$ in context window $C$, LAPS extracts six normalized features $\in [0.0, 1.0]$:

#### Feature 1: Borrowing Frequency ($\text{BF}$)
Measures empirical adoption of English token $t_i$ in code-mixed Hindi:
1. **Primary Source:** Empirical index (`empirical_loanword_index.json`) derived from 16,024 authentic LinCE and PHINC sentences:
   $$\text{BPI}(t_i) = \sigma\left(2.2 \cdot (0.6 \log_{10}(1 + \text{count}) + 0.4 \log_{10}(1 + \text{df}) - 0.9)\right)$$
2. **Secondary Source:** SQLite Knowledge Graph `nodes.f_borrow`.
3. **Fallback:** Continuous logistic sigmoid over global English Zipf frequency:
   $$\text{BF}_{\text{Zipf}}(t_i) = \sigma(1.1 \cdot (\text{Zipf}(t_i) - 3.2))$$
   clamped to $[0.20, 0.98]$.
- **Code Location:** `src/laps/feature_extractor.py:145-168`

#### Feature 2: Domain Specificity ($\text{DS}$)
Quantifies the semantic niche of $t_i$ relative to professional domains:
$$\text{DS}(t_i) = \max_{d \in D} \cos\left(\mathbf{e}(t_i \text{ in context: } C), \mathbf{e}(\text{Centroid}_d)\right)$$
where $D = \{\text{Tech}, \text{Business}, \text{Academic}, \text{Medical}, \text{Finance}\}$, using `all-MiniLM-L6-v2` contextual vector representations.
- **Code Location:** `src/laps/feature_extractor.py:171-191`

#### Feature 3: Semantic Importance ($\text{SI}$)
Measures information content and resistance to translation drift:
$$\text{IC}(t_i) = \text{clip}\left(\frac{8.0 - \text{Zipf}(t_i)}{6.0}, 0.15, 0.98\right)$$
$$\text{SI}(t_i) = 0.60 \cdot \text{IC}(t_i) + 0.28 \cdot \text{BF}(t_i) + \Delta_{\text{morph}}$$
where $\Delta_{\text{morph}} = 0.12$ if $t_i$ matches nominal derivational affixes (`-tion`, `-ment`, `-ity`, `-ence`, `-ance`, `-ware`, `-base`, etc.).
- **Code Location:** `src/laps/feature_extractor.py:194-218`

#### Feature 4: Context Consistency ($\text{CC}$)
Synthesizes local window loanword density with Knowledge Graph co-occurrence PMI:
$$\text{CC}(t_i, C) = \frac{\text{PMI}_{\text{KG}}(t_i, C) + \text{Density}_{\text{window}}(C)}{2.0}$$
- **Code Location:** `src/laps/feature_extractor.py:220-244`

#### Feature 5: Audience Familiarity ($\text{AF}$)
Measures bilingual recognizability from continuous Zipf distribution:
$$\text{AF}(t_i) = \text{clip}\left(\frac{\text{Zipf}(t_i) - 1.5}{5.5}, 0.30, 0.98\right)$$
- **Code Location:** `src/laps/feature_extractor.py:247-259`

#### Feature 6: Translation Benefit ($\text{TB}$)
Subtractive functional penalty quantifying the syntactic gain of translating $t_i$:
- Closed-class functional words and modal modifiers: $\text{TB} = 0.85$.
- Verbal affixes (`-ize`, `-ify`, `-ed`, `-ing`, `-ate`): $\text{TB} = 0.80$.
- Pre-Hindi light verb action predicates (e.g., *[Verb] + [karna/hona/dena]*): $\text{TB} = 0.75$.
- Core domain nouns and nominal entities: $\text{TB} = 0.15\text{--}0.20$.
- **Code Location:** `src/laps/feature_extractor.py:261-296`

---

### 1.3 Composite Preservation Score

The weighted preservation score combines features linearly with a subtractive penalty for Translation Benefit:

$$\text{Score}(t_i) = w_1 \text{BF} + w_2 \text{DS} + w_3 \text{SI} + w_4 \text{CC} + w_5 \text{AF} - w_6 \text{TB}$$

clamped to $[0.0, 1.0]$.

#### Default Simplex Weights:
$$\mathbf{w}_{\text{default}} = [0.20, 0.20, 0.20, 0.15, 0.15, 0.10]^\top$$

#### Register-Calibrated Optimal Weight Vectors ($\mathbf{w}^*$):
Produced by SLSQP optimization on development splits (`data/weights/`):
- **Academic:** $\mathbf{w}^* = [0.00, 0.20, 0.20, 0.40, 0.20, 0.00]$ (Prioritizes context consistency & domain specificity)
- **Business:** $\mathbf{w}^* = [0.15, 0.25, 0.20, 0.20, 0.10, 0.10]$
- **Casual:** $\mathbf{w}^* = [0.30, 0.10, 0.15, 0.20, 0.20, 0.05]$ (Prioritizes empirical borrowing frequency)
- **Social:** $\mathbf{w}^* = [0.25, 0.15, 0.15, 0.20, 0.20, 0.05]$

---

### 1.4 Soft Sigmoid Logistic Preservation Boundary

LAPS replaces binary step functions with a continuous logistic probability boundary:

$$P(\text{Preserve} \mid t_i, C, \theta_R) = \sigma(\lambda \cdot (\text{Score}(t_i) - \theta_R)) = \frac{1}{1 + e^{-\lambda(\text{Score}(t_i) - \theta_R)}}$$

where $\lambda = 10.0$ (steepness parameter).

#### Register Thresholds ($\theta_R$):
- **Casual ($R=\text{casual}$):** $\theta_R = 0.35$ (High preservation of informal slang/contractions)
- **Social ($R=\text{social}$):** $\theta_R = 0.40$
- **Business ($R=\text{business}$):** $\theta_R = 0.50$ (Balanced professional threshold)
- **Academic ($R=\text{academic}$):** $\theta_R = 0.65$ (High formality requirement)

#### Knowledge Graph Ego-Density Modulation:
$$P_{\text{adj}} = \text{clip}\left(P + (\text{CC}_{\text{ego}} - 0.50) \cdot 0.10, 0.0, 1.0\right)$$

#### Three-Zone Decision Classification:
1. **$P_{\text{adj}} \ge 0.70 \implies \textbf{PRESERVE}$:** High confidence. Explicitly preserved.
2. **$0.50 \le P_{\text{adj}} < 0.70 \implies \textbf{UNCERTAIN}$:** Borderline confidence. Preserved in output with audit warning.
3. **$P_{\text{adj}} < 0.50 \implies \textbf{REWRITE}$:** Low confidence. Eligible for translation.

---

## 2. Mathematical Audit & Code Discrepancies Table

| Item | Documented Formulation | Actual Code Implementation | Discrepancy Analysis | Verification Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Edge Schema** | `source_id`, `target_id` in `LAPS.md` | `source`, `target` in `graph_engine.py` & DB | Column name naming mismatch | Code is authoritative; queries function correctly |
| **TB Weight Sign** | Documented as $- w_6 \cdot \text{TB}$ | In `scorer.py`: $- w[\text{"w6"}] \cdot \text{features}[\text{"TB"}]$ | Consistent subtractive formulation | Match verified |
| **Sum of Weights** | Simplex $\sum w_j = 1.0$ | Sum of positive weights is $0.90$, with $-0.10 \text{TB}$ | Net sum $\le 1.0$, preventing output score explosion | Verified safe; scores naturally bounded $[0, 1]$ |
| **Hard Step Ablation** | Expected distinct F1 score | Produced identical F1 ($0.9603$) | At cutoff $0.50$, $\sigma(\lambda(S - \theta)) \ge 0.5 \iff S \ge \theta$ | Mathematically identical at $\theta$ boundary |
| **Ablation Weight Indexing** | Weight removal should drop target feature | `run_ablation_study.py` re-indexed remaining weights by loop position | Caused weight shifts (e.g., $w_1 \to \text{DS}$ when $-\text{BF}$) | Verified defect; fixed in code |
| **Annotator Count** | `gold_corpus.json` header: 3 | `annotations.csv`: 2 raters | Header comment out of sync with actual rater file | 2 raters independently verified ($\kappa=0.8739$) |
