# Inferential Statistics & Significance Testing Audit (v3)

**Document:** `docs/research_validation_v3/statistical_analysis.md`  
**Date:** October 13, 2026  
**Scope:** Rigorous audit of statistical methodology, resampling design, and hypothesis testing  

---

## 1. Statistical Methodology & Independent Evaluation Units

### 1.1 Choice of Independent Evaluation Unit
In machine translation evaluation, words within a single utterance share syntactic context and language model conditioning. Treating individual tokens as independent observations leads to severe anticonservatism (inflated degrees of freedom and artificially tiny $p$-values).

**Audit Decision:**
All statistical hypothesis testing in LAPS is conducted at the **sentence level** ($N=20$ on held-out test split, $N=100$ on gold corpus), using per-sentence Output F1 differences as the paired test statistic:

$$D_i = \text{F1}_{\text{LAPS}}(s_i) - \text{F1}_{\text{Baseline}}(s_i)$$

Where each sentence $s_i$ is treated as an independent observation.

---

## 2. Hypothesis Testing: Paired Monte Carlo Permutation Test

### 2.1 Formal Hypothesis Definition
- **Null Hypothesis ($H_0$):** There is no difference in Output Retention F1 between LAPS and the baseline system ($\mathbb{E}[D] = 0$). Any observed difference is due to random label assignment.
- **Alternative Hypothesis ($H_1$):** LAPS achieves significantly higher Output Retention F1 than the baseline system ($\mathbb{E}[D] > 0$).

### 2.2 Test Procedure & Parameters
- **Resampling Procedure:** Two-sided Monte Carlo Paired Permutation Test with random sign flips.
- **Permutation Samples ($B$):** 5,000 iterations.
- **Random Seed:** 42 (fixed for exact reproducibility).
- **Multiple Comparisons Correction:** Bonferroni adjustment across the 3 baseline comparisons:
  $$\alpha_{\text{adjusted}} = \frac{0.05}{3} \approx 0.0167$$

### 2.3 Mathematical Note on $p = 0.0002$
In a Monte Carlo permutation test with $B = 5,000$ resamples:

$$\hat{p} = \frac{1 + \sum_{b=1}^B \mathbb{I}(|T^{(b)}| \ge |T_{\text{obs}}|)}{1 + B}$$

When zero simulated resamples exceed the observed test statistic ($T_{\text{obs}} = 0.7063$), the empirical estimate is:

$$\hat{p} = \frac{1 + 0}{5,001} \approx 0.00019996 \approx 0.0002$$

> **Scientific Disclosure:** The reported $p$-value of **0.0002** represents the **empirical floor** of the 5,000-sample permutation test. It demonstrates that under 5,000 random sign flips, not a single permuted sample achieved a difference as extreme as LAPS vs Unconstrained NMT.

---

## 3. Inferential Results & Bootstrap Confidence Intervals

### 3.1 Held-Out Test Split ($N=20$)

| Comparison: LAPS vs System | Observed Mean Difference ($\Delta$ F1) | 95% Bootstrap Confidence Interval | Empirical $p$-value | Bonferroni Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **vs Unconstrained NMT Baseline** | **+0.7063** | **[0.5619, 0.8307]** | **0.0002** | **Yes** ($\alpha < 0.0167$) |
| **vs Static Dictionary Filter** | **+0.6930** | **[0.5452, 0.8250]** | **0.0002** | **Yes** ($\alpha < 0.0167$) |
| **vs Few-Shot Vocabulary Filter** | **+0.6596** | **[0.5167, 0.7908]** | **0.0002** | **Yes** ($\alpha < 0.0167$) |

### 3.2 Full Gold Corpus ($N=100$)

| Comparison: LAPS vs System | Observed Mean Difference ($\Delta$ F1) | 95% Bootstrap Confidence Interval | Empirical $p$-value | Bonferroni Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **vs Unconstrained NMT Baseline** | **+0.7472** | **[0.6968, 0.7950]** | **0.0002** | **Yes** ($\alpha < 0.0167$) |
| **vs Static Dictionary Filter** | **+0.7125** | **[0.6545, 0.7642]** | **0.0002** | **Yes** ($\alpha < 0.0167$) |
| **vs Few-Shot Vocabulary Filter** | **+0.6662** | **[0.6041, 0.7260]** | **0.0002** | **Yes** ($\alpha < 0.0167$) |

---

## 4. Methodological Limitations & Sample Size Caveats

1. **Test Set Scale ($N=20$):** While the test split is statistically sufficient to reject the null hypothesis at $p < 0.001$ due to the massive effect size ($\Delta\text{F1} > 0.70$), 20 sentences is a compact pilot test split. The paper should present this as a rigorous proof-of-concept benchmark rather than a large-scale industrial benchmark.
2. **Confidence Interval Width:** The 95% CI on the test split spans $[0.56, 0.83]$, reflecting sample variance. On the 100-sentence corpus, the interval tightens to $[0.70, 0.80]$.
3. **Statistical Significance vs Semantic Adequacy:** A statistically significant increase in token retention proves that constrained decoding prevents borrowing erasure, but does not imply that overall sentence naturalness is improved.
