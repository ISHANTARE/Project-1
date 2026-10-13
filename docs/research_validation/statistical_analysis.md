# Statistical Significance and Hypothesis Testing Audit

**Author:** NLP Research & Systems Evaluation Team  
**Date:** October 2026  
**Status:** Audited & Statistically Validated  
**Associated Benchmark Artifacts:**  
- `results/benchmark_real_neural_gold_100.json`
- `results/benchmark_real_neural_test_split.json`

---

## 1. Hypothesis Testing Framework

To establish whether observed performance advantages of LAPS over baseline systems are statistically meaningful rather than artifacts of sample variance, we employ non-parametric hypothesis testing with rigorous multiple-comparison corrections.

### 1.1. Monte Carlo Paired Permutation Test
For two systems $A$ (LAPS) and $B$ (Baseline) evaluated on paired sentences $i = 1, \dots, N$:
1. Let $d_i = \text{Score}_A(s_i) - \text{Score}_B(s_i)$ be the paired metric difference on sentence $i$.
2. The observed test statistic is the sample mean difference:
   $$\bar{d}_{\text{obs}} = \frac{1}{N} \sum_{i=1}^N d_i$$
3. Under the null hypothesis $H_0: \mathbb{E}[d_i] = 0$, the assignment of scores to systems $A$ and $B$ is exchangeable. For $B = 5,000$ iterations:
   $$d_i^{(b)} = r_i \cdot d_i, \quad r_i \sim \text{Uniform}(\{-1, +1\})$$
   $$\bar{d}^{(b)} = \frac{1}{N} \sum_{i=1}^N d_i^{(b)}$$
4. The two-tailed empirical $p$-value with pseudo-count correction is:
   $$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(|\bar{d}^{(b)}| \ge |\bar{d}_{\text{obs}}|)}{B + 1}$$

### 1.2. Multiple Testing Correction: Bonferroni Adjustment
Comparing LAPS against $M = 3$ baseline systems introduces family-wise error rate inflation. To maintain a global significance level of $\alpha = 0.05$:
$$\alpha_{\text{adjusted}} = \frac{\alpha}{M} = \frac{0.05}{3} \approx 0.01667$$
A result is declared statistically significant if and only if $p < 0.01667$.

### 1.3. 95% Bootstrap Confidence Intervals
Paired metric deltas $\Delta = \bar{d}_{\text{obs}}$ are accompanied by non-parametric percentile bootstrap confidence intervals ($B = 5,000$ resamples with replacement, seed = 42).

---

## 2. Statistical Testing Results

### Table 1: Output F1 Significance Testing (Gold Corpus, N=100)

| Comparison Pair | Observed Delta ($\Delta$) | 95% Bootstrap CI | Permutation Samples ($B$) | Empirical $p$-value | Bonferroni Significant ($\alpha = 0.0167$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LAPS vs Few-Shot Filter** | **+0.7577** | $[+0.7000, +0.8107]$ | 5,000 | **$0.0002$** | **Yes (Statistically Significant)** |
| **LAPS vs Static Dict Filter** | **+0.8040** | $[+0.7527, +0.8511]$ | 5,000 | **$0.0002$** | **Yes (Statistically Significant)** |
| **LAPS vs Unconstrained NMT** | **+0.8387** | $[+0.7912, +0.8825]$ | 5,000 | **$0.0002$** | **Yes (Statistically Significant)** |

*Note: With $B=5,000$ iterations where zero permuted samples exceed $|\bar{d}_{\text{obs}}|$, the minimum achievable non-zero $p$-value is $1 / (5000 + 1) \approx 0.0002$.*

---

### Table 2: Output F1 Significance Testing (Held-Out Test Split, N=20: 15 Held-Out + 5 OOV)

| Comparison Pair | Observed Delta ($\Delta$) | 95% Bootstrap CI | Permutation Samples ($B$) | Empirical $p$-value | Bonferroni Significant ($\alpha = 0.0167$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LAPS vs Few-Shot Filter** | **+0.8183** | $[+0.7100, +0.9000]$ | 5,000 | **$0.0002$** | **Yes (Statistically Significant)** |
| **LAPS vs Static Dict Filter** | **+0.8516** | $[+0.7500, +0.9250]$ | 5,000 | **$0.0002$** | **Yes (Statistically Significant)** |
| **LAPS vs Unconstrained NMT** | **+0.8649** | $[+0.7650, +0.9350]$ | 5,000 | **$0.0002$** | **Yes (Statistically Significant)** |

---

## 3. Statistical Power and Sample Size Analysis

1. **Massive Effect Size**:
   The effect size for loanword retention between LAPS and Unconstrained NMT is exceptionally large: Cohen's $d > 3.5$ ($\Delta = +0.8387$ on an absolute $[0, 1]$ scale). In statistical power analysis, detecting an effect size of this magnitude at $\beta = 0.99$ and $\alpha = 0.0167$ requires a minimum sample size of $N \ge 12$. The test split ($N=20$) and full gold corpus ($N=100$) provide well over $99.9\%$ statistical power.
2. **Zero-Crossing Confidence Intervals**:
   None of the 95% bootstrap confidence intervals approach zero. The lower bound for the advantage over Unconstrained NMT remains above $+0.7650$ even on the small held-out split, demonstrating that the preservation advantage is robust across all sampling configurations.
3. **Caveat on Out-of-Domain Generalization**:
   While statistical significance on the Hinglish academic, business, technical, and casual test splits is definitive ($p < 0.0005$), statistical claims are bounded by the domain distributions present in the current corpus. Performance on colloquial social media text (e.g., Twitter/X) with heavy non-standard Romanization may exhibit higher variance.
