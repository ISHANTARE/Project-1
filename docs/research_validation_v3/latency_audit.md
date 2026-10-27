# Operational Latency & Computational Profile Audit (v3)

**Document:** `docs/research_validation_v3/latency_audit.md`  
**Date:** October 13, 2026  
**Script:** `scripts/profile_operational_latency.py`  
**Artifact:** `results/research_validation_v2/operational_latency_profile.json`  
**Test Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM), Intel Core i7-13th Gen, 16 GB RAM  

---

## 1. Disentangled Stage-by-Stage Latency Breakdown

Latency was profiled across $N=20$ sentences following a 3-sentence GPU warm-up phase:

| Pipeline Stage | Mean (ms) | Median (ms) | Std Dev (ms) | Min (ms) | Max (ms) | P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: CMI Detection** | 4.42 | 3.94 | 1.85 | 2.12 | 8.45 | 7.29 |
| **Stage 2: Tokenization & LID** | 2.89 | 2.63 | 1.34 | 1.45 | 6.12 | 5.26 |
| **Stage 3: Live Feature Extraction (MiniLM + KG)** | 57.45 | 58.84 | 18.22 | 28.14 | 104.50 | 100.31 |
| **Stage 4: Preservation Scoring (Dot Product)** | 0.01 | 0.01 | 0.00 | 0.01 | 0.02 | 0.02 |
| **Stage 5: Dynamic Thresholding (`batch_decide`)** | 6.78 | 6.85 | 2.10 | 3.20 | 11.40 | 10.54 |
| **Stage 6A: Unconstrained Neural Translation** | 218.15 | 233.32 | 48.72 | 145.20 | 320.10 | 311.44 |
| **Stage 6B: LAPS Constrained Neural Generation** | 1,690.22 | 1,892.51 | 412.35 | 980.40 | 2,410.20 | 2,349.43 |
| **End-to-End Live LAPS Processing** | 1,979.93 | 2,183.76 | 435.12 | 1,180.50 | 2,750.30 | 2,618.26 |
| *Pre-Cached Scoring Alone (Memory-only)* | *0.02* | *0.02* | *0.00* | *0.01* | *0.02* | *0.02* |

---

## 2. Forensic Reconciliation: Historical Sub-Millisecond Claims vs Live Latency

A frequent point of confusion in earlier repository audit notes was the claim that LAPS operates in "less than 0.25 milliseconds".

**Audit Finding & Scientific Clarification:**
1. **Pre-Cached Memory Arithmetic ($\approx 0.01 - 0.02$ ms):**
   When feature vectors are pre-computed and stored in memory, evaluating the weighted dot product and applying the sigmoid equation requires only a few floating-point CPU operations per token, executing in $\sim 10 - 20$ microseconds ($0.01 - 0.02$ ms).
2. **Live Uncached Feature Extraction ($\approx 57.45$ ms):**
   When an unseen sentence arrives (such as OOV utterances), computing MiniLM dense sentence embeddings and querying SQLite Knowledge Graph tables requires $\sim 57$ ms.
3. **Constrained Beam Search Generation ($\approx 1,690$ ms):**
   During real neural decoding on the GPU, HuggingFace Constrained Beam Search tracks trie-based lexical constraints across 8 to 16 beams. This combinatorial search space increases translation latency from $\sim 218$ ms (unconstrained) to $\sim 1.69$ s (constrained).
4. **Conclusion:** LAPS's decision engine adds only $\sim 71$ ms of overhead. Total end-to-end latency ($\sim 1.98$ s) is entirely dominated by the constrained beam search decoding algorithm.

---

## 3. Production Deployment Implications

For latency-critical production environments (e.g., interactive messaging):
- **Lexical Cache:** Caching frequent Hinglish borrowing features eliminates Stage 3, reducing pre-generation overhead to $< 15$ ms.
- **Speculative / Fast Decoding:** Replacing full constrained beam search with speculative token insertion or guided greedy decoding can reduce Stage 6B latency by $3-5\times$.
