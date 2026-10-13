# Experimental Reproducibility Guide

**Author:** NLP Research & Systems Evaluation Team  
**Date:** October 2026  
**Status:** Reproducible Research Standard  

---

## 1. System & Environment Specifications

All experiments, neural benchmarks, and latency profiles were executed and verified on the following hardware and software configuration:

- **Operating System:** Microsoft Windows 11 Home (x86_64)
- **Host GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM)
- **Host CPU:** 13th Gen Intel(R) Core(TM) i5-13500H (16 threads)
- **Python Runtime:** Python 3.10.9
- **PyTorch:** `torch==2.11.0+cu128` (CUDA 12.8 enabled)
- **Hugging Face Transformers:** `transformers==5.15.1`
- **Sentence-Transformers:** `sentence-transformers==3.0.1`
- **SacreBLEU:** `sacrebleu==2.6.0`
- **Fixed Random Seed:** `42` (Permutation test samples: $B = 5,000$)

---

## 2. Environment Setup

```bash
# 1. Clone repository and navigate to root
cd Project-1-main

# 2. Install pinned dependencies
pip install -r requirements.txt
pip install sacrebleu==2.6.0 sentence-transformers==3.0.1

# 3. Verify PyTorch CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

## 3. End-to-End Experiment Replication Commands

### 3.1. Verify Test Suite (53 Tests Passing)
Executes all regression, unit, and neural integration tests:
```bash
pytest -v
```
*Expected Result:* `53 passed in ~30s`.

### 3.2. Reproduce Held-Out Test Split Benchmark (N=20)
Runs all 4 systems on the held-out test split with live GPU constrained decoding:
```bash
python scripts/benchmark.py \
  --corpus data/evaluation/test_split.json \
  --backend real_neural \
  --model Helsinki-NLP/opus-mt-hi-en \
  --device cuda \
  --permutations 5000 \
  --seed 42 \
  --output results/benchmark_real_neural_test_split.json
```
*Expected Outputs:*
- `results/benchmark_real_neural_test_split.json`
- `results/predictions_real_neural_test_split.json`

### 3.3. Reproduce Feature & Architectural Ablation Study (N=327)
Executes mathematical feature ablations with normalized reweighting and register gating:
```bash
python scripts/run_ablation_study.py \
  --corpus data/evaluation/gold_corpus.json \
  --output results/ablation_results_audited.json
```
*Expected Output:*
- `results/ablation_results_audited.json`

### 3.4. Reproduce Operational Latency Profile
Disentangles preprocessing, live feature extraction, and GPU beam search decoding:
```bash
python scripts/profile_operational_latency.py \
  --device cuda \
  --output results/operational_latency_profile.json
```
*Expected Output:*
- `results/operational_latency_profile.json`

---

## 4. Key Artifact Locations

| Artifact Type | File Path | Description |
| :--- | :--- | :--- |
| **Neural Test Benchmark** | `results/benchmark_real_neural_test_split.json` | 4-system metrics on 20 held-out sentences |
| **Neural Test Predictions** | `results/predictions_real_neural_test_split.json` | Generated text & preserved tokens per system |
| **Neural Gold Benchmark** | `results/benchmark_real_neural_gold_100.json` | 4-system metrics on 100 gold sentences |
| **Ablation Matrix** | `results/ablation_results_audited.json` | 11 ablation variants with normalized weights |
| **Latency Profile** | `results/operational_latency_profile.json` | Stage-by-stage microsecond measurements |
| **Test Suite** | `tests/test_laps.py` | 53 unit and regression test specifications |
