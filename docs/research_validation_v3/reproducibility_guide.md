# LAPS Master Reproducibility Guide (v3)

**Document:** `docs/research_validation_v3/reproducibility_guide.md`  
**Date:** October 13, 2026  
**Scope:** Exact, validated command-line instructions to reproduce all experimental findings from scratch  

---

## 1. Environment Setup & Dependency Verification

Ensure Python 3.10+ and a CUDA-enabled PyTorch environment are active:

```powershell
# 1. Verify CUDA availability and device name
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 2. Verify key dependencies
python -c "import transformers, sacrebleu, sentence_transformers; print('Transformers:', transformers.__version__); print('SacreBLEU:', sacrebleu.__version__)"
```

---

## 2. Test Suite Execution

Run the complete regression and unit test suite (55 tests):

```powershell
python -m pytest tests/ -v
```
*Expected Result:* `55 passed, 17 warnings in ~36-39s (Exit Code: 0)`.

---

## 3. Strict Real-Neural Paired Benchmark Execution

### 3.1 Held-Out Test Split ($N=20$, including 5 OOV sentences)
Executes strictly on GPU, failing immediately if any surrogate fallback occurs:

```powershell
python scripts/benchmark.py `
  --corpus data/evaluation/test_split.json `
  --backend-mode real_neural `
  --require-real-neural `
  --device cuda `
  --output results/research_validation_v2/benchmark_real_neural_test_split.json `
  --output-v2 results/research_validation_v2/benchmark_real_neural_test_split_v2.json `
  --save-predictions results/research_validation_v2/predictions_real_neural_test_split.json
```

### 3.2 Full Gold Corpus ($N=100$, 327 Candidate Loanwords)

```powershell
python scripts/benchmark.py `
  --corpus data/evaluation/gold_corpus.json `
  --backend-mode real_neural `
  --require-real-neural `
  --device cuda `
  --output results/research_validation_v2/benchmark_real_neural_gold_100.json `
  --output-v2 results/research_validation_v2/benchmark_real_neural_gold_100_v2.json `
  --save-predictions results/research_validation_v2/predictions_real_neural_gold_100.json
```

---

## 4. Execution Mode Verification Script

Verify that zero surrogate fallbacks occurred across all saved predictions:

```powershell
python -c "
import json
for split in ['test_split', 'gold_100']:
    path = f'results/research_validation_v2/predictions_real_neural_{split}.json'
    data = json.load(open(path, encoding='utf-8'))
    recs = data['records']
    modes = {sys: set(r['predictions'][sys].get('backend_execution_mode') for r in recs) for sys in recs[0]['predictions']}
    print(f'=== MODES FOR {split} (N={len(recs)}) ===')
    for s, m in modes.items():
        print(f'  {s}: {m}')
"
```
*Expected Result:*
- LAPS: `{'REAL_INFERENCE_CONSTRAINED'}` (or `REAL_INFERENCE_UNCONSTRAINED` when no candidates exist)
- Unconstrained NMT: `{'REAL_INFERENCE_UNCONSTRAINED'}`
- Zero `DETERMINISTIC_SURROGATE_FALLBACK` instances.

---

## 5. Architectural & Feature Ablation Study

Run the audited ablation study across the held-out test split, validation split, and gold corpus:

```powershell
# 1. Held-Out Test Split (Primary Generalization Benchmark)
python scripts/run_ablation_study.py --corpus data/evaluation/test_split.json --output results/research_validation_v2/ablation_results_test.json

# 2. Validation Split (Threshold Calibration)
python scripts/run_ablation_study.py --corpus data/evaluation/val_split.json --output results/research_validation_v2/ablation_results_val.json

# 3. Full Gold Corpus (Descriptive Full-Scale Evaluation)
python scripts/run_ablation_study.py --corpus data/evaluation/gold_corpus.json --output results/research_validation_v2/ablation_results_audited.json
```

---

## 6. Operational Latency Profiling

Profile all 6 pipeline stages on the GPU across 20 sentences:

```powershell
python scripts/profile_operational_latency.py --output results/research_validation_v2/operational_latency_profile.json --sample-size 20
```

---

## 7. Evidence Manifest Recomputation & Hash Verification

Verify the SHA-256 integrity of all artifacts in `results/research_validation_v2/`:

```powershell
python -c "
import json, hashlib
from pathlib import Path
m = json.load(open('results/research_validation_v2/evidence_manifest.json'))
for a in m['artifacts']:
    p = Path(a['relative_path'])
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert h == a['sha256'], f'Hash mismatch on {p}'
print('All manifest hashes verified 100% successfully.')
"
```
