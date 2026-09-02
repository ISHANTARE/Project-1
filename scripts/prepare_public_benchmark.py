"""
Prepare Public Benchmark Split & OOV Expansion
-----------------------------------------------
Creates strict Train (70%), Validation (15%), and Test (15%) splits.
Introduces authentic out-of-vocabulary (OOV) technical and conversational tokens
into the held-out test split to rigorously measure out-of-domain generalization.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "evaluation"
GOLD_PATH = EVAL_DIR / "gold_corpus.json"

TRAIN_PATH = EVAL_DIR / "train_split.json"
VAL_PATH   = EVAL_DIR / "val_split.json"
TEST_PATH  = EVAL_DIR / "test_split.json"

# Authentic OOV sentences representing unseen technical & colloquial domains
OOV_TEST_SENTENCES = [
    {
        "id": "OOV_001",
        "sentence": "Kubernetes cluster mein ingress controller configure karo taaki traffic route ho sake",
        "register": "business",
        "reference": "Configure the ingress controller in the Kubernetes cluster so traffic can be routed.",
        "tokens": [
            {"token": "Kubernetes", "label": 1, "domain": "Tech"},
            {"token": "cluster", "label": 1, "domain": "Tech"},
            {"token": "ingress", "label": 1, "domain": "Tech"},
            {"token": "controller", "label": 1, "domain": "Tech"},
            {"token": "configure", "label": 0, "domain": "Tech"},
            {"token": "traffic", "label": 1, "domain": "Tech"},
            {"token": "route", "label": 0, "domain": "Tech"}
        ]
    },
    {
        "id": "OOV_002",
        "sentence": "Is patient ka MRI scan aur blood biochemistry report urgently mangwao",
        "register": "business",
        "reference": "Urgently request this patient's MRI scan and blood biochemistry report.",
        "tokens": [
            {"token": "patient", "label": 1, "domain": "Medical"},
            {"token": "MRI", "label": 1, "domain": "Medical"},
            {"token": "scan", "label": 1, "domain": "Medical"},
            {"token": "blood", "label": 0, "domain": "Medical"},
            {"token": "biochemistry", "label": 1, "domain": "Medical"},
            {"token": "report", "label": 1, "domain": "Medical"}
        ]
    },
    {
        "id": "OOV_003",
        "sentence": "Portfolio ka asset allocation rebalance karo taaki capital gains tax optimize ho",
        "register": "business",
        "reference": "Rebalance the portfolio's asset allocation to optimize capital gains tax.",
        "tokens": [
            {"token": "Portfolio", "label": 1, "domain": "Finance"},
            {"token": "asset", "label": 1, "domain": "Finance"},
            {"token": "allocation", "label": 1, "domain": "Finance"},
            {"token": "rebalance", "label": 0, "domain": "Finance"},
            {"token": "capital", "label": 1, "domain": "Finance"},
            {"token": "gains", "label": 1, "domain": "Finance"},
            {"token": "tax", "label": 1, "domain": "Finance"},
            {"token": "optimize", "label": 0, "domain": "Finance"}
        ]
    },
    {
        "id": "OOV_004",
        "sentence": "Microservices architecture mein circuit breaker pattern implement karna zaroori hai",
        "register": "academic",
        "reference": "Implementing the circuit breaker pattern is essential in microservices architecture.",
        "tokens": [
            {"token": "Microservices", "label": 1, "domain": "Tech"},
            {"token": "architecture", "label": 1, "domain": "Tech"},
            {"token": "circuit", "label": 1, "domain": "Tech"},
            {"token": "breaker", "label": 1, "domain": "Tech"},
            {"token": "pattern", "label": 1, "domain": "Tech"},
            {"token": "implement", "label": 0, "domain": "Tech"}
        ]
    },
    {
        "id": "OOV_005",
        "sentence": "Bhai naye firmware patch ka checksum verify kiya ya direct flash mar diya?",
        "register": "casual",
        "reference": "Bro, did you verify the checksum of the new firmware patch or directly flash it?",
        "tokens": [
            {"token": "firmware", "label": 1, "domain": "Tech"},
            {"token": "patch", "label": 1, "domain": "Tech"},
            {"token": "checksum", "label": 1, "domain": "Tech"},
            {"token": "verify", "label": 0, "domain": "Tech"},
            {"token": "direct", "label": 0, "domain": "General"},
            {"token": "flash", "label": 1, "domain": "Tech"}
        ]
    }
]

def prepare_splits():
    if not GOLD_PATH.exists():
        print(f"Error: {GOLD_PATH} not found.")
        return

    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    sentences = list(gold_data["sentences"])
    random.seed(42)
    random.shuffle(sentences)

    # 70 Train, 15 Val, 15 Test + 5 OOV Test items
    train_sents = sentences[:70]
    val_sents   = sentences[70:85]
    test_sents  = sentences[85:] + OOV_TEST_SENTENCES

    # Save splits
    for path, data_subset, name in [
        (TRAIN_PATH, train_sents, "Train (70 sentences)"),
        (VAL_PATH, val_sents, "Validation (15 sentences)"),
        (TEST_PATH, test_sents, "Held-out Test (20 sentences with OOV)")
    ]:
        out = {
            "description": f"LAPS Partitioned Benchmark Split - {name}",
            "count": len(data_subset),
            "sentences": data_subset
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {name} -> {path} ({len(data_subset)} sentences)")

if __name__ == "__main__":
    prepare_splits()
