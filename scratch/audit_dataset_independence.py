import json
from pathlib import Path
from collections import defaultdict
import difflib

PROJECT_ROOT = Path("d:/Project-1-main/Project-1-main")

def load_split(name):
    path = PROJECT_ROOT / f"data/evaluation/{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sentences", data)

gold = load_split("gold_corpus")
train = load_split("train_split")
val = load_split("val_split")
test = load_split("test_split")

print("=== DATASET SPLIT SIZES ===")
print("Gold Corpus:", len(gold) if gold else "None")
print("Train Split:", len(train) if train else "None")
print("Val Split:  ", len(val) if val else "None")
print("Test Split: ", len(test) if test else "None")

splits = {
    "train": train or [],
    "val": val or [],
    "test": test or [],
    "gold": gold or [],
}

# 1. ID Intersections
train_ids = {s["id"] for s in splits["train"]}
val_ids = {s["id"] for s in splits["val"]}
test_ids = {s["id"] for s in splits["test"]}
gold_ids = {s["id"] for s in splits["gold"]}

print("\n=== ID INTERSECTIONS ===")
print("Train & Val ID overlap:", len(train_ids & val_ids), train_ids & val_ids)
print("Train & Test ID overlap:", len(train_ids & test_ids), train_ids & test_ids)
print("Val & Test ID overlap:", len(val_ids & test_ids), val_ids & test_ids)
print("Train + Val + Test total unique IDs:", len(train_ids | val_ids | test_ids))
print("Gold IDs in (Train | Val | Test):", len(gold_ids & (train_ids | val_ids | test_ids)))

# Is Gold corpus a union of Train + Val + Test or a separate benchmark?
diff_gold = gold_ids - (train_ids | val_ids | test_ids)
diff_splits = (train_ids | val_ids | test_ids) - gold_ids
print("IDs in Gold but not in splits:", len(diff_gold), diff_gold)
print("IDs in splits but not in Gold:", len(diff_splits), diff_splits)

# 2. Text Duplicates & Near-Duplicates
train_texts = {s["sentence"].strip().lower(): s["id"] for s in splits["train"]}
val_texts = {s["sentence"].strip().lower(): s["id"] for s in splits["val"]}
test_texts = {s["sentence"].strip().lower(): s["id"] for s in splits["test"]}

print("\n=== EXACT TEXT DUPLICATES ACROSS SPLITS ===")
train_val_dup = set(train_texts.keys()) & set(val_texts.keys())
train_test_dup = set(train_texts.keys()) & set(test_texts.keys())
val_test_dup = set(val_texts.keys()) & set(test_texts.keys())
print("Train & Val text duplicate count:", len(train_val_dup))
print("Train & Test text duplicate count:", len(train_test_dup))
print("Val & Test text duplicate count:", len(val_test_dup))

# 3. Near-duplicates (Similarity >= 0.85) between Test and Train
near_dups = []
for t_sent, t_id in test_texts.items():
    for tr_sent, tr_id in train_texts.items():
        sim = difflib.SequenceMatcher(None, t_sent, tr_sent).ratio()
        if sim >= 0.85:
            near_dups.append((t_id, tr_id, sim, t_sent, tr_sent))

print(f"\n=== NEAR-DUPLICATES (Test vs Train, similarity >= 0.85): {len(near_dups)} ===")
for t_id, tr_id, sim, t_sent, tr_sent in near_dups:
    print(f"Test {t_id} vs Train {tr_id} (sim={sim:.2f}):")
    print(f"  Test:  {t_sent}")
    print(f"  Train: {tr_sent}")
