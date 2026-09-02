"""
Train Lightweight PyTorch Character-CNN Language Identifier
------------------------------------------------------------
Trains a small, fast 1D-CNN over character embeddings on English vs Romanized Hindi words.
Saves model weights to data/models/char_lid.pt.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "char_lid.pt"

# Character Vocabulary
CHARS = " abcdefghijklmnopqrstuvwxyz0123456789-.,!?#'\""
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is padding, len+1 is UNK
UNK_IDX = len(CHAR2IDX) + 1
VOCAB_SIZE = UNK_IDX + 1

def encode_word(word: str, max_len: int = 24) -> list[int]:
    w = word.lower().strip()[:max_len]
    idxs = [CHAR2IDX.get(c, UNK_IDX) for c in w]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    return idxs

class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, num_filters=64, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=4, padding=2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(num_filters * 2, num_classes)

    def forward(self, x):
        emb = self.embed(x).transpose(1, 2)  # [B, embed_dim, L]
        c1 = F.relu(self.conv1(emb))
        p1 = F.adaptive_max_pool1d(c1, 1).squeeze(2)
        c2 = F.relu(self.conv2(c1))
        p2 = F.adaptive_max_pool1d(c2, 1).squeeze(2)
        feat = torch.cat([p1, p2], dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)

def get_training_data():
    try:
        import wordfreq
        top_en = [w for w in wordfreq.top_n_list("en", 3000) if w.isalpha() and len(w) >= 2]
    except ImportError:
        top_en = ["meeting", "calendar", "server", "client", "project", "deadline", "schedule", "database"] * 200

    hindi_stems = [
        "kisi", "kiske", "kiska", "kiski", "kahan", "kaise", "kab", "kyun", "kya", "yeh", "woh", "voh",
        "main", "hum", "aap", "tum", "mera", "meri", "mere", "tera", "teri", "tere", "hamara", "hamari",
        "tumhara", "unka", "uska", "unhe", "inhe", "mujhe", "tujhe", "humein", "kuch", "koi", "sab",
        "apna", "apni", "apne", "mein", "par", "pe", "tak", "saath", "liye", "baad", "pehle", "paas",
        "samne", "aage", "piche", "andar", "baahar", "hai", "hain", "hoon", "gaya", "gayi", "raha",
        "rahi", "hoga", "hogi", "karna", "karein", "kiya", "dena", "diya", "lena", "liya", "aana",
        "jaana", "sakta", "chahiye", "milla", "lekin", "parantu", "phir", "kabhi", "hamesha", "aaj",
        "kal", "parso", "subah", "shaam", "bahut", "zyada", "thoda", "accha", "sahi", "galat", "nahin",
        "chahye", "smjh", "smjhna", "samajh", "samajhna", "samajhe", "kro", "kren", "krna", "hogya",
        "hogyi", "bhot", "ziyada", "achha", "karega", "karenge", "karegi", "aayega", "aayegi", "samjho",
        "rakho", "dekho", "batao", "kijiye", "dekhna", "bolna", "sunna", "puchna", "chhodna", "bhejna",
        "mangna", "pahunch", "rokna", "milna", "bhai", "yaar", "theek", "shukriya", "dost", "waqt",
        "zaroorat", "sawal", "jawab", "baat", "baatein", "khabar", "din", "raat", "kaam", "chalo"
    ]

    suffixes = ["ega", "egi", "enge", "unga", "ungi", "kar", "ke", "ne", "ko", "wala", "wali", "wale", "daar"]
    aug_hi = list(hindi_stems)
    for s in hindi_stems:
        for suf in suffixes:
            if len(s) + len(suf) < 18:
                aug_hi.append(s + suf)

    data = []
    # EN is class 0, HI is class 1
    for w in top_en:
        data.append((w, 0))
    for w in aug_hi:
        data.append((w, 1))

    random.seed(42)
    random.shuffle(data)
    return data

def train():
    print("Collecting training data for Romanized LID...")
    data = get_training_data()
    print(f"Total training tokens: {len(data)}")

    xs = [encode_word(w) for w, _ in data]
    ys = [label for _, label in data]

    X = torch.tensor(xs, dtype=torch.long)
    Y = torch.tensor(ys, dtype=torch.long)

    model = CharCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    epochs = 6
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        correct = 0
        for i in range(0, len(X), batch_size):
            b_idx = perm[i:i+batch_size]
            bx, by = X[b_idx], Y[b_idx]
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()

        acc = correct / len(X)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(X):.4f} - Accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    train()
