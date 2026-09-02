"""
LAPS Neural Subword & Statistical Language Identification (LID) Engine
-----------------------------------------------------------------------
Provides noise-resilient Language Identification for Romanized Indic (Hinglish) text.
Features:
  1. PyTorch Character-CNN Classifier (`CharCNN`) trained on Latin-script Hindi vs English words.
  2. Calibrated Bayesian Subword Character N-Gram Log-Likelihood Ratio fallback.
  3. Phonetic Transliteration Normalization (handling chat contractions: 'smjh', 'cl', 'plzz', 'chahiye').
  4. Continuous Calibrated Posterior Probabilities: P(HI | token), P(EN | token).
  5. Devanagari script detection.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import wordfreq
except ImportError:
    wordfreq = None

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "data" / "models" / "char_lid.pt"

# Character Vocabulary for PyTorch Model
CHARS = " abcdefghijklmnopqrstuvwxyz0123456789-.,!?#'\""
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
UNK_IDX = len(CHAR2IDX) + 1
VOCAB_SIZE = UNK_IDX + 1

# ---------------------------------------------------------------------------
# Phonetic & Transliteration Equivalence Normalizer
# ---------------------------------------------------------------------------

PHONETIC_NORMALIZATION_MAP = {
    # Common Hindi chat contractions & phonetic variants
    r'\b(chahye|chaheye|chaiye|chye|chahie|chahey)\b': 'chahiye',
    r'\b(smjh|smjhna|smjha|smjhe|smjhana)\b':           'samajh',
    r'\b(kro|kren|krna|krr|karo|karein)\b':             'karna',
    r'\b(hogya|hogyi|hogaye|hogya)\b':                  'hoya',
    r'\b(bhot|bahot|bohot)\b':                          'bahut',
    r'\b(zyada|jiyada|ziyada)\b':                       'zyada',
    r'\b(thoda|thodi|thode|thoda)\b':                   'thoda',
    r'\b(accha|achha|acha|achhi)\b':                    'accha',
    r'\b(plz|plzz|pleez|pls)\b':                        'please',
    r'\b(thx|tq|thanx|thnx)\b':                         'thanks',
    r'\b(cl|kal|kl)\b':                                 'kal',
}

# Empirical Subword character n-gram profiles (Log frequencies)
HINDI_CHAR_NGRAMS: Dict[str, float] = {
    "ega": 4.8, "enge": 4.5, "unga": 4.6, "aaye": 4.4, "aaya": 4.7, "aayi": 4.6,
    "iye": 4.3, "iyega": 4.4, "kare": 4.5, "karna": 4.8, "raha": 4.9, "rahi": 4.9,
    "rahe": 4.9, "hoga": 4.8, "hogi": 4.8, "honge": 4.7, "wala": 4.5, "wali": 4.5,
    "wale": 4.5, "daar": 4.2, "baaz": 4.1, "iyat": 4.0, "panti": 4.1, "bh": 4.9,
    "dh": 4.8, "gh": 4.7, "jh": 4.9, "kh": 4.9, "ph": 4.7, "th": 4.2, "ch": 4.5,
    "chh": 5.0, "aa": 4.2, "ee": 4.1, "oo": 4.0, "kuch": 4.8, "kaha": 4.7,
    "kais": 4.8, "kyun": 4.9, "apna": 4.6, "apni": 4.6, "apne": 4.6, "huma": 4.7,
    "tumh": 4.8, "unke": 4.7, "iske": 4.7, "usse": 4.7, "isse": 4.7, "pass": 4.3,
    "paas": 4.8, "saath": 4.8, "pehl": 4.6, "baad": 4.6, "leki": 4.7, "parn": 4.6,
}

ENGLISH_CHAR_NGRAMS: Dict[str, float] = {
    "tion": 5.0, "ment": 4.9, "able": 4.8, "ible": 4.7, "ence": 4.7, "ance": 4.7,
    "ness": 4.8, "less": 4.6, "ship": 4.5, "ized": 4.6, "ting": 4.7, "ling": 4.4,
    "date": 4.5, "ject": 4.8, "link": 4.9, "base": 4.8, "server": 5.0, "client": 5.0,
    "pro": 4.6, "con": 4.5, "inter": 4.8, "trans": 4.9, "sub": 4.4, "auto": 4.6,
    "micro": 4.7, "tele": 4.6, "graph": 4.7, "phone": 4.8, "sync": 4.9, "code": 4.9,
    "post": 4.6, "sched": 4.9, "acad": 4.8, "viva": 4.9, "exam": 4.9, "soft": 4.8,
}

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def normalize_phonetic_transliteration(token: str) -> str:
    """Normalize noisy chat transliterations to canonical subword forms."""
    t_clean = token.lower().strip()
    for pattern, canonical in PHONETIC_NORMALIZATION_MAP.items():
        t_clean = re.sub(pattern, canonical, t_clean)
    return t_clean


def compute_char_ngram_score(token: str, profile: Dict[str, float]) -> float:
    """Score token against character n-gram profile (orders 2, 3, 4)."""
    t = f"^{token.lower()}$"
    score = 0.0
    matches = 0
    for n in (2, 3, 4):
        for i in range(len(t) - n + 1):
            gram = t[i:i+n]
            if gram in profile:
                score += profile[gram]
                matches += 1
    return score / (len(t) + 1e-9)


if HAS_TORCH:
    class _CharCNNNet(nn.Module):
        def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, num_filters=64, num_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=4, padding=2)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(num_filters * 2, num_classes)

        def forward(self, x):
            emb = self.embed(x).transpose(1, 2)
            c1 = F.relu(self.conv1(emb))
            p1 = F.adaptive_max_pool1d(c1, 1).squeeze(2)
            c2 = F.relu(self.conv2(c1))
            p2 = F.adaptive_max_pool1d(c2, 1).squeeze(2)
            feat = torch.cat([p1, p2], dim=1)
            feat = self.dropout(feat)
            return self.fc(feat)
else:
    _CharCNNNet = None


class SubwordLanguageIdentifier:
    """
    Subword Neural & Statistical Language Identifier for Romanized Indic Text.
    Uses trained PyTorch 1D-CNN with Bayesian Character N-Gram fallback.
    """

    def __init__(self):
        self.model = None
        if HAS_TORCH and MODEL_PATH.exists():
            try:
                self.model = _CharCNNNet()
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
                self.model.eval()
            except Exception:
                self.model = None

    def _encode_token(self, word: str, max_len: int = 24) -> list[int]:
        w = word.lower().strip()[:max_len]
        idxs = [CHAR2IDX.get(c, UNK_IDX) for c in w]
        if len(idxs) < max_len:
            idxs += [0] * (max_len - len(idxs))
        return idxs

    def predict_token_language(self, token: str) -> Dict[str, Any]:
        tok_raw = token.strip()
        tok_low = tok_raw.lower()

        # 1. Punctuation & Numeric
        if re.fullmatch(r'[\d\W]+', tok_raw):
            return {
                "token": tok_raw,
                "language": "PUNCT",
                "is_candidate": False,
                "p_hi": 0.0,
                "p_en": 0.0,
                "script": "punct"
            }

        # 2. Devanagari script
        if DEVANAGARI_RE.search(tok_raw):
            return {
                "token": tok_raw,
                "language": "HI",
                "is_candidate": False,
                "p_hi": 1.0,
                "p_en": 0.0,
                "script": "devanagari"
            }

        # 3. Frequency & Phonetic Normalization
        normalized = normalize_phonetic_transliteration(tok_low)

        # 4. Neural PyTorch Prediction if available
        if self.model is not None and HAS_TORCH:
            try:
                with torch.no_grad():
                    inp = torch.tensor([self._encode_token(normalized)], dtype=torch.long)
                    logits = self.model(inp)
                    probs = F.softmax(logits, dim=1).squeeze(0).tolist()
                    # Class 0: EN, Class 1: HI
                    p_en_neural = round(probs[0], 4)
                    p_hi_neural = round(probs[1], 4)

                    # Check empirical wordfreq prior (only for unambiguous English frequency where neural model is not confident of Hindi)
                    zipf_en = wordfreq.zipf_frequency(tok_low, "en") if wordfreq else 0.0
                    if zipf_en >= 4.2 and probs[1] < 0.50:
                        p_en_neural = max(p_en_neural, 0.85)
                        p_hi_neural = min(p_hi_neural, 0.15)

                    lang = "HI" if p_hi_neural > p_en_neural else "EN"
                    is_cand = (lang == "EN" and len(tok_low) >= 2)

                    return {
                        "token": tok_raw,
                        "language": lang,
                        "is_candidate": is_cand,
                        "p_hi": p_hi_neural,
                        "p_en": p_en_neural,
                        "script": "latin",
                        "normalized": normalized,
                        "backend": "pytorch_char_cnn"
                    }
            except Exception:
                pass

        # 5. Statistical Bayesian Subword Fallback
        hi_score = compute_char_ngram_score(normalized, HINDI_CHAR_NGRAMS)
        en_score = compute_char_ngram_score(normalized, ENGLISH_CHAR_NGRAMS)

        exp_hi = math.exp(min(5.0, hi_score * 2.0))
        exp_en = math.exp(min(5.0, en_score * 2.0))
        total_exp = exp_hi + exp_en
        p_hi = round(exp_hi / total_exp, 4)
        p_en = round(exp_en / total_exp, 4)

        if wordfreq and wordfreq.zipf_frequency(tok_low, "en") >= 3.0:
            p_en = max(p_en, 0.85)
            p_hi = min(p_hi, 0.15)

        is_hindi = (p_hi > p_en) or (hi_score > 0.35 and en_score < 0.20)
        lang = "HI" if is_hindi else "EN"
        is_cand = (lang == "EN" and len(tok_low) >= 2)

        return {
            "token": tok_raw,
            "language": lang,
            "is_candidate": is_cand,
            "p_hi": p_hi,
            "p_en": p_en,
            "script": "latin",
            "normalized": normalized,
            "backend": "bayesian_subword_ngram"
        }


# Alias for backward compatibility
NeuralLanguageIdentifier = SubwordLanguageIdentifier

_LID_INSTANCE = None

def get_neural_lid() -> SubwordLanguageIdentifier:
    global _LID_INSTANCE
    if _LID_INSTANCE is None:
        _LID_INSTANCE = SubwordLanguageIdentifier()
    return _LID_INSTANCE
