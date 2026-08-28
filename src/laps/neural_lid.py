"""
LAPS Neural Subword & Character N-Gram Language Identification (LID) Engine
---------------------------------------------------------------------------
Provides robust, noise-resilient Language Identification for Romanized Indic (Hinglish) text.
Features:
  1. Character N-Gram (2, 3, 4-gram) Language Model scoring for Latin-script Hindi vs English.
  2. Phonetic Transliteration Normalization (handling chat contractions: 'smjh', 'cl', 'plzz', 'chahiye').
  3. Continuous Language Posterior Probabilities: P(HI | token), P(EN | token).
  4. Devanagari script detection.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple, Any

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

# Subword character n-gram profiles for Romanized Hindi vs English
# High-frequency Hindi character n-grams (prefix, infix, suffix patterns)
HINDI_CHAR_NGRAMS: Dict[str, float] = {
    # Suffixes
    "ega": 4.8, "enge": 4.5, "unga": 4.6, "aaye": 4.4, "aaya": 4.7, "aayi": 4.6,
    "iye": 4.3, "iyega": 4.4, "kare": 4.5, "karna": 4.8, "raha": 4.9, "rahi": 4.9,
    "rahe": 4.9, "hoga": 4.8, "hogi": 4.8, "honge": 4.7, "wala": 4.5, "wali": 4.5,
    "wale": 4.5, "daar": 4.2, "baaz": 4.1, "iyat": 4.0, "panti": 4.1,
    # Infixes & Stems
    "bh": 4.9, "dh": 4.8, "gh": 4.7, "jh": 4.9, "kh": 4.9, "ph": 4.7, "th": 4.2,
    "ch": 4.5, "chh": 5.0, "aa": 4.2, "ee": 4.1, "oo": 4.0, "kuch": 4.8, "kaha": 4.7,
    "kais": 4.8, "kyun": 4.9, "apna": 4.6, "apni": 4.6, "apne": 4.6, "huma": 4.7,
    "tumh": 4.8, "unke": 4.7, "iske": 4.7, "usse": 4.7, "isse": 4.7, "pass": 4.3,
    "paas": 4.8, "saath": 4.8, "pehl": 4.6, "baad": 4.6, "leki": 4.7, "parn": 4.6,
}

# High-frequency English character n-grams
ENGLISH_CHAR_NGRAMS: Dict[str, float] = {
    # Suffixes
    "tion": 5.0, "ment": 4.9, "able": 4.8, "ible": 4.7, "ence": 4.7, "ance": 4.7,
    "ness": 4.8, "less": 4.6, "ship": 4.5, "ized": 4.6, "ting": 4.7, "ling": 4.4,
    "date": 4.5, "ject": 4.8, "link": 4.9, "base": 4.8, "server": 5.0, "client": 5.0,
    # Prefixes & Infixes
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


class NeuralLanguageIdentifier:
    """
    Subword & Character N-gram Language Identifier for Romanized Code-Mixed Text.
    """

    def __init__(self):
        # Known loanword English anchors
        self.common_loanwords = {
            "viva", "calendar", "calender", "link", "meeting", "client", "server",
            "project", "postpone", "deadline", "schedule", "manager", "code",
            "design", "debug", "database", "deploy", "application", "interface",
            "product", "release", "testing", "feedback", "backup", "network",
            "system", "update", "upload", "download", "install", "cancel",
            "confirm", "email", "chat", "call", "message", "login", "logout",
            "password", "data", "file", "folder", "image", "video", "app",
            "zoom", "teams", "assignment", "lab", "exam", "test", "class",
            "lecture", "course", "hi", "hello", "hey", "ok", "okay", "pdf",
            "doc", "ppt", "wifi", "laptop", "ticket", "presentation", "report"
        }

    def predict_token_language(self, token: str) -> Dict[str, Any]:
        """
        Predict language tag and candidate status for a given token.
        """
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

        # 3. Known Loanword check & English Wordfreq frequency check
        try:
            import wordfreq
            is_valid_en = wordfreq.zipf_frequency(tok_low, "en") >= 2.5
        except Exception:
            is_valid_en = False

        if tok_low in self.common_loanwords or is_valid_en:
            return {
                "token": tok_raw,
                "language": "EN",
                "is_candidate": True,
                "p_hi": 0.02,
                "p_en": 0.98,
                "script": "latin"
            }

        # 4. Phonetic normalization & subword n-gram scoring
        normalized = normalize_phonetic_transliteration(tok_low)
        hi_score = compute_char_ngram_score(normalized, HINDI_CHAR_NGRAMS)
        en_score = compute_char_ngram_score(normalized, ENGLISH_CHAR_NGRAMS)

        # Softmax probability distribution
        exp_hi = math.exp(min(5.0, hi_score * 2.0))
        exp_en = math.exp(min(5.0, en_score * 2.0))
        total_exp = exp_hi + exp_en
        p_hi = round(exp_hi / total_exp, 4)
        p_en = round(exp_en / total_exp, 4)

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
        }


_LID_INSTANCE = None

def get_neural_lid() -> NeuralLanguageIdentifier:
    global _LID_INSTANCE
    if _LID_INSTANCE is None:
        _LID_INSTANCE = NeuralLanguageIdentifier()
    return _LID_INSTANCE
