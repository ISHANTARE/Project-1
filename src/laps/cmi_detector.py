"""
LAPS Sentence-Level Code-Mixing & Language Detector
---------------------------------------------------
Calculates the Code-Mixing Index (CMI) and detects whether an input
sentence is Monolingual English, Monolingual Hindi, or Code-Mixed.

CMI Formula (Gambäck & Das, 2014):
  CMI = 100 * [1 - (max(N_lang) / N_total)]

If CMI == 0.0 → Monolingual sentence (No code-mixing).
"""

import re
from src.laps.tokenizer import tokenize, HINDI_MARKERS

# Words that exist in English dictionaries but collide with Hinglish transliterated words
AMBIGUOUS_HINDI_EN_COLLISIONS = {"the", "is", "so", "me", "to", "do", "it", "in", "on", "no", "he", "be", "an", "at", "as"}


def detect_sentence_language(sentence: str) -> dict:
    """
    Analyzes a sentence and computes its Code-Mixing Index (CMI) and overall language mode.

    Returns:
        {
            "mode":       "MONOLINGUAL_EN" | "MONOLINGUAL_HI" | "CODE_MIXED",
            "cmi":        float (0.0 to 50.0),
            "en_ratio":   float,
            "hi_ratio":   float,
            "is_code_mixed": bool
        }
    """
    tokens = tokenize(sentence)
    alpha_tokens = [t for t in tokens if t["script"] != "punct"]

    if not alpha_tokens:
        return {"mode": "MONOLINGUAL_EN", "cmi": 0.0, "en_ratio": 1.0, "hi_ratio": 0.0, "is_code_mixed": False}

    n_total = len(alpha_tokens)
    
    # Check if sentence is pure English (has zero definitive Hindi markers)
    has_hindi_markers = any(t["lower"] in HINDI_MARKERS for t in alpha_tokens if t["lower"] not in AMBIGUOUS_HINDI_EN_COLLISIONS)
    has_devanagari = any(t["script"] == "devanagari" for t in alpha_tokens)

    if not has_hindi_markers and not has_devanagari:
        # Check if most words are standard English
        return {
            "mode": "MONOLINGUAL_EN",
            "cmi": 0.0,
            "en_ratio": 1.0,
            "hi_ratio": 0.0,
            "is_code_mixed": False
        }

    # Count language distribution for CMI calculation
    n_hi = sum(1 for t in alpha_tokens if t["language"] == "HI")
    n_en = sum(1 for t in alpha_tokens if t["language"] == "EN")

    max_lang = max(n_hi, n_en)
    cmi = round(100.0 * (1.0 - (max_lang / n_total)), 2)

    if n_hi == 0:
        mode = "MONOLINGUAL_EN"
        is_mixed = False
    elif n_en == 0:
        mode = "MONOLINGUAL_HI"
        is_mixed = False
    elif cmi < 10.0:
        mode = "MONOLINGUAL_EN" if n_en > n_hi else "MONOLINGUAL_HI"
        is_mixed = False
    else:
        mode = "CODE_MIXED"
        is_mixed = True

    return {
        "mode": mode,
        "cmi": cmi,
        "en_ratio": round(n_en / n_total, 2),
        "hi_ratio": round(n_hi / n_total, 2),
        "is_code_mixed": is_mixed
    }


if __name__ == "__main__":
    print(detect_sentence_language("Hi! Does anyone have the link for the viva calendar?"))
    print(detect_sentence_language("hi! Kisi ke pass viva calender ki link h kya?"))
