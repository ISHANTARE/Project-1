"""
LAPS Advanced Tokenizer & Subword Neural Language Tagger (LID)
--------------------------------------------------------------
Hybrid Language Tagger for Code-Mixed Hinglish/Indic Text:
1. Devanagari Script Detection.
2. Neural Subword Character N-Gram Classifier (NeuralLanguageIdentifier).
3. Phonetic Normalization for Romanized Hindi chat contractions ('smjh', 'cl', 'kro', 'chahye').
4. Candidate Loanword Flagging for LAPS feature vector calculation.
"""

import re
import difflib
from pathlib import Path

try:
    from src.laps.neural_lid import get_neural_lid, normalize_phonetic_transliteration
except ModuleNotFoundError:
    from laps.neural_lid import get_neural_lid, normalize_phonetic_transliteration

# ── Devanagari regex ───────────────────────────────────────────────────────────
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_RE      = re.compile(r'[A-Za-z]')

# ── Ambiguous homographs (English words that match Hinglish transliterated forms)
AMBIGUOUS_HOMOGRAPHS = {"the", "is", "so", "me", "to", "do", "it", "in", "on", "no", "he", "be", "an", "at", "as", "hi", "us"}


# ── Comprehensive Transliterated Hindi Marker Vocabulary (Lexical Anchor) ────────
HINDI_MARKERS = {
    # Pronouns & Demonstratives
    "kisi", "kiske", "kiska", "kiski", "kahan", "kaise", "kab", "kyun", "kya",
    "yeh", "woh", "voh", "main", "hum", "aap", "tum", "mera", "meri", "mere",
    "tera", "teri", "tere", "hamara", "hamari", "hamare", "tumhara", "tumhari",
    "tumhare", "unka", "unki", "unke", "iska", "iski", "iske", "uska", "uski",
    "uske", "unhe", "inhe", "mujhe", "tujhe", "humein", "unko", "inko", "kuch",
    "koi", "sab", "sabhi", "sabko", "aapas", "apna", "apni", "apne",

    # Postpositions & Particles
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "pr", "tak", "dwara",
    "saath", "sath", "bina", "liye", "baad", "pehle", "paas", "pass", "samne",
    "aage", "piche", "andar", "baahar", "beech", "uppar", "neeche",

    # Verbs & Auxiliaries
    "hai", "hain", "ho", "hoon", "hun", "h", "tha", "thi", "gaya", "gayi",
    "gaye", "raha", "rahi", "rahe", "hoga", "hogi", "hoge", "honge", "karna",
    "karni", "karne", "kar", "karo", "karein", "karta", "karti", "karte",
    "kiya", "kiyi", "kiye", "dene", "dena", "diya", "diye", "lene", "lena",
    "liya", "liye", "aana", "aaye", "aaya", "aayi", "jaana", "jaaye", "jaaya",
    "jaayein", "sakta", "sakti", "sakte", "chahiye", "milla", "mili", "mile",

    # Conjunctions & Adverbs
    "aur", "ya", "lekin", "parantoo", "parantu", "toh", "to", "phir", "fir",
    "ab", "jab", "tab", "kabhi", "sada", "hamesha", "aaj", "kal", "parso",
    "subah", "shaam", "raat", "dopahar", "bhi", "hee", "bhot", "bahut",
    "zyada", "ziyada", "kam", "thoda", "thodi", "accha", "acchi", "acche",
    "bura", "buri", "bure", "sahi", "galat", "theek", "haan", "nahin", "nahi",
    "na", "mat", "bas", "sirf", "vaise", "vaisebhi", "jaise", "jaisebhi",
    "bhai", "yaar", "ji", "tu", "baat", "ek", "ye", "ne", "rt", "log", "waley"
}

HINDI_SUFFIX_PATTERNS = (
    r'.*(ege|egi|ega|enge|ungi|unga|oge|ogi|aaye|aaya|aayi|aate|aati|aata|'
    r'iyega|iye|iyeh|ingey|ingee|ingea|rakho|dekho|suno|batao|samjho)$'
)
HINDI_SUFFIX_RE = re.compile(HINDI_SUFFIX_PATTERNS, re.IGNORECASE)

try:
    import wordfreq
except ImportError:
    wordfreq = None


def _is_english_dictionary_word(word_lower: str) -> bool:
    """Check if token is a recognized English word via global corpus frequency."""
    if wordfreq is not None:
        return wordfreq.zipf_frequency(word_lower, "en") >= 2.5
    return False


def _is_hindi_word(word_lower: str) -> bool:
    """Check if token is a transliterated Hindi word using neural LID and lexical fallback."""
    if word_lower in HINDI_MARKERS:
        return True
    # If the word is recognized in the global English lexicon and not in Hindi markers,
    # it is an English candidate, avoiding false positive Hindi classification
    if _is_english_dictionary_word(word_lower) and not HINDI_SUFFIX_RE.match(word_lower):
        return False
    if HINDI_SUFFIX_RE.match(word_lower) and not _is_english_dictionary_word(word_lower):
        return True
    
    # Neural subword character n-gram prediction
    lid = get_neural_lid()
    pred = lid.predict_token_language(word_lower)
    return pred["language"] == "HI"


def tokenize(sentence: str) -> list[dict]:
    """
    Tokenize a code-mixed sentence with noise-resilient neural subword LID.

    Returns tagged tokens:
      - token: raw token string
      - lower: lowercased token string
      - script: 'latin', 'devanagari', or 'punct'
      - language: 'EN', 'HI', 'PUNCT'
      - is_candidate: True if English loanword candidate subject to LAPS scoring
      - p_hi / p_en: Continuous language probability distributions
    """
    raw_tokens = re.findall(r"[\w'\-]+|[^\w\s]", sentence)
    tagged = []
    lid = get_neural_lid()

    for tok in raw_tokens:
        tok_low = tok.lower()

        # Punctuation / Numbers
        if re.fullmatch(r'[\d\W]+', tok):
            tagged.append({
                "token": tok, "lower": tok_low, "script": "punct",
                "language": "PUNCT", "is_candidate": False,
                "p_hi": 0.0, "p_en": 0.0
            })
            continue

        # Devanagari script
        if DEVANAGARI_RE.search(tok):
            tagged.append({
                "token": tok, "lower": tok_low, "script": "devanagari",
                "language": "HI", "is_candidate": False,
                "p_hi": 1.0, "p_en": 0.0
            })
            continue

        # Neural subword LID prediction
        pred = lid.predict_token_language(tok)
        
        # Check dictionary overrides for ambiguous cases
        if _is_english_dictionary_word(tok_low) and tok_low not in HINDI_MARKERS:
            lang = "EN"
            is_cand = tok_low not in AMBIGUOUS_HOMOGRAPHS
            p_en = 0.95
            p_hi = 0.05
        elif tok_low in HINDI_MARKERS or (HINDI_SUFFIX_RE.match(tok_low) and not _is_english_dictionary_word(tok_low)):
            lang = "HI"
            is_cand = False
            p_hi = 0.95
            p_en = 0.05
        else:
            lang = pred["language"]
            is_cand = pred["is_candidate"]
            p_hi = pred["p_hi"]
            p_en = pred["p_en"]

        tagged.append({
            "token": tok,
            "lower": tok_low,
            "script": "latin",
            "language": lang,
            "is_candidate": is_cand,
            "p_hi": p_hi,
            "p_en": p_en,
            "normalized": pred.get("normalized", tok_low)
        })

    # Contextual sequence-level homograph disambiguation
    tagged = lid.disambiguate_homographs(tagged)

    return tagged



if __name__ == "__main__":
    test_cases = [
        "hi! Kisi ke pass viva calender ki link h kya?",
        "bhai smjh ni aya kal ka ppt share kr dena plzz",
        "yeh client meeting postpone ho gayi hai"
    ]
    for sample in test_cases:
        print(f"\nSentence: {sample}")
        for t in tokenize(sample):
            print(f"  {t['token']:12s} -> lang:{t['language']:5s} cand:{str(t['is_candidate']):5s} P(HI):{t['p_hi']:.2f} P(EN):{t['p_en']:.2f}")
