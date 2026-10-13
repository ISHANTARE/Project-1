"""
collect_real_data.py — Annotation Template Generator for LAPS Evaluation
--------------------------------------------------------------------------
This script generates a CSV annotation template from a set of Hinglish sentences.
The template is designed to be filled in by 2-3 independent human annotators.

WORKFLOW:
  1. Add your Hinglish sentences to SENTENCES_TO_ANNOTATE below.
  2. Run this script to generate data/evaluation/annotations.csv.
  3. Share the CSV with your annotators (e.g., via Google Sheets).
  4. Each annotator fills in their column: 1 = Preserve, 0 = Translate.
  5. Run: python scripts/compute_agreement.py --annotations data/evaluation/annotations.csv

SOURCES FOR REAL HINGLISH SENTENCES:
  • Your own WhatsApp/Telegram group chats (with participants' permission)
  • LINCE Shared Task dataset (code-switched NER + LID, open access):
      https://ritual.uh.edu/lince/
  • HuggingFace: 'lince_codemix' dataset (Hindi-English code-mixed tweets)
  • Twitter/X academic API — search for Hinglish hashtags
  • GLUECoS benchmark — code-mixed NLI/SA tasks

HOW TO ANNOTATE:
  For each candidate token (English word in a Hinglish sentence):
    1 = PRESERVE — Keep this English word as-is in the translated output.
        (It's a loanword, technical term, or code-mixed word that should not be translated)
    0 = TRANSLATE — This English word should be translated to Hindi/Devanagari equivalent.
        (It's a common word that has a natural Hindi alternative)

  Example:
    Sentence: "Kal client meeting postpone ho gayi"
    Token: "client"  → 1 (PRESERVE — "client" is a domain-specific loanword)
    Token: "meeting" → 1 (PRESERVE — standard business term, commonly retained)
    Token: "postpone"→ 0 (TRANSLATE — "sthagit" or similar exists in Hindi)
"""

import csv
import re
from pathlib import Path

# ============================================================
# ADD YOUR REAL HINGLISH SENTENCES HERE
# ============================================================
SENTENCES_TO_ANNOTATE = [
    # --- ACADEMIC DOMAIN ---
    "Kisi ke paas viva ka link h kya?",
    "Assignment submit karne ki deadline kal hai, bhai",
    "Professor ne syllabus update kar diya hai, check kar lena",
    "Aaj lecture cancel ho gayi kya?",
    "Presentation ka feedback kal milega",

    # --- PROFESSIONAL / WORKPLACE ---
    "Kal client meeting postpone ho gayi hai, kya hum schedule fix kar sakte hain?",
    "Bhai server down hai, database backup lena padega",
    "Manager ne project ka proposal bheja hai, review karo",
    "Contract sign karna hai, kab free ho?",
    "Deploy se pehle code review karna mat bhoolo",

    # --- CASUAL / SOCIAL ---
    "Aaj party mein kaun aa raha hai? Confirm karo na",
    "Bhai, download speed bahut slow hai aaj",
    "WhatsApp mein status update kiya kya?",
    "YouTube par video upload ho gayi?",
    "Instagram pe photo post kar de, tag bhi kar dena",

    # --- TECHNICAL ---
    "Git pull karo pehle, fir merge karna",
    "Bug fix ho gayi hai, PR raise kar do",
    "API ka response timeout ho raha hai",
    "Frontend aur backend sync nahi ho raha",
    "Cache clear karo, phir try karo",

    # ADD MORE SENTENCES BELOW (aim for 50 total)
    # "...",
]

# Common English tokens in Hinglish that are candidates for annotation
# (The script auto-detects these; add more patterns if needed)
ENGLISH_WORD_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')

# Words that are almost certainly Hindi/Urdu grammatical particles
# (not worth annotating — they are not code-mixed loanwords)
HINDI_PARTICLES = {
    "hai", "hain", "ho", "kya", "ke", "ka", "ki", "ko", "me", "mein",
    "kisi", "ke", "paas", "ne", "aur", "par", "pe", "se", "bhi",
    "nahi", "kal", "aaj", "karo", "kar", "do", "le", "de",
    "mat", "phir", "bahut", "na", "bhai", "yaar", "mera", "tera",
    "hum", "aap", "tum", "woh", "iska", "uska",
}


def extract_candidate_tokens(sentence: str) -> list[str]:
    """Extract English candidate tokens from a Hinglish sentence."""
    tokens = ENGLISH_WORD_PATTERN.findall(sentence)
    seen = set()
    candidates = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower not in HINDI_PARTICLES and t_lower not in seen:
            candidates.append(t_lower)
            seen.add(t_lower)
    return candidates


def generate_annotation_template(output_path: Path, n_raters: int = 2) -> None:
    """Generate the annotation CSV template."""
    rater_cols = [f"rater_{i+1}" for i in range(n_raters)]
    fieldnames = ["sentence_id", "sentence", "token"] + rater_cols + ["notes"]

    rows = []
    for i, sentence in enumerate(SENTENCES_TO_ANNOTATE, start=1):
        sid = f"S{i:03d}"
        candidates = extract_candidate_tokens(sentence)
        if not candidates:
            continue
        for token in candidates:
            row = {
                "sentence_id": sid,
                "sentence": sentence,
                "token": token,
                "notes": "",
            }
            for rc in rater_cols:
                row[rc] = ""  # Left blank for human annotators to fill
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Annotation template generated: {output_path}")
    print(f"  Sentences: {len(SENTENCES_TO_ANNOTATE)}")
    print(f"  Candidate tokens: {len(rows)}")
    print(f"  Rater columns: {', '.join(rater_cols)}")
    print("\nINSTRUCTIONS:")
    print("  1. Open the CSV in Excel or Google Sheets.")
    print("  2. Each rater fills in their column: 1 = Preserve, 0 = Translate.")
    print("  3. Raters must work INDEPENDENTLY (do not share answers).")
    print("  4. After both raters finish, run:")
    print(f"       python scripts/compute_agreement.py --annotations {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate annotation template CSV for LAPS evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/annotations.csv"),
        help="Output path for the annotation CSV template."
    )
    parser.add_argument(
        "--raters",
        type=int,
        default=2,
        help="Number of independent human raters (default: 2)."
    )
    args = parser.parse_args()
    generate_annotation_template(args.output, args.raters)
