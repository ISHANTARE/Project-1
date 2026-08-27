"""
LAPS PPT Slide Generator
------------------------
Programmatically populates the official VIT Review-2 PPT template
with LAPS project details using python-pptx.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

TEMPLATE_PATH = r"c:\Users\Admin\VIT_Projects\PROJECT-1\Review-2-guidelines\4 Review 2 PPT Template.pptx"
OUTPUT_PATH   = r"c:\Users\Admin\VIT_Projects\PROJECT-1\Review-2\LAPS_Review_2_Presentation.pptx"


def build_presentation():
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template not found at {TEMPLATE_PATH}")
        return

    prs = Presentation(TEMPLATE_PATH)
    print(f"Loaded template with {len(prs.slides)} slides.")

    # ── Slide Data Mapping ────────────────────────────────────────────────────
    slide_content = {
        0: {
            "title": "BCSE497J – Project-I\nLAPS: Linguistic Adaptive Preservation Scoring for Explainable Token-Level Decision Making in Code-Mixed Text Rewriting",
            "body": "Student Name: [Your Name]\nRegister No: [Your Reg No]\nCourse: B.Tech Computer Science and Engineering\n\nFaculty Guide: [Guide Name]\nSchool of Computer Science and Engineering\nSeptember 2026",
        },
        1: {
            "title": "Approval Mail From Guide",
            "body": "Project Approval Status:\n✔ Project Title: LAPS - Linguistic Adaptive Preservation Scoring\n✔ Objectives & Methodology Approved\n✔ Review 2 Presentation & Report Approved by Faculty Guide\n\nDate of Guide Approval: ____ / ____ / 2026",
        },
        2: {
            "title": "Aim",
            "body": "To design and implement an explainable, token-level pre-generation decision framework (LAPS) that prevents over-translation in code-mixed text rewriting by adaptively scoring loanword preservation based on linguistic features and communication context.",
        },
        3: {
            "title": "Abstract",
            "body": "Code-mixed communication (Hinglish/Tanglish) is widespread across digital platforms. Multilingual LLMs frequently over-translate natural borrowed words ('meeting', 'client', 'server'), destroying naturalness. LAPS introduces a pre-generation decision layer evaluating tokens across 6 linguistic vectors (BF, DS, SI, CC, AF, TB) against dynamic context thresholds. It guides LLMs with explicit constraints while generating transparent token-level rationales.",
        },
        4: {
            "title": "Literature Review",
            "body": "1. Mave et al. (2018) – Language ID in code-switched text using BiLSTMs (identifies origin, but lacks preservation scoring).\n2. Post & Vilar (2018) – Grid Beam Search for NMT lexical constraints (requires hardcoded static glossaries).\n3. Srivastava et al. (2024) – Benchmarked LLMs on Hinglish normalization, identifying severe over-translation of loanwords without prompt constraints.",
        },
        5: {
            "title": "Research Gap",
            "body": "• Existing systems rely on end-to-end black-box LLMs that lack explainability.\n• Traditional machine translation translates all words indiscriminately, causing over-translation.\n• Static lookup rules cannot dynamically adjust to different communication registers (casual chat vs. executive report).",
        },
        6: {
            "title": "Objectives & Expected Outcomes",
            "body": "Objectives:\n1. Isolate code-mixed candidate tokens via language identification.\n2. Quantify loanword preservation via 6-vector feature fusion.\n3. Implement register-aware dynamic thresholding.\n4. Guide LLMs via constrained prompt generation.\n\nIdentified SDG: SDG 9 (Industry, Innovation & Infrastructure) & SDG 4 (Quality Education)\nExpected Outcomes: Scopus/IEEE Conference Paper + Provisional Patent Draft",
        },
        7: {
            "title": "Framework / Architecture",
            "body": "Pipeline Architecture:\nInput Text → Tokenizer & LID → Feature Extractor (BF, DS, SI, CC, AF, TB) → LAPS Scorer Engine (Score = Σ w_i * F_i) → Adaptive Threshold Engine (Register θ) → Constrained Prompt Builder → LLM Generator → Rewritten Output + Token Explanation Table",
        },
        8: {
            "title": "Functional Requirements",
            "body": "FR-1: Token-level language tagger (EN/HI/MX).\nFR-2: 6-vector feature scoring engine.\nFR-3: Weighted LAPS preservation score calculator.\nFR-4: Context register threshold engine (Casual, Business, Academic, Social).\nFR-5: Constrained LLM prompt builder & rewriter.\nFR-6: Human-readable explanation generator & CSV export.",
        },
        9: {
            "title": "Modules Implemented (60% - 75%)",
            "body": "Completed Modules:\n✔ src/laps/tokenizer.py (Tokenizer & Script/Language ID)\n✔ src/laps/feature_extractor.py (6 Feature Extractor)\n✔ src/laps/scorer.py (Weighted Scorer Engine)\n✔ src/laps/threshold_engine.py (Adaptive Register Engine)\n✔ src/llm/rewriter.py & explainer.py (Constrained LLM Client & Explainer)\n✔ app.py (Interactive Streamlit Demo Dashboard at http://localhost:8501)",
        },
        10: {
            "title": "Experiments and Demonstration",
            "body": "Demonstration Highlights:\n• Live interactive analysis of Hinglish inputs (e.g., 'Kal client meeting postpone ho gayi...').\n• Preserved tokens ('client', 'meeting') kept intact while Hindi verbs are rewritten.\n• Real-time Radar & Bar charts visualizing feature scores per token.\n• Automated export of token-by-token explanation reports (CSV).",
        },
        11: {
            "title": "Conclusion & Future Work",
            "body": "Conclusion:\n• Successfully implemented 75% of the core LAPS system with working UI.\n• Proved that pre-generation scoring effectively mitigates over-translation.\n\nFuture Work:\n• Fine-tune weights via Bayesian Optimization on LinCE/GLUECoS datasets.\n• Conduct full human preference evaluation surveys.",
        },
        12: {
            "title": "References",
            "body": "[1] M. Mave et al., 'Language Identification and Analysis of Code-Switched Data,' NAACL-HLT, 2018.\n[2] M. Post & D. Vilar, 'Fast Lexically Constrained Decoding with Grid Beam Search,' NAACL-HLT, 2018.\n[3] P. Srivastava et al., 'Evaluating Large Language Models on Hinglish Code-Mixed Text Normalization,' IEEE Access, 2024.\n[4] H. Tanwar et al., 'Benchmarking Multilingual LLMs on Code-Mixed South Asian Languages,' EMNLP, 2023.",
        },
    }

    # ── Apply to slides ───────────────────────────────────────────────────────
    for slide_idx, content in slide_content.items():
        if slide_idx < len(prs.slides):
            slide = prs.slides[slide_idx]

            # Update shapes that contain text
            text_shapes = [s for s in slide.shapes if s.has_text_frame]
            if text_shapes:
                # First shape is usually title
                if len(text_shapes) >= 1 and "title" in content:
                    tf = text_shapes[0].text_frame
                    tf.text = content["title"]

                # Second shape is body/content
                if len(text_shapes) >= 2 and "body" in content:
                    tf2 = text_shapes[1].text_frame
                    tf2.text = content["body"]
                elif len(text_shapes) == 1 and "body" in content:
                    # Add body text to existing frame as new paragraph
                    tf = text_shapes[0].text_frame
                    p = tf.add_paragraph()
                    p.text = "\n" + content["body"]

    prs.save(OUTPUT_PATH)
    print(f"Successfully saved generated presentation to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_presentation()
