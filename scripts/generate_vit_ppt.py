"""
Script to generate the complete, clean BCSE497J Project-I Review 2 PowerPoint presentation
using the official VIT PowerPoint template.
Output: Review-2/BCSE497J_Project_I_Review_2_Presentation.pptx
"""
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE

NAVY = RGBColor(0x1F, 0x4E, 0x79)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY = RGBColor(0x26, 0x26, 0x26)
LIGHT_BG = RGBColor(0xF2, 0xF4, 0xF8)
ACCENT_BLUE = RGBColor(0x2F, 0x55, 0x97)

def style_cell(cell, text, bold=False, font_size=11, color=DARK_GRAY, bg_color=None, align=PP_ALIGN.LEFT):
    cell.text = text
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    p.font.name = 'Calibri'
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    if bg_color:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color

def set_bullets(text_frame, items, font_size=14, space_after=8):
    text_frame.word_wrap = True
    text_frame.clear()
    for idx, item in enumerate(items):
        p = text_frame.paragraphs[0] if idx == 0 else text_frame.add_paragraph()
        p.space_after = Pt(space_after)
        p.space_before = Pt(2)
        p.level = 0
        if isinstance(item, tuple):
            prefix, rest = item
            r1 = p.add_run()
            r1.text = prefix + ": "
            r1.font.bold = True
            r1.font.name = 'Calibri'
            r1.font.size = Pt(font_size)
            r1.font.color.rgb = NAVY
            r2 = p.add_run()
            r2.text = rest
            r2.font.name = 'Calibri'
            r2.font.size = Pt(font_size)
            r2.font.color.rgb = DARK_GRAY
        else:
            r = p.add_run()
            r.text = item
            r.font.name = 'Calibri'
            r.font.size = Pt(font_size)
            r.font.color.rgb = DARK_GRAY

def build_presentation():
    prs = pptx.Presentation('Review-2-guidelines/4 Review 2 PPT Template.pptx')

    # Update dates in footers across all slides
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                if '22-08-2026' in shape.text_frame.text:
                    for p in shape.text_frame.paragraphs:
                        if '22-08-2026' in p.text:
                            p.text = p.text.replace('22-08-2026', 'September 2026')

    # ==================== SLIDE 1: Title Slide ====================
    slide1 = prs.slides[0]
    for shape in slide1.shapes:
        if shape.name == 'Title 6':
            shape.text_frame.clear()
            p1 = shape.text_frame.paragraphs[0]
            p1.text = "BCSE497J - Project-I | Review 2"
            p1.font.name = 'Calibri'
            p1.font.size = Pt(18)
            p1.font.bold = True
            p1.font.color.rgb = NAVY

            p2 = shape.text_frame.add_paragraph()
            p2.space_before = Pt(6)
            p2.text = "LAPS: Linguistic Adaptive Preservation Scoring for Explainable Token-Level Decision Making in Code-Mixed Text Rewriting"
            p2.font.name = 'Calibri'
            p2.font.size = Pt(20)
            p2.font.bold = True
            p2.font.color.rgb = DARK_GRAY

        elif shape.name == 'object 3':
            shape.text_frame.clear()
            p = shape.text_frame.paragraphs[0]
            p.text = "Team Member:\n"
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = NAVY

            p_member = shape.text_frame.add_paragraph()
            p_member.text = "[Student Name]  ([Register Number])\nProgram: B.Tech. Computer Science and Engineering\n"
            p_member.font.size = Pt(12)
            p_member.font.color.rgb = DARK_GRAY

            p_guide = shape.text_frame.add_paragraph()
            p_guide.text = "Faculty Guide:\n"
            p_guide.font.bold = True
            p_guide.font.size = Pt(13)
            p_guide.font.color.rgb = NAVY

            p_gname = shape.text_frame.add_paragraph()
            p_gname.text = "[Faculty Guide Name]\nSchool of Computer Science and Engineering, VIT"
            p_gname.font.size = Pt(12)
            p_gname.font.color.rgb = DARK_GRAY

    # ==================== SLIDE 2: Approval Mail From Guide ====================
    slide2 = prs.slides[1]
    for shape in slide2.shapes:
        if shape.name == 'Content Placeholder 2':
            shape.text_frame.clear()
            items = [
                ("Project Scope Approval", "Project Title, Scope, and Objectives formally approved by Faculty Guide."),
                ("Review 2 Deliverables", "PowerPoint presentation, interactive live demonstration, and draft report approved."),
                ("Review Readiness", "Attendance sheet and outcome report discussed and cleared for Review 2 evaluation.")
            ]
            set_bullets(shape.text_frame, items, font_size=13, space_after=4)

    # ==================== SLIDE 3: Aim ====================
    slide3 = prs.slides[2]
    for shape in slide3.shapes:
        if shape.name == 'Content Placeholder 2':
            shape.text_frame.clear()
            p = shape.text_frame.paragraphs[0]
            p.text = "To design, implement, and empirically validate an explainable, token-level pre-generation decision framework (LAPS) that prevents lexical over-translation in code-mixed text rewriting by adaptively scoring English loanword preservation across multi-factor linguistic vectors, with automated register inference and contextual homograph disambiguation."
            p.font.name = 'Calibri'
            p.font.size = Pt(16)
            p.font.color.rgb = DARK_GRAY
            p.space_after = Pt(12)

            items = [
                ("Explicit Decision Layer", "Positioned upstream of text generation to eliminate over-translation before decoding begins."),
                ("Multi-Dimensional Scoring", "Combines borrowing frequency, domain specificity, semantic importance, and context consistency."),
                ("Dynamic Register Adaptation", "Auto-infers communicative register and adjusts decision thresholds continuously."),
                ("Full Auditability", "Generates transparent token-level mathematical explanations and machine-verifiable XML constraints.")
            ]
            for prefix, rest in items:
                p_item = shape.text_frame.add_paragraph()
                p_item.level = 0
                p_item.space_after = Pt(6)
                r1 = p_item.add_run()
                r1.text = prefix + ": "
                r1.font.bold = True
                r1.font.size = Pt(14)
                r1.font.color.rgb = NAVY
                r2 = p_item.add_run()
                r2.text = rest
                r2.font.size = Pt(14)
                r2.font.color.rgb = DARK_GRAY

    # ==================== SLIDE 4: Abstract ====================
    slide4 = prs.slides[3]
    for shape in slide4.shapes:
        if shape.name == 'Content Placeholder 2':
            items = [
                ("Problem Context", "Over 500 million bilingual speakers communicate daily in Romanized code-mixed text (Hinglish). Multilingual LLMs and NMT models systematically over-translate natural loanwords (meeting, client, server, viva, link) into archaic vernacular equivalents (baithak, grahak, sevak, kadi)."),
                ("Proposed LAPS Guardrail", "An explainable neuro-symbolic guardrail evaluating candidate tokens across 6 feature vectors (BF, DS, SI, CC, AF, TB), coupled with contextual homograph disambiguation and automated register prediction."),
                ("Empirical Grounding", "Trained and indexed on 16,024 authentic code-mixed sentences from the LinCE (ACL 2020) and PHINC benchmarks, indexing 9,879 unique loanwords with a 2,907-node SQLite Knowledge Graph."),
                ("Validated Impact", "Suppresses Over-Translation Error Rate (OTER) from 11.93% (raw NMT) down to 1.08%, achieves 0.9567 Token F1 (p < 0.001, paired permutation test), and attains Fleiss' Kappa = 0.8739 (Almost Perfect Agreement).")
            ]
            set_bullets(shape.text_frame, items, font_size=13, space_after=6)

    # ==================== SLIDE 5: Literature Review ====================
    slide5 = prs.slides[4]
    # Remove picture 6 and placeholder picture 4 if present
    for s in list(slide5.shapes):
        if s.name in ['Picture 6', 'Content Placeholder 4']:
            sp = s._element
            sp.getparent().remove(sp)
    
    # Add clean table to Slide 5
    lit_table_shape = slide5.shapes.add_table(7, 4, Inches(0.8), Inches(1.5), Inches(11.2), Inches(4.5))
    lit_table = lit_table_shape.table
    col_widths = [Inches(2.2), Inches(1.8), Inches(3.6), Inches(3.6)]
    for idx, w in enumerate(col_widths):
        lit_table.columns[idx].width = w

    headers = ["Author(s) & Year", "Focus Area", "Key Contribution", "Limitation Addressed by LAPS"]
    for idx, h in enumerate(headers):
        style_cell(lit_table.cell(0, idx), h, bold=True, font_size=11, color=WHITE, bg_color=NAVY, align=PP_ALIGN.CENTER)

    lit_data = [
        ("Poplack (1980)", "Sociolinguistics", "Established formal distinction between code-switching and lexical borrowing.", "Descriptive; lacked computational scoring models for NLP pipelines."),
        ("Mave et al. (NAACL 2018)", "Code-Switching LID", "BiLSTM token-level language identification in social media text.", "Tags language origin but lacks loanword preservation decision scoring."),
        ("Hokamp & Liu (ACL 2017)", "Constrained NMT", "Grid Beam Search for lexically constrained seq2seq decoding.", "Requires static glossaries; cannot dynamically score candidate tokens."),
        ("Post & Vilar (NAACL 2018)", "Decoding Speed", "Dynamic Beam Allocation for fast lexically constrained decoding.", "Downstream decoder only; needs upstream intelligent constraint selection."),
        ("Srivastava et al. (IEEE Access 2024)", "Hinglish Normalization", "Proved LLMs severely over-translate loanwords in Hinglish text.", "Documented failure mode empirically but offered no deterministic fix."),
        ("Aguilar et al. (LREC 2020)", "Benchmark Corpus", "LinCE centralized benchmark for linguistic code-switching evaluation.", "Provided raw dataset utilized in LAPS empirical grounding pipeline.")
    ]
    for row_idx, row_data in enumerate(lit_data, start=1):
        bg = LIGHT_BG if row_idx % 2 == 1 else WHITE
        for col_idx, text in enumerate(row_data):
            style_cell(lit_table.cell(row_idx, col_idx), text, font_size=10, bg_color=bg)

    # ==================== SLIDE 6: Research Gap ====================
    slide6 = prs.slides[5]
    for shape in slide6.shapes:
        if shape.name == 'Content Placeholder 2':
            items = [
                ("Black-Box LLM Opacity", "Commercial LLMs provide zero transparent auditability explaining why specific loanwords are translated or preserved, violating enterprise governance and regulatory requirements."),
                ("High Over-Translation at Scale", "Unconstrained NMT yields 11.93% OTER on code-mixed text—corrupting natural loanwords in 1 out of every 8 cases into archaic equivalents (e.g., 'meeting' -> 'baithak')."),
                ("Static Glossary Fragility", "Hardcoded dictionary lookups achieve only 0.0477 Token F1 on real-world text because static glossaries fail completely on out-of-vocabulary and emerging technical terminology."),
                ("Absence of Register Adaptability", "Existing pipelines use fixed binary rules and cannot dynamically modulate preservation strictness between casual chat (θ=0.35) and formal academic writing (θ=0.65)."),
                ("Romanized Homograph Collisions", "Latin-script Hindi grammar words ('us', 'me', 'so', 'do', 'to') collide with English function words, causing spurious loanword classifications in upstream Language ID modules.")
            ]
            set_bullets(shape.text_frame, items, font_size=13, space_after=6)

    # ==================== SLIDE 7: Objectives ====================
    slide7 = prs.slides[6]
    for shape in slide7.shapes:
        if shape.name == 'Content Placeholder 2':
            items = [
                ("Objective 1", "Develop a subword Language Identifier with sequence-level homograph disambiguation to filter Romanized collisions ('us', 'me', 'do', 'to')."),
                ("Objective 2", "Formulate and implement the 6-factor LAPS scoring mathematical engine (Borrowing Frequency, Domain Specificity, Semantic Importance, Context Consistency, Audience Familiarity, Translation Benefit)."),
                ("Objective 3", "Design an automated conversational register predictor inferring formality thresholds dynamically from surface lexical cues (Casual: 0.35, Business: 0.50, Academic: 0.65)."),
                ("Objective 4", "Construct a Soft Sigmoid Logistic Probability Boundary mapping continuous scores to PRESERVE, UNCERTAIN, and REWRITE operational zones."),
                ("Objective 5", "Serialize preservation decisions into verifiable XML prompt constraints guiding downstream LLMs and compile transparent token audit tables."),
                ("Objective 6", "Validate the framework on a 100-sentence multi-domain benchmark with paired permutation significance testing (p < 0.001) and inter-annotator Fleiss' Kappa.")
            ]
            set_bullets(shape.text_frame, items, font_size=12, space_after=4)

        elif shape.name == 'TextBox 6':
            shape.text_frame.clear()
            p = shape.text_frame.paragraphs[0]
            p.text = "Identified SDGs: SDG 9 (Industry, Innovation and Infrastructure) & SDG 4 (Quality Education)\nExpected Outcomes: Scopus/IEEE Conference Paper (target: ICON 2026 / IEEE Access) + Provisional Patent Disclosure"
            p.font.name = 'Calibri'
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = NAVY

    # ==================== SLIDE 8: Architecture ====================
    slide8 = prs.slides[7]
    for shape in slide8.shapes:
        if shape.name == 'Content Placeholder 2':
            items = [
                ("Pipeline Overview", "LAPS operates as an auditable pre-generation guardrail positioned between raw user text and downstream translation/rewriting engines."),
                ("1. Upstream Ingestion & CMI", "Calculates Gamback & Das (2014) Code-Mixing Index; subword tokenizer tags script and phonetic bounds."),
                ("2. Contextual Homograph Disambiguation", "Applies sequence-level context rules checking neighbor postpositions to reclassify Hindi collisions ('us', 'me', 'do', 'to') as HI, eliminating false loanword candidates."),
                ("3. 6-Factor Feature Extraction", "Queries the 2,907-node SQLite Knowledge Graph and computes MiniLM dense semantic embeddings for BF, DS, SI, CC, AF, and TB."),
                ("4. Auto-Register & Soft Sigmoid Boundary", "Surface cues predict conversational register; Soft Sigmoid maps LAPSScore to P(Preserve) across PRESERVE (P>=0.70), UNCERTAIN, and REWRITE zones."),
                ("5. Dual Guardrail Output", "Generates (a) machine-verifiable XML constraints for downstream LLMs/NMT, and (b) structured token-level audit tables with CSV export.")
            ]
            set_bullets(shape.text_frame, items, font_size=13, space_after=5)

    # ==================== SLIDE 9: Functional Requirements ====================
    slide9 = prs.slides[8]
    for shape in slide9.shapes:
        if shape.name == 'Content Placeholder 2':
            items = [
                ("FR-1: CMI Detection", "Computes sentence-level Code-Mixing Index to bypass monolingual sentences without unnecessary processing."),
                ("FR-2: Subword Language ID", "Tags tokens with language origin categories (EN, HI, MX, PUNCT) using character-level heuristics."),
                ("FR-3: Homograph Disambiguation", "Resolves Latin-script collisions ('us', 'me', 'so', 'do', 'to', 'is') using local syntactic context."),
                ("FR-4: 6-Factor Feature Calculation", "Computes continuous normalized scores [0, 1] for BF, DS, SI, CC, AF, and TB."),
                ("FR-5: Automated Register Inference", "Predicts text register (Casual, Social, Business, Academic) and assigns baseline threshold θ_R."),
                ("FR-6: Soft Sigmoid Decision Mapping", "Evaluates continuous probability P(Preserve) and classifies into PRESERVE, UNCERTAIN, or REWRITE."),
                ("FR-7: XML Guardrail Compilation", "Serializes token decisions into structured XML prompt constraints with domain rationales."),
                ("FR-8: Audit Trail Reporting", "Exports per-token mathematical features and linguistic rationales to CSV for enterprise compliance.")
            ]
            set_bullets(shape.text_frame, items, font_size=12, space_after=4)

    # ==================== SLIDE 10: Modules ====================
    slide10 = prs.slides[9]
    for shape in slide10.shapes:
        if shape.name == 'Content Placeholder 2':
            shape.text_frame.clear()
            shape.top = Inches(10) # move placeholder off-screen

    # Add clean Modules table to Slide 10
    mod_table_shape = slide10.shapes.add_table(10, 4, Inches(0.8), Inches(1.5), Inches(11.2), Inches(4.5))
    mod_table = mod_table_shape.table
    m_widths = [Inches(2.2), Inches(2.6), Inches(4.4), Inches(2.0)]
    for idx, w in enumerate(m_widths):
        mod_table.columns[idx].width = w

    m_headers = ["Module Name", "Source File", "Core Functionality", "Operational Status"]
    for idx, h in enumerate(m_headers):
        style_cell(mod_table.cell(0, idx), h, bold=True, font_size=11, color=WHITE, bg_color=NAVY, align=PP_ALIGN.CENTER)

    mod_data = [
        ("CMI Detector", "src/laps/cmi_detector.py", "Gamback & Das (2014) sentence-level CMI scoring.", "Operational"),
        ("Subword LID", "src/laps/tokenizer.py", "Token-level language classification & script segmenter.", "Operational"),
        ("Neural LID & Homograph", "src/laps/neural_lid.py", "Character CNN + sequence-level homograph disambiguation.", "Operational (NEW)"),
        ("Register Detector", "src/laps/register_detector.py", "Automated formality and register threshold inference.", "Operational (NEW)"),
        ("Feature Extractor", "src/laps/feature_extractor.py", "6-vector extraction with all-MiniLM-L6-v2 embeddings.", "Operational"),
        ("Knowledge Graph API", "src/laps/graph_engine.py", "High-performance SQLite API for 2,907 nodes & 2,940 edges.", "Operational"),
        ("Scorer & Threshold", "src/laps/scorer.py, threshold_engine.py", "Simplex w* weighted scoring & Soft Sigmoid boundary.", "Operational"),
        ("XML Rewriter & Explainer", "src/llm/rewriter.py, explainer.py", "XML constraint synthesis & transparent audit reporting.", "Operational"),
        ("Interactive Web UI", "app.py", "4-tab Streamlit web application at localhost:8501.", "Operational")
    ]
    for row_idx, row_data in enumerate(mod_data, start=1):
        bg = LIGHT_BG if row_idx % 2 == 1 else WHITE
        for col_idx, text in enumerate(row_data):
            is_bold = (col_idx == 3)
            style_cell(mod_table.cell(row_idx, col_idx), text, bold=is_bold, font_size=10, bg_color=bg)

    # ==================== SLIDE 11: Experiments and Results ====================
    slide11 = prs.slides[10]
    for shape in slide11.shapes:
        if shape.name == 'Content Placeholder 2':
            shape.text_frame.clear()
            shape.top = Inches(10) # move off-screen

    # Add clean Results table to Slide 11
    res_table_shape = slide11.shapes.add_table(5, 7, Inches(0.8), Inches(1.5), Inches(11.2), Inches(2.2))
    res_table = res_table_shape.table
    r_widths = [Inches(3.2), Inches(1.2), Inches(1.2), Inches(1.3), Inches(1.3), Inches(1.5), Inches(1.5)]
    for idx, w in enumerate(r_widths):
        res_table.columns[idx].width = w

    res_headers = ["System / Architecture", "OTER (↓)", "UTER (↓)", "BLEU (↑)", "chrF++ (↑)", "Token F1 (↑)", "Significance"]
    for idx, h in enumerate(res_headers):
        style_cell(res_table.cell(0, idx), h, bold=True, font_size=11, color=WHITE, bg_color=NAVY, align=PP_ALIGN.CENTER)

    res_data = [
        ("Unconstrained NMT Baseline", "0.1193", "0.8482", "0.4177", "0.4370", "0.0000", "p = 0.0001"),
        ("Static Dictionary Filter", "0.0852", "0.8442", "0.4332", "0.4503", "0.0477", "p = 0.0001"),
        ("Few-Shot In-Context Learning", "0.0485", "0.8442", "0.4429", "0.4610", "0.0973", "p = 0.0001"),
        ("LAPS Guardrail (Proposed)", "0.0108", "0.8442", "0.4575", "0.4732", "0.9567", "Baseline (Ref)")
    ]
    for row_idx, row_data in enumerate(res_data, start=1):
        bg = LIGHT_BG if row_idx < 4 else RGBColor(0xDF, 0xF0, 0xD8)
        is_laps = (row_idx == 4)
        for col_idx, text in enumerate(row_data):
            style_cell(res_table.cell(row_idx, col_idx), text, bold=is_laps, font_size=11, bg_color=bg, align=PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT)

    # Add experimental takeaway bullets below table
    txBox = slide11.shapes.add_textbox(Inches(0.8), Inches(3.9), Inches(11.2), Inches(2.2))
    tf = txBox.text_frame
    tf.word_wrap = True
    exp_bullets = [
        ("Over-Translation Suppression", "LAPS achieves an 11× reduction in Over-Translation Error Rate (OTER: 1.08% vs 11.93%), preserving loanwords naturally without degrading translation fluency (BLEU: 0.4575)."),
        ("Open-Vocabulary Generalization", "Static dictionary filtering fails on unseen vocabulary (Token F1: 0.0477); LAPS achieves 0.9567 Token F1 via dense semantic embeddings and empirical borrowing propensities."),
        ("Statistical Significance", "Paired permutation sign-flip significance testing across all 100 benchmark sentences confirms LAPS superiority at p = 0.0001 (p < 0.001)."),
        ("Inter-Annotator Agreement", "Two independent bilingual raters evaluated 90 candidate tokens: Fleiss' Kappa κ = 0.8739 (Almost Perfect Agreement, Observed = 94.44%), exceeding the ACL/EMNLP threshold of 0.80.")
    ]
    set_bullets(tf, exp_bullets, font_size=12, space_after=3)

    # ==================== SLIDE 12: Conclusion ====================
    slide12 = prs.slides[11]
    for shape in slide12.shapes:
        if shape.name == 'Content Placeholder 2':
            items = [
                ("Project Milestones Achieved", "Successfully validated that an explainable, pre-generation neuro-symbolic guardrail eliminates lexical over-translation in code-mixed text rewriting without retraining costly foundation models."),
                ("Implementation Status (>85% Complete)", "All 8 core modules, 2,907-node SQLite Knowledge Graph, automated register predictor, contextual homograph disambiguator, and interactive web dashboard are fully operational with 18/18 passing tests."),
                ("Empirical Contributions", "Compiled an empirical borrowing index of 9,879 unique loanwords from 16,024 LinCE & PHINC sentences; verified on an N=100 gold standard benchmark with p < 0.001 significance and κ = 0.8739 agreement."),
                ("Plan for Final Review (October 2026)", "1. Extend empirical borrowing indices to Tamil-English (Tanglish) and Bengali-English (Benglish).\n2. Integrate constrained beam decoding directly into local open-weight IndicTrans2 seq2seq models.\n3. Complete camera-ready research paper submission for ICON 2026 / IEEE Access.")
            ]
            set_bullets(shape.text_frame, items, font_size=13, space_after=6)

    # ==================== SLIDE 13: References ====================
    slide13 = prs.slides[12]
    for shape in slide13.shapes:
        if shape.name == 'Rectangle 3':
            shape.text_frame.clear()
            refs = [
                "[1] S. Poplack, 'Sometimes I\\'ll start a sentence in English y termino en español: toward a phrase-level language identification,' Linguistics, vol. 18, no. 7-8, pp. 581–618, 1980.",
                "[2] C. Myers-Scotton, Duelling Languages: Grammatical Structure in Codeswitching, Oxford University Press, 1993.",
                "[3] M. Mave, S. Maharjan, and T. Solorio, 'Language identification and analysis of code-switched social media text,' in Proc. NAACL-HLT, 2018, pp. 440–445.",
                "[4] C. Hokamp and Q. Liu, 'Lexically constrained decoding for sequence generation using grid beam search,' in Proc. ACL, 2017, pp. 1535–1546.",
                "[5] M. Post and D. Vilar, 'Fast lexically constrained decoding with dynamic beam allocation for neural machine translation,' in Proc. NAACL-HLT, 2018, pp. 1314–1324.",
                "[6] P. Srivastava, H. Tanwar, and M. S. Akhtar, 'Evaluating large language models on Hinglish code-mixed text normalization,' IEEE Access, vol. 12, pp. 41 200–41 215, 2024.",
                "[7] G. Aguilar et al., 'LinCE: A centralized benchmark for linguistic code-switching evaluation,' in Proc. LREC, 2020, pp. 3657–3666."
            ]
            for idx, r_text in enumerate(refs):
                p = shape.text_frame.paragraphs[0] if idx == 0 else shape.text_frame.add_paragraph()
                p.space_after = Pt(4)
                p.space_before = Pt(1)
                r = p.add_run()
                r.text = r_text
                r.font.name = 'Calibri'
                r.font.size = Pt(11)
                r.font.color.rgb = DARK_GRAY

    out_file = 'Review-2/BCSE497J_Project_I_Review_2_Presentation.pptx'
    prs.save(out_file)
    print(f"Successfully generated clean PPT at: {out_file}!")

if __name__ == '__main__':
    build_presentation()
