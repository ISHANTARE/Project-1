"""
LAPS Audit Report Exporter Engine
----------------------------------
Implements multi-format report exports for research auditing:
1. JSON: Complete machine-readable audit record with full metadata, feature breakdown, and XML.
2. CSV: Structured token-level decision matrix with feature values and justifications.
3. PDF: Human-readable publication-quality audit report (via fpdf2) with header metrics,
   token breakdown table, XML constraints, and model metadata.
"""

from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def format_audit_record(
    result: Dict[str, Any],
    run_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    backend_info: Optional[str] = None
) -> Dict[str, Any]:
    """
    Assemble a standardized, complete audit dictionary from pipeline execution results.
    """
    ts = datetime.utcnow().isoformat() + "Z"
    rid = run_id or f"LAPS-{int(time.time() * 1000)}"

    rewrite_res = result.get("rewrite_result", {})
    cmi_info = result.get("cmi_info", {})
    tokens = result.get("tokens", [])
    explanations = result.get("explanations", [])
    summary = result.get("summary") or {}

    # Extract candidates with detailed feature breakdowns
    candidates_audit = []
    for tok in tokens:
        if tok.get("is_candidate"):
            candidates_audit.append({
                "token": tok.get("token"),
                "language": tok.get("language"),
                "laps_score": tok.get("laps_score"),
                "probability": tok.get("probability"),
                "decision": tok.get("decision"),
                "confidence": tok.get("confidence"),
                "threshold": tok.get("threshold"),
                "domain": tok.get("domain", "General"),
                "naturalness_penalty": tok.get("naturalness_penalty", 0.0),
                "ego_density_bonus": tok.get("ego_density_bonus", 0.0),
                "features": tok.get("features", {}),
            })

    return {
        "metadata": {
            "run_id": rid,
            "timestamp": ts,
            "framework": "LAPS v2.0 (Linguistic Adaptive Preservation Scoring)",
            "institution": "Vellore Institute of Technology, Vellore",
            "course": "BCSE497J - Project-I",
            "backend": rewrite_res.get("backend", "offline"),
            "backend_details": backend_info or rewrite_res.get("backend", "offline"),
            "strategy": rewrite_res.get("strategy", "xml_laps"),
            "execution_duration_ms": duration_ms or 0.0,
            "is_monolingual": result.get("is_monolingual", False),
        },
        "sentence_analysis": {
            "original_sentence": rewrite_res.get("original", ""),
            "rewritten_sentence": rewrite_res.get("rewritten", ""),
            "cmi_mode": cmi_info.get("mode", "UNKNOWN"),
            "cmi_score": cmi_info.get("cmi", 0.0),
            "en_ratio": cmi_info.get("en_ratio", 0.0),
            "hi_ratio": cmi_info.get("hi_ratio", 0.0),
            "register": rewrite_res.get("register", "business"),
            "preserved_tokens": result.get("preserve_list", []),
        },
        "summary_metrics": summary,
        "candidate_tokens": candidates_audit,
        "token_explanations": explanations,
        "xml_constraints": rewrite_res.get("xml_prompt", ""),
    }


def export_to_json(
    record: Dict[str, Any],
    output_path: Optional[str | Path] = None
) -> str:
    """
    Export audit record as a formatted JSON string or file.
    """
    json_str = json.dumps(record, indent=2, ensure_ascii=False)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_str, encoding="utf-8")
    return json_str


def export_to_csv(
    record: Dict[str, Any],
    output_path: Optional[str | Path] = None
) -> str:
    """
    Export candidate tokens and decisions as a CSV table.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Token", "Language", "Domain", "BF", "DS", "SI", "CC", "AF", "TB",
        "LAPS_Score", "P_Preserve", "Decision", "Confidence", "Justification"
    ])

    expl_map = {ex["token"]: ex.get("primary", "") for ex in record.get("token_explanations", [])}

    for c in record.get("candidate_tokens", []):
        f = c.get("features", {})
        writer.writerow([
            c.get("token", ""),
            c.get("language", ""),
            c.get("domain", ""),
            f.get("BF", ""),
            f.get("DS", ""),
            f.get("SI", ""),
            f.get("CC", ""),
            f.get("AF", ""),
            f.get("TB", ""),
            c.get("laps_score", ""),
            c.get("probability", ""),
            c.get("decision", ""),
            c.get("confidence", ""),
            expl_map.get(c.get("token", ""), ""),
        ])

    csv_str = output.getvalue()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_str, encoding="utf-8")
    return csv_str


def _clean_latin(text: Any) -> str:
    """
    Normalize and encode text safely for FPDF core Helvetica font (Latin-1 / ASCII safe).
    Replaces Unicode dashes, bullets, quotes, and symbols with ASCII counterparts.
    """
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2022": "*",    # bullet
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u03b8": "theta",# greek theta
        "\u2265": ">=",   # greater or equal
        "\u2264": "<=",   # less or equal
        "\u2192": "->",   # right arrow
        "\u2190": "<-",   # left arrow
        "\u00b1": "+/-",  # plus-minus
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def export_to_pdf(
    record: Dict[str, Any],
    output_path: Optional[str | Path] = None
) -> bytes:
    """
    Generate a publication-ready PDF audit report using fpdf2 with safe character sanitization.
    """
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Colors
    header_color = (15, 23, 42)
    accent_color = (59, 130, 246)
    subtle_color = (100, 116, 139)
    text_color = (30, 41, 59)

    # Title Banner
    pdf.set_fill_color(*header_color)
    pdf.rect(0, 0, 210, 26, "F")
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(12, 6)
    pdf.cell(0, 7, "LAPS: Linguistic Adaptive Preservation Scoring", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(12, 14)
    pdf.cell(0, 5, "Token-Level Decision Audit & Explainability Report | VIT Project-I", ln=True)

    # Metadata Panel
    pdf.set_xy(12, 32)
    pdf.set_text_color(*text_color)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "1. RUN METADATA", ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    meta = record.get("metadata", {})
    sent_info = record.get("sentence_analysis", {})

    pdf.cell(90, 5, _clean_latin(f"Run ID: {meta.get('run_id')}"), ln=False)
    pdf.cell(90, 5, _clean_latin(f"Timestamp: {meta.get('timestamp')}"), ln=True)
    pdf.cell(90, 5, _clean_latin(f"Backend: {meta.get('backend_details')}"), ln=False)
    pdf.cell(90, 5, _clean_latin(f"Register: {sent_info.get('register')}"), ln=True)
    pdf.cell(90, 5, _clean_latin(f"Language Mode: {sent_info.get('cmi_mode')}"), ln=False)
    pdf.cell(90, 5, _clean_latin(f"Code-Mixing Index (CMI): {sent_info.get('cmi_score')} %"), ln=True)
    if meta.get("execution_duration_ms"):
        pdf.cell(90, 5, _clean_latin(f"Latency: {meta.get('execution_duration_ms'):.1f} ms"), ln=True)

    pdf.ln(3)

    # Sentences
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(12)
    pdf.cell(186, 6, "2. SENTENCE REWRITING SUMMARY", ln=True)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(12)
    pdf.cell(186, 5, "Original Input:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    orig_clean = _clean_latin(sent_info.get("original_sentence", ""))
    pdf.set_x(12)
    pdf.multi_cell(186, 5, f"  {orig_clean}")
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(12)
    pdf.cell(186, 5, "Rewritten Output:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    rew_clean = _clean_latin(sent_info.get("rewritten_sentence", ""))
    pdf.set_x(12)
    pdf.multi_cell(186, 5, f"  {rew_clean}")
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(12)
    pdf.cell(35, 5, "Preserved Entities:", ln=False)
    pdf.set_font("Helvetica", "", 9)
    preserved_str = ", ".join(sent_info.get("preserved_tokens", [])) or "None"
    pdf.cell(151, 5, _clean_latin(preserved_str), ln=True)

    pdf.ln(4)

    # Token Decision Table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(12)
    pdf.cell(186, 6, "3. TOKEN-LEVEL LAPS DECISION MATRIX", ln=True)

    # Table Header
    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Helvetica", "B", 8)
    col_widths = [26, 18, 16, 16, 16, 16, 16, 16, 20, 22]
    headers = ["Token", "Domain", "BF", "DS", "SI", "CC", "AF", "TB", "Score", "Decision"]
    pdf.set_x(12)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 6, h, border=1, fill=True, align="C")
    pdf.ln(6)

    # Rows
    pdf.set_font("Helvetica", "", 8)
    for c in record.get("candidate_tokens", []):
        f = c.get("features", {})
        dec = c.get("decision", "")
        sc = c.get("laps_score")
        sc_str = f"{sc:.3f}" if sc is not None else "-"
        
        pdf.set_x(12)
        pdf.cell(col_widths[0], 5, _clean_latin(c.get("token", "")), border=1)
        pdf.cell(col_widths[1], 5, _clean_latin(str(c.get("domain", "General"))[:10]), border=1, align="C")
        pdf.cell(col_widths[2], 5, f"{f.get('BF', 0):.2f}", border=1, align="C")
        pdf.cell(col_widths[3], 5, f"{f.get('DS', 0):.2f}", border=1, align="C")
        pdf.cell(col_widths[4], 5, f"{f.get('SI', 0):.2f}", border=1, align="C")
        pdf.cell(col_widths[5], 5, f"{f.get('CC', 0):.2f}", border=1, align="C")
        pdf.cell(col_widths[6], 5, f"{f.get('AF', 0):.2f}", border=1, align="C")
        pdf.cell(col_widths[7], 5, f"{f.get('TB', 0):.2f}", border=1, align="C")
        pdf.cell(col_widths[8], 5, sc_str, border=1, align="C")
        pdf.cell(col_widths[9], 5, _clean_latin(dec), border=1, align="C")
        pdf.ln(5)

    pdf.ln(4)

    # Justifications
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(12)
    pdf.cell(186, 6, "4. EXPLANATIONS & RATIONALES", ln=True)
    pdf.set_font("Helvetica", "", 8)
    for ex in record.get("token_explanations", []):
        tok = ex.get("token", "")
        dec = ex.get("decision", "")
        prim = ex.get("primary", "")
        line_txt = _clean_latin(f"* [{tok} -> {dec}]: {prim}")
        pdf.set_x(12)
        pdf.multi_cell(186, 4, line_txt)

    pdf.ln(4)

    # XML Section if present
    xml_data = record.get("xml_constraints", "")
    if xml_data and xml_data != "Monolingual English - No rewriting required.":
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(12)
        pdf.cell(186, 6, "5. MACHINE-PARSEABLE XML CONSTRAINTS", ln=True)
        pdf.set_font("Courier", "", 7)
        # Indent and truncate if very long
        for line in xml_data.splitlines()[:25]:
            pdf.set_x(12)
            pdf.cell(186, 3.8, _clean_latin(line), ln=True)

    pdf_bytes = pdf.output()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pdf_bytes)
    return pdf_bytes
