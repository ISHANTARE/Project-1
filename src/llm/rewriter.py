"""
LAPS LLM Rewriter – Graph-Enriched XML Prompt Builder
------------------------------------------------------
Implements the structured XML prompt architecture from §4 of the LAPS
Research Blueprint.  Instead of a plain-text instruction, each rewrite
request is compiled into a machine-parseable XML document embedding:

  • Per-token preservation constraints with domain tags and naturalness_penalty.
  • Anti-hallucination penalty declarations ("Do not translate X").
  • Register-level rewrite instructions.

Supports backends:
  1. OpenAI API  (GPT-4o, GPT-4o-mini)
  2. Local Ollama (Llama 3, Qwen, Mistral)
  3. Smart Offline Hinglish-to-English Translator (no API key required)
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Literal

# ---------------------------------------------------------------------------
# XML Prompt Builder (§4 Research Blueprint)
# ---------------------------------------------------------------------------

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "Tech":       "technical computing or software domain",
    "Academic":   "academic or university context",
    "Business":   "professional business or corporate context",
    "Medical":    "medical or clinical domain",
    "Social":     "social media or casual digital context",
    "General":    "general everyday usage",
    "Legal":      "legal or regulatory domain",
    "Finance":    "financial or banking domain",
}

ANTI_TRANSLATION_TEMPLATES: dict[str, str] = {
    "Tech":     "Converting to a literal translation would alter technical clarity and confuse technical readers.",
    "Academic": "Converting to a vernacular form incurs severe naturalness penalty in academic discourse.",
    "Business": "Replacing with a non-English term disrupts professional register and formal tone.",
    "Medical":  "Clinical terminology must remain unchanged to preserve precise medical meaning.",
    "General":  "This is a widely recognised loanword; translating it reduces fluency.",
    "Legal":    "Legal terms carry precise meanings; any translation risks ambiguity.",
    "Finance":  "Financial terminology must remain unchanged for professional clarity.",
    "Social":   "This term is part of natural digital vernacular; translating it sounds unnatural.",
}


def build_xml_prompt(
    sentence:       str,
    preserved_map:  dict[str, dict],   # {token: {"domain": str, "naturalness_penalty": float, "action": str}}
    source_lang:    str = "Hinglish",
    target_lang:    str = "English",
    register:       str = "business",
) -> str:
    """
    Construct a Graph-Enriched XML Prompt Context per §4 of the research plan.

    Args:
        sentence      : Original code-mixed sentence.
        preserved_map : Dict of tokens marked PRESERVE with their KG metadata.
        source_lang   : Source language tag.
        target_lang   : Target language tag.
        register      : Communicative register (casual/business/academic/social).

    Returns:
        str – Well-formed XML prompt string ready for LLM injection.
    """
    root = ET.Element("rewrite_task", attrib={
        "source_language": source_lang,
        "target_language": target_lang,
        "register":        register,
    })

    # <input_sentence>
    inp = ET.SubElement(root, "input_sentence")
    inp.text = sentence

    # <preservation_constraints>
    if preserved_map:
        constraints = ET.SubElement(root, "preservation_constraints")
        for token, meta in preserved_map.items():
            action  = meta.get("action", "PRESERVE")
            domain  = meta.get("domain", "General")
            penalty = meta.get("naturalness_penalty", 0.70)
            descr   = DOMAIN_DESCRIPTIONS.get(domain, "general usage domain")
            rationale = ANTI_TRANSLATION_TEMPLATES.get(
                domain, ANTI_TRANSLATION_TEMPLATES["General"]
            )

            tok_el = ET.SubElement(constraints, "token", attrib={
                "name":                token,
                "action":              action,
                "domain":              domain,
                "naturalness_penalty": f"{penalty:.2f}",
            })
            tok_el.text = (
                f"Do not translate '{token}' ({descr}). "
                f"{rationale}"
            )

    # <register_instruction>
    reg_instructions: dict[str, str] = {
        "casual":   "Rewrite in a friendly, conversational English tone.",
        "social":   "Rewrite in an engaging, modern social-media English style.",
        "business": "Rewrite in formal, professional business English.",
        "academic": "Rewrite in precise, formal academic English.",
    }
    instr = ET.SubElement(root, "instruction")
    instr.text = (
        f"{reg_instructions.get(register, reg_instructions['business'])} "
        "Translate all surrounding Hinglish or Hindi text into natural English "
        "while strictly retaining all tokens listed in <preservation_constraints>. "
        "Output ONLY the final rewritten sentence, with no explanations or preamble."
    )

    # Pretty-print with indentation
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def _map_to_preserved(
    decided_tokens: list[dict],
    kg_engine=None,
) -> dict[str, dict]:
    """
    Convert the output of threshold_engine.batch_decide() into a
    preserved_map dict suitable for build_xml_prompt().
    """
    preserved: dict[str, dict] = {}
    for tok in decided_tokens:
        if tok.get("decision") in ("PRESERVE", "UNCERTAIN"):
            token = tok["token"]
            # Pull KG metadata
            nat_penalty = tok.get("naturalness_penalty", 0.70)
            domain      = tok.get("domain", "General")
            if nat_penalty == 0.0 and kg_engine is not None:
                try:
                    edge = kg_engine.get_en_to_hi_edge(token)
                    nat_penalty = edge.get("naturalness_penalty", 0.70)
                    attrs       = kg_engine.get_node_attributes(token)
                    domain      = attrs.get("domain", "General")
                except Exception:
                    pass

            preserved[token] = {
                "action":              tok["decision"],
                "domain":              domain,
                "naturalness_penalty": nat_penalty,
                "probability":         tok.get("probability", 0.70),
                "is_borderline":       tok.get("is_borderline", False),
            }
    return preserved


# ---------------------------------------------------------------------------
# Offline Hinglish-to-English Natural Translation Engine
# ---------------------------------------------------------------------------

HINGLISH_PATTERNS = [
    # Interrogatives & possessives
    (r'(?i)\bkisi ke pa+ss?\b', 'anyone has'),
    (r'(?i)\bh kya\??\b',       '?'),
    (r'(?i)\bhai kya\??\b',     '?'),
    (r'(?i)\bki link\b',        'the link for'),
    (r'(?i)\bki pdf\b',         'the pdf for'),
    (r'(?i)\bki report\b',      'the report for'),
    (r'(?i)\bki presentation\b','the presentation for'),
    (r'(?i)\bka debug\b',       'the debugging of'),
    (r'(?i)\bka code\b',        'the code for'),
    (r'(?i)\bka update\b',      'the update for'),
    # Verb phrases
    (r'(?i)\bpostpone ho gay[ai]\b',        'has been postponed'),
    (r'(?i)\bcancel ho gaya\b',             'has been cancelled'),
    (r'(?i)\bdebug karna padega\b',         'needs to be debugged'),
    (r'(?i)\bsubmit karna hai\b',           'needs to be submitted'),
    (r'(?i)\bleni padegi\b',                'needs to be taken'),
    (r'(?i)\bphir se schedule karein\b',    'reschedule it'),
    (r'(?i)\bmat bhoolna\b',                'do not forget'),
    (r'(?i)\bbhej dena\b',                  'please send'),
    (r'(?i)\bshare kar dena\b',             'please share'),
    (r'(?i)\bcheck kar lena\b',             'please check'),
    (r'(?i)\bkarna padega\b',               'will need to be done'),
    # Simple vocabulary
    (r'(?i)\bkal\b',    'tomorrow'),
    (r'(?i)\baaj\b',    'today'),
    (r'(?i)\bkyu\b',    'why'),
    (r'(?i)\bkya\b',    'what'),
    (r'(?i)\bhum\b',    'we'),
    (r'(?i)\baap\b',    'you'),
    (r'(?i)\btum\b',    'you'),
    (r'(?i)\bko\b',     'to'),
    (r'(?i)\bmein\b',   'in'),
    (r'(?i)\bpar\b',    'on'),
    (r'(?i)\bpe\b',     'on'),
    (r'(?i)\bpr\b',     'on'),
    (r'(?i)\bse\b',     'from'),
    (r'(?i)\bne kaha\b','said'),
    (r'(?i)\bne\b',     'has'),
    (r'(?i)\bha\b',     'yes'),
    (r'(?i)\bnahi\b',   'no'),
    (r'(?i)\bnahi\b',   'not'),
    (r'(?i)\bbhi\b',    'also'),
    (r'(?i)\bsab\b',    'everyone'),
]


def _call_fluent_translator(sentence: str, preserve_tokens: list[str]) -> str:
    """
    Rule-based Hinglish-to-English translator.
    Handles structural sentence templates first, then lexical substitution.
    """
    result = sentence

    # Fix common loanword typos first
    result = re.sub(r'(?i)\bcalender\b', 'calendar', result)
    result = re.sub(r'(?i)\bscedule\b',  'schedule',  result)
    result = re.sub(r'(?i)\bpresntation\b', 'presentation', result)

    # Structural templates (order matters)
    # Template: "Kisi ke pass <X> ki link h kya?"
    m_link = re.search(r'(?i)kisi ke pa+ss?\s+(.+?)\s+ki link h(?:ai)?\s+kya\??', result)
    if m_link:
        obj    = m_link.group(1).strip()
        prefix = "Hi! " if re.match(r'(?i)^hi[!\s,]', result) else ""
        return f"{prefix}Does anyone have the link for the {obj}?"

    # Template: "Kisi ke pass <X> h kya?"
    m_pass = re.search(r'(?i)kisi ke pa+ss?\s+(.+?)\s+h(?:ai)?\s+kya\??', result)
    if m_pass:
        obj    = m_pass.group(1).strip()
        prefix = "Hi! " if re.match(r'(?i)^hi[!\s,]', result) else ""
        return f"{prefix}Does anyone have {obj}?"

    # Template: "Kal <event> postpone ho gay*, kya hum phir se schedule karein?"
    m_postpone = re.search(
        r'(?i)kal\s+(.+?)\s+postpone ho gay[ai],?\s+kya hum phir se schedule karein\??',
        result,
    )
    if m_postpone:
        event = m_postpone.group(1).strip()
        return f"Tomorrow's {event} was postponed. Should we reschedule it?"

    # Generic lexical substitution pass
    for pattern, repl in HINGLISH_PATTERNS:
        result = re.sub(pattern, repl, result)

    # Cleanup
    result = re.sub(r'\?\s*\?', '?',  result)
    result = re.sub(r'\s+',     ' ',  result).strip()
    if result and result[0].isalpha():
        result = result[0].upper() + result[1:]

    return result


# ---------------------------------------------------------------------------
# OpenAI & Ollama Callers
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI API Error: {e}]"


def _call_ollama(prompt: str, model: str = "llama3") -> str:
    try:
        import urllib.request, json
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": False
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as e:
        return f"[Ollama Error: {e}]"


# ---------------------------------------------------------------------------
# Main Rewriter Class
# ---------------------------------------------------------------------------

class LAPSRewriter:
    """
    Orchestrates the Graph-Enriched XML Prompt → LLM → Rewritten Output pipeline.

    Usage:
        rw = LAPSRewriter(backend="openai")
        result = rw.rewrite(
            sentence="hi! Kisi ke pass viva calender ki link h kya?",
            decided_tokens=[...],   # from threshold_engine.batch_decide()
            register="academic",
        )
    """

    def __init__(
        self,
        backend:   str | None = None,
        model:     str | None = None,
        kg_engine=None,
    ):
        self.backend    = (backend or os.getenv("LAPS_LLM_BACKEND", "mock")).lower()
        self.model      = model
        self.kg_engine  = kg_engine

    def rewrite(
        self,
        sentence:       str,
        preserve_tokens: list[str] | None = None,  # legacy simple list
        decided_tokens: list[dict] | None = None,   # rich threshold_engine output
        source_lang:    str = "Hinglish",
        target_lang:    str = "English",
        register:       str = "business",
    ) -> dict:
        """
        Generate a rewrite using the XML prompt strategy.

        Args:
            sentence       : Original code-mixed sentence.
            preserve_tokens: Simple list of tokens to preserve (legacy fallback).
            decided_tokens : Full output of threshold_engine.batch_decide().
            source_lang    : Source language label.
            target_lang    : Target language label.
            register       : Communicative register.

        Returns:
            dict with keys:
              original, rewritten, preserved, xml_prompt, backend, register,
              preservation_map (detailed metadata per preserved token)
        """
        # --- Build preservation map ---
        if decided_tokens is not None:
            preserved_map = _map_to_preserved(decided_tokens, self.kg_engine)
            preserve_list = list(preserved_map.keys())
        elif preserve_tokens:
            # Legacy fallback – minimal metadata
            preserved_map = {
                t: {"action": "PRESERVE", "domain": "General",
                    "naturalness_penalty": 0.70}
                for t in preserve_tokens
            }
            preserve_list = preserve_tokens
        else:
            preserved_map = {}
            preserve_list = []

        # --- Build XML prompt ---
        xml_prompt = build_xml_prompt(
            sentence      = sentence,
            preserved_map = preserved_map,
            source_lang   = source_lang,
            target_lang   = target_lang,
            register      = register,
        )

        # --- Dispatch to backend ---
        if self.backend == "openai" and os.getenv("OPENAI_API_KEY"):
            rewritten = _call_openai(xml_prompt, self.model or "gpt-4o-mini")
        elif self.backend == "ollama":
            rewritten = _call_ollama(xml_prompt, self.model or "llama3")
        else:
            # Fluent offline translator (default / mock)
            rewritten = _call_fluent_translator(sentence, preserve_list)

        return {
            "original":         sentence,
            "rewritten":        rewritten,
            "preserved":        preserve_list,
            "xml_prompt":       xml_prompt,
            "backend":          self.backend,
            "register":         register,
            "preservation_map": preserved_map,
        }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rw = LAPSRewriter(backend="mock")

    r1 = rw.rewrite(
        sentence="hi! Kisi ke pass viva calender ki link h kya?",
        preserve_tokens=["viva", "calendar", "link"],
        register="academic",
    )
    print("=== XML PROMPT ===")
    print(r1["xml_prompt"])
    print("\n=== REWRITTEN ===")
    print(r1["rewritten"])

    print("\n---")
    r2 = rw.rewrite(
        sentence="Kal client meeting postpone ho gayi, kya hum phir se schedule karein?",
        preserve_tokens=["client", "meeting"],
        register="business",
    )
    print("Rewritten:", r2["rewritten"])
