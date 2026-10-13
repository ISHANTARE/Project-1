"""
LAPS LLM & NMT Rewriter – Graph-Enriched XML Prompt & Baseline Adapter
----------------------------------------------------------------------
Implements multi-strategy prompt builders and model backends:
  1. Graph-Enriched XML Prompt Builder (LAPS Proposed System)
  2. Few-Shot In-Context Learning (ICL) Prompt Builder (Standard LLM Baseline)
  3. Zero-Shot Unconstrained Prompt Builder (Ablation Baseline)
  4. Real NMT Backend (AI4Bharat IndicTrans2 / NLLB / Neural Surrogate)

Supported backends:
  • 'openai'   : OpenAI API (gpt-4o-mini, gpt-4o)
  • 'ollama'   : Local LLMs via Ollama (Llama 3.3, Qwen 2.5)
  • 'nmt'      : Neural Machine Translation (IndicTrans2, NLLB-200)
  • 'offline'  : Deterministic NMT Engine (no API key required)
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Literal, Any

try:
    from src.llm.nmt_backend import get_nmt_engine
except ModuleNotFoundError:
    from llm.nmt_backend import get_nmt_engine

# ---------------------------------------------------------------------------
# Domain Descriptions & Anti-Translation Rationales (§4 Research Plan)
# ---------------------------------------------------------------------------

DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    "Tech":       "technical computing or software domain",
    "Academic":   "academic or university context",
    "Business":   "professional business or corporate context",
    "Medical":    "medical or clinical domain",
    "Social":     "social media or casual digital context",
    "General":    "general everyday usage",
    "Legal":      "legal or regulatory domain",
    "Finance":    "financial or banking domain",
}

ANTI_TRANSLATION_TEMPLATES: Dict[str, str] = {
    "Tech":     "Converting to a literal translation would alter technical clarity and confuse technical readers.",
    "Academic": "Converting to a vernacular form incurs severe naturalness penalty in academic discourse.",
    "Business": "Replacing with a non-English term disrupts professional register and formal tone.",
    "Medical":  "Clinical terminology must remain unchanged to preserve precise medical meaning.",
    "General":  "This is a widely recognised loanword; translating it reduces fluency.",
    "Legal":    "Legal terms carry precise meanings; any translation risks ambiguity.",
    "Finance":  "Financial terminology must remain unchanged for professional clarity.",
    "Social":   "This term is part of natural digital vernacular; translating it sounds unnatural.",
}


# ---------------------------------------------------------------------------
# 1. Graph-Enriched XML Prompt Builder (LAPS Proposed)
# ---------------------------------------------------------------------------

def build_xml_prompt(
    sentence:       str,
    preserved_map:  Dict[str, Dict[str, Any]],
    source_lang:    str = "Hinglish",
    target_lang:    str = "English",
    register:       str = "business",
) -> str:
    """
    Construct a Graph-Enriched XML Prompt Context embedding token-level
    preservation constraints, domain labels, and edge naturalness penalties.
    """
    root = ET.Element("rewrite_task", attrib={
        "source_language": source_lang,
        "target_language": target_lang,
        "register":        register,
    })

    inp = ET.SubElement(root, "input_sentence")
    inp.text = sentence

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

    reg_instructions: Dict[str, str] = {
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

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


# ---------------------------------------------------------------------------
# 2. Few-Shot In-Context Learning Prompt Builder (Standard LLM Baseline)
# ---------------------------------------------------------------------------

def build_few_shot_prompt(
    sentence: str,
    register: str = "business",
) -> str:
    """
    Construct a standard Few-Shot In-Context Learning (ICL) prompt containing
    human demonstrations of loanword retention without external neuro-symbolic scoring.
    """
    return f"""You are an expert bilingual translator specialized in code-mixed Hinglish to English rewriting.
Rewrite the following Hinglish sentence into fluent, natural English suitable for a {register} context.
Guidelines:
- Translate the Hindi grammatical structures and vocabulary cleanly into English.
- Keep standard technical, academic, and business English loanwords (such as meeting, link, project, calendar, viva, server) in English; do not replace them with archaic or overly literal words.
- Return ONLY the rewritten English sentence without preamble or notes.

Examples:
Hinglish: Kal ki team meeting postpone ho gayi hai, update bhej do.
English: Tomorrow's team meeting has been postponed, please send the update.

Hinglish: Kya aapne viva schedule ka link check kiya?
English: Have you checked the link for the viva schedule?

Hinglish: Client ne presentation submit karne ke liye deadline extend kar di.
English: The client extended the deadline for submitting the presentation.

Now rewrite this sentence:
Hinglish: {sentence}
English:"""


# ---------------------------------------------------------------------------
# 3. Zero-Shot Unconstrained Prompt Builder (Ablation Baseline)
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(
    sentence: str,
    register: str = "business",
) -> str:
    """
    Construct a Zero-Shot prompt with no preservation constraints.
    """
    return f"""Translate the following Hinglish sentence into fluent English for a {register} context.
Output ONLY the translated sentence without preamble.

Sentence: {sentence}
Translation:"""


# ---------------------------------------------------------------------------
# Helper: Convert Decided Tokens to Preservation Map
# ---------------------------------------------------------------------------

def _map_to_preserved(
    decided_tokens: List[Dict[str, Any]],
    kg_engine=None,
) -> Dict[str, Dict[str, Any]]:
    preserved: Dict[str, Dict[str, Any]] = {}
    for tok in decided_tokens:
        if tok.get("decision") in ("PRESERVE", "UNCERTAIN"):
            token = tok["token"]
            nat_penalty = tok.get("naturalness_penalty", 0.70)
            domain = tok.get("domain", "General")
            if nat_penalty == 0.0 and kg_engine is not None:
                try:
                    edge = kg_engine.get_en_to_hi_edge(token)
                    nat_penalty = edge.get("naturalness_penalty", 0.70)
                    attrs = kg_engine.get_node_attributes(token)
                    domain = attrs.get("domain", "General")
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
# LLM API Callers
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip().strip('"')
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
            return data.get("response", "").strip().strip('"')
    except Exception as e:
        return f"[Ollama Error: {e}]"


def _call_fluent_translator(sentence: str, preserve_tokens: List[str]) -> str:
    """
    Direct wrapper over NMTTranslationEngine for offline translation.
    """
    nmt = get_nmt_engine()
    force_constraints = bool(preserve_tokens)
    return nmt.translate(sentence, preserve_tokens, force_constraints=force_constraints)


# ---------------------------------------------------------------------------
# Master Rewriter Class
# ---------------------------------------------------------------------------

class LAPSRewriter:
    """
    Unified rewriter supporting LAPS XML prompt injection,
    Few-Shot ICL baseline, Zero-Shot baseline, and Neural NMT backends.
    """

    def __init__(
        self,
        backend:   Optional[str] = None,
        model:     Optional[str] = None,
        strategy:  Literal["xml_laps", "few_shot_icl", "zero_shot"] = "xml_laps",
        kg_engine=None,
    ):
        self.backend   = (backend or os.getenv("LAPS_LLM_BACKEND", "mock")).lower()
        self.model     = model
        self.strategy  = strategy
        self.kg_engine = kg_engine
        self.nmt_engine = get_nmt_engine()

    def rewrite(
        self,
        sentence:        str,
        preserve_tokens: Optional[List[str]] = None,
        decided_tokens:  Optional[List[Dict[str, Any]]] = None,
        source_lang:     str = "Hinglish",
        target_lang:     str = "English",
        register:        str = "business",
        constrained:     bool = True,
    ) -> Dict[str, Any]:
        """
        Execute rewrite across configured prompt strategy and backend.
        """
        # 1. Build preservation mapping
        if decided_tokens is not None and constrained:
            preserved_map = _map_to_preserved(decided_tokens, self.kg_engine)
            preserve_list = list(preserved_map.keys())
        elif preserve_tokens and constrained:
            preserved_map = {
                t: {"action": "PRESERVE", "domain": "General", "naturalness_penalty": 0.70}
                for t in preserve_tokens
            }
            preserve_list = preserve_tokens
        else:
            preserved_map = {}
            preserve_list = []

        # 2. Select prompt string based on strategy
        if self.strategy == "xml_laps":
            prompt = build_xml_prompt(sentence, preserved_map, source_lang, target_lang, register)
        elif self.strategy == "few_shot_icl":
            prompt = build_few_shot_prompt(sentence, register)
        else:  # zero_shot
            prompt = build_zero_shot_prompt(sentence, register)

        # 3. Dispatch to backend
        if self.backend == "openai" and os.getenv("OPENAI_API_KEY"):
            rewritten = _call_openai(prompt, self.model or "gpt-4o-mini")
        elif self.backend == "ollama":
            rewritten = _call_ollama(prompt, self.model or "llama3")
        elif self.backend in ("nmt", "indictrans"):
            rewritten = self.nmt_engine.translate(
                sentence,
                preserve_tokens=preserve_list if constrained else None,
                force_constraints=constrained
            )
        else:
            # Deterministic NMT Engine / Mock
            rewritten = self.nmt_engine.translate(
                sentence,
                preserve_tokens=preserve_list if constrained else None,
                force_constraints=constrained
            )

        return {
            "original":         sentence,
            "rewritten":        rewritten,
            "preserved":        preserve_list,
            "prompt":           prompt,
            "backend":          self.backend,
            "strategy":         self.strategy,
            "register":         register,
            "preservation_map": preserved_map,
            "execution_mode":   getattr(self.nmt_engine, "last_execution_mode", "UNKNOWN"),
        }
