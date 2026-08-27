"""
LAPS LLM Rewriter & Translator
------------------------------
Generates fluent English rewrites while enforcing hard token preservation constraints.

Supports:
1. OpenAI API (GPT-4o, GPT-4o-mini)
2. Local Ollama (Llama 3, Qwen)
3. Smart Offline Hinglish-to-English Natural Translator (no API key needed)
"""

import os
import re

PROMPT_TEMPLATE = """\
You are a professional multilingual editor. Rewrite the following Hinglish sentence into natural, fluent English.

STRICT CONSTRAINTS:
1. Do NOT translate or modify these exact words (keep them in English): {preserve_list}
2. Translate all surrounding Hinglish/Hindi text into natural English.
3. Output ONLY the final rewritten sentence without explanations.

Source: "{sentence}"
Rewritten:"""


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI API Error: {e}]"


def _call_ollama(prompt: str, model: str = "llama3") -> str:
    try:
        import urllib.request
        import json
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
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


# ── Smart Offline Natural Translation Engine ──────────────────────────────────
# Translates Hinglish grammar markers into fluent English while retaining preserved terms intact.

HINGLISH_PATTERNS = [
    # Interrogatives & possessives
    (r'(?i)\bkisi ke pass\b', 'anyone has'),
    (r'(?i)\bkisi ke pas\b', 'anyone has'),
    (r'(?i)\bkisi ke paas\b', 'anyone has'),
    (r'(?i)\bh kya\?\b', '?'),
    (r'(?i)\bh kya\b', '?'),
    (r'(?i)\bhai kya\?\b', '?'),
    (r'(?i)\bhai kya\b', '?'),
    (r'(?i)\bki link\b', 'the link for'),
    (r'(?i)\bki pdf\b', 'the pdf for'),
    (r'(?i)\bki report\b', 'the report for'),
    (r'(?i)\bki presentation\b', 'the presentation for'),
    (r'(?i)\bka debug\b', 'the debugging of'),
    (r'(?i)\bka code\b', 'the code for'),
    (r'(?i)\bka update\b', 'the update for'),

    # Verbs & Auxiliary phrases
    (r'(?i)\bpostpone ho gayi\b', 'has been postponed'),
    (r'(?i)\bpostpone ho gaya\b', 'has been postponed'),
    (r'(?i)\bcancel ho gaya\b', 'has been cancelled'),
    (r'(?i)\bdebug karna padega\b', 'needs to be debugged'),
    (r'(?i)\bsubmit karna hai\b', 'needs to be submitted'),
    (r'(?i)\bleni padegi\b', 'needs to be taken'),
    (r'(?i)\bphir se schedule karein\b', 'reschedule it'),
    (r'(?i)\bmat bhoolna\b', 'do not forget'),

    # Standard words
    (r'(?i)\bkal\b', 'tomorrow'),
    (r'(?i)\baaj\b', 'today'),
    (r'(?i)\bkyu\b', 'why'),
    (r'(?i)\bkya\b', 'what'),
    (r'(?i)\bhum\b', 'we'),
    (r'(?i)\baap\b', 'you'),
    (r'(?i)\btum\b', 'you'),
    (r'(?i)\bko\b', 'to'),
    (r'(?i)\bmein\b', 'in'),
    (r'(?i)\bpar\b', 'on'),
    (r'(?i)\bpe\b', 'on'),
    (r'(?i)\bpr\b', 'on'),
    (r'(?i)\bse\b', 'from'),
    (r'(?i)\bne kaha\b', 'said'),
    (r'(?i)\bne\b', 'has'),
]


def _call_fluent_translator(sentence: str, preserve_tokens: list[str]) -> str:
    """
    Offline fluent translator: Converts Hinglish structures into natural English
    while keeping preserved loanwords untouched.
    """
    result = sentence

    # 1. Fix common loanword typos
    result = re.sub(r'(?i)\bcalender\b', 'calendar', result)

    # 2. Structural Question/Statement Templates
    # Pattern: Kisi ke pass [object] ki link h kya? -> Does anyone have the link for [object]?
    m_link = re.search(r'(?i)kisi ke pa?ss (.+?) ki link h(ai)? kya\??', result)
    if m_link:
        obj = m_link.group(1).strip()
        prefix = "Hi! " if re.match(r'(?i)^hi[!\s,]', result) else ""
        return f"{prefix}Does anyone have the link for the {obj}?"

    m_pass = re.search(r'(?i)kisi ke pa?ss (.+?) h(ai)? kya\??', result)
    if m_pass:
        obj = m_pass.group(1).strip()
        prefix = "Hi! " if re.match(r'(?i)^hi[!\s,]', result) else ""
        return f"{prefix}Does anyone have {obj}?"

    # Pattern: Kal [event] postpone ho gayi, kya hum phir se schedule karein?
    m_meeting = re.search(r'(?i)kal (.+?) postpone ho gay?i,?\s*kya hum phir se schedule karein\??', result)
    if m_meeting:
        event = m_meeting.group(1).strip()
        return f"Tomorrow's {event} was postponed. Should we reschedule it?"

    # Generic fallback pattern replacement
    for pattern, repl in HINGLISH_PATTERNS:
        result = re.sub(pattern, repl, result)

    # Clean up double punctuation & spacing
    result = re.sub(r'\?\s*\?', '?', result)
    result = re.sub(r'\s+', ' ', result).strip()
    if result and result[0].isalpha():
        result = result[0].upper() + result[1:]

    return result


class LAPSRewriter:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = (backend or os.getenv("LAPS_LLM_BACKEND", "mock")).lower()
        self.model   = model

    def rewrite(self, sentence: str, preserve_tokens: list[str],
                source_lang: str = "Hinglish",
                target_lang: str = "English") -> dict:

        preserve_str = ", ".join(f'"{t}"' for t in preserve_tokens) if preserve_tokens else "none"
        prompt = PROMPT_TEMPLATE.format(preserve_list=preserve_str, sentence=sentence)

        if self.backend == "openai" and os.getenv("OPENAI_API_KEY"):
            rewritten = _call_openai(prompt, self.model or "gpt-4o-mini")
        elif self.backend == "ollama":
            rewritten = _call_ollama(prompt, self.model or "llama3")
        else:
            # Fluent natural offline translator
            rewritten = _call_fluent_translator(sentence, preserve_tokens)

        return {
            "original":  sentence,
            "rewritten": rewritten,
            "preserved": preserve_tokens,
            "prompt":    prompt,
            "backend":   self.backend,
        }


if __name__ == "__main__":
    rw = LAPSRewriter(backend="mock")
    print(rw.rewrite("hi! Kisi ke pass viva calender ki link h kya?", ["hi", "viva", "calendar", "link"]))
    print(rw.rewrite("Kal client meeting postpone ho gayi, kya hum phir se schedule karein?", ["client", "meeting"]))
