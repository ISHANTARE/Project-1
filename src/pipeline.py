"""
LAPS Pipeline
-------------
End-to-end orchestrator with Sentence-Level Code-Mixing Detection (CMI):
1. Detect sentence mode (Monolingual EN / Monolingual HI / Code-Mixed).
2. Tokenize & LID.
3. Extract 6 neural feature vectors per candidate token.
4. Compute LAPS preservation scores.
5. Apply dynamic register thresholds.
6. Constrained LLM rewriting & explanation generation.
"""

from src.laps.cmi_detector import detect_sentence_language
from src.laps.tokenizer import tokenize
from src.laps.feature_extractor import extract_features
from src.laps.scorer import score_all_tokens
from src.laps.threshold_engine import batch_decide
from src.llm.rewriter import LAPSRewriter
from src.llm.explainer import explain_all


def run(
    sentence: str,
    register: str = "business",
    llm_backend: str = "mock",
    llm_model: str | None = None,
    weights: dict | None = None,
) -> dict:
    """
    Execute the full LAPS pipeline.
    """
    # 1. Sentence-level Code-Mixing Detection (CMI)
    cmi_info = detect_sentence_language(sentence)

    # 2. Tokenize & LID
    tagged = tokenize(sentence)

    # Handle Monolingual English Input
    if cmi_info["mode"] == "MONOLINGUAL_EN":
        for t in tagged:
            if t["script"] != "punct":
                t["is_candidate"] = True
                t["laps_score"]   = 0.90
                t["decision"]     = "PRESERVE"
                t["confidence"]   = 0.40
                t["threshold"]    = 0.50
                t["features"]     = {"BF": 0.95, "DS": 0.90, "SI": 0.90, "CC": 1.0, "AF": 0.95, "TB": 0.10}
            else:
                t["is_candidate"] = False
                t["laps_score"]   = None
                t["decision"]     = "NATIVE"
                t["confidence"]   = None
                t["threshold"]    = None
                t["features"]     = {}

        preserve_list = [t["token"] for t in tagged if t["decision"] == "PRESERVE"]
        explanations  = explain_all(tagged)

        return {
            "cmi_info":       cmi_info,
            "tokens":         tagged,
            "preserve_list":  preserve_list,
            "rewrite_result": {
                "original":  sentence,
                "rewritten": sentence,  # Pure English sentence remains as-is
                "preserved": preserve_list,
                "prompt":    "Monolingual English Input - No rewriting required.",
                "backend":   llm_backend,
            },
            "explanations":   explanations,
            "is_monolingual": True,
        }

    # 3. Code-Mixed Pipeline: Feature extraction for candidate tokens
    surface_forms = [t["token"] for t in tagged]
    features_map  = {}
    for i, tok in enumerate(tagged):
        if tok["is_candidate"]:
            features_map[i] = extract_features(tok["token"], surface_forms, i)

    # 4. Score
    scored = score_all_tokens(tagged, features_map, weights)

    # 5. Threshold decisions
    decided = batch_decide(scored, register)

    # 6. Build preserve list
    preserve_list = [t["token"] for t in decided if t.get("decision") == "PRESERVE"]

    # 7. Rewrite via LLM
    rewriter = LAPSRewriter(backend=llm_backend, model=llm_model)
    rewrite_result = rewriter.rewrite(
        sentence=sentence,
        preserve_tokens=preserve_list,
    )

    # 8. Explain
    explanations = explain_all(decided)

    return {
        "cmi_info":       cmi_info,
        "tokens":         decided,
        "preserve_list":  preserve_list,
        "rewrite_result": rewrite_result,
        "explanations":   explanations,
        "is_monolingual": False,
    }


if __name__ == "__main__":
    print("=== MONOLINGUAL EN TEST ===")
    res_en = run("Hi! Does anyone have the link for the viva calendar?")
    print("Mode:", res_en["cmi_info"]["mode"])
    print("Rewritten:", res_en["rewrite_result"]["rewritten"])

    print("\n=== CODE-MIXED TEST ===")
    res_cm = run("hi! Kisi ke pass viva calender ki link h kya?")
    print("Mode:", res_cm["cmi_info"]["mode"])
    print("Preserved:", res_cm["preserve_list"])
    print("Rewritten:", res_cm["rewrite_result"]["rewritten"])
