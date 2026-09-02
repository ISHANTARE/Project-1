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
from src.laps.graph_engine import get_graph_engine
from src.laps.scorer import score_all_tokens
from src.laps.threshold_engine import batch_decide, batch_summarise
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
            t["is_candidate"]      = False
            t["laps_score"]        = None
            t["decision"]          = "PASSTHROUGH"
            t["probability"]       = None
            t["confidence"]        = None
            t["threshold"]         = None
            t["is_borderline"]     = False
            t["ego_density_bonus"] = 0.0
            t["naturalness_penalty"] = 0.0
            t["domain"]            = "N/A"
            t["features"]          = {}

        preserve_list = [t["token"] for t in tagged if t["decision"] == "PRESERVE"]
        explanations  = explain_all(tagged)

        return {
            "cmi_info":       cmi_info,
            "tokens":         tagged,
            "preserve_list":  preserve_list,
            "rewrite_result": {
                "original":  sentence,
                "rewritten": sentence,  # Pure English: no rewriting needed
                "preserved": preserve_list,
                "xml_prompt": "Monolingual English - No rewriting required.",
                "backend":   llm_backend,
            },
            "explanations":   explanations,
            "is_monolingual": True,
            "summary":        None,
        }

    # 3. Code-Mixed Pipeline: Feature extraction for candidate tokens
    kg_engine     = get_graph_engine()
    surface_forms = [t["token"] for t in tagged]
    features_map  = {}
    for i, tok in enumerate(tagged):
        if tok["is_candidate"]:
            features_map[i] = extract_features(tok["token"], surface_forms, i)

    # 4. Score
    scored = score_all_tokens(tagged, features_map, weights)

    # 5. Sigmoid Soft-Boundary Threshold decisions (with KG ego-density enrichment)
    decided = batch_decide(scored, register, kg_engine=kg_engine)
    summary = batch_summarise(decided)

    # 6. Build preserve list (PRESERVE + UNCERTAIN both retained)
    preserve_list = [
        t["token"] for t in decided
        if t.get("decision") in ("PRESERVE", "UNCERTAIN")
    ]

    # 7. Rewrite via LLM – passes decided_tokens for XML prompt builder
    rewriter = LAPSRewriter(backend=llm_backend, model=llm_model, kg_engine=kg_engine)
    rewrite_result = rewriter.rewrite(
        sentence=sentence,
        decided_tokens=decided,
        register=register,
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
        "summary":        {
            "total_candidates": summary.total_candidates,
            "preserve_count":   summary.preserve_count,
            "uncertain_count":  summary.uncertain_count,
            "rewrite_count":    summary.rewrite_count,
            "avg_probability":  summary.avg_probability,
        },
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
