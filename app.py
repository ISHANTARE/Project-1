"""
LAPS Interactive Demo — Streamlit App
======================================
Run: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.pipeline import run
from src.laps.threshold_engine import available_registers
from src.llm.explainer import format_explanation_table

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LAPS – Linguistic Adaptive Preservation Scoring",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d1117; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f172a 60%, #1e1040 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero p { color: #94a3b8; margin: 0; font-size: 0.95rem; }

/* Token pills */
.pill-preserve {
    display:inline-block; padding:3px 10px; border-radius:20px;
    background:#1e3a2f; color:#4ade80; border:1px solid #22c55e;
    font-size:0.8rem; font-weight:600; margin:2px;
}
.pill-rewrite {
    display:inline-block; padding:3px 10px; border-radius:20px;
    background:#3b1f1f; color:#f87171; border:1px solid #ef4444;
    font-size:0.8rem; font-weight:600; margin:2px;
}
.pill-native {
    display:inline-block; padding:3px 10px; border-radius:20px;
    background:#1e293b; color:#94a3b8; border:1px solid #475569;
    font-size:0.8rem; margin:2px;
}

/* Cards */
.card {
    background:#1e293b; border:1px solid #2d3748; border-radius:12px;
    padding:1.2rem 1.5rem; margin-bottom:1rem;
}
.card h4 { color:#e2e8f0; margin:0 0 0.5rem 0; font-size:0.9rem; font-weight:600; }

/* Rewrite box */
.rewrite-box {
    background: linear-gradient(135deg,#1a1f2e,#1e1040);
    border:1px solid #4f46e5; border-radius:12px;
    padding:1.2rem 1.5rem; color:#c7d2fe;
    font-size:1.05rem; font-style:italic; margin-top:0.5rem;
}

/* Metric override */
[data-testid="stMetric"] { background:#1e293b; border-radius:10px; padding:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    register = st.selectbox(
        "Context Register",
        options=available_registers(),
        index=1,
        help="Adjusts the preservation threshold (θ) based on communication context."
    )

    st.markdown("---")
    st.markdown("### 🔧 Scoring Weights")
    w1 = st.slider("w1 – Borrowing Frequency",  0.0, 1.0, 0.20, 0.05)
    w2 = st.slider("w2 – Domain Specificity",   0.0, 1.0, 0.20, 0.05)
    w3 = st.slider("w3 – Semantic Importance",  0.0, 1.0, 0.20, 0.05)
    w4 = st.slider("w4 – Context Consistency",  0.0, 1.0, 0.15, 0.05)
    w5 = st.slider("w5 – Audience Familiarity", 0.0, 1.0, 0.15, 0.05)
    w6 = st.slider("w6 – Translation Benefit",  0.0, 1.0, 0.10, 0.05)

    weights = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6}

    st.markdown("---")
    llm_backend = st.selectbox(
        "LLM Backend",
        ["mock", "openai", "ollama"],
        help="'mock' runs fully offline for demo purposes."
    )
    llm_model = None
    if llm_backend == "openai":
        llm_model = st.text_input("OpenAI Model", "gpt-4o-mini")
    elif llm_backend == "ollama":
        llm_model = st.text_input("Ollama Model", "llama3")

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>LAPS · VIT Semester Project 2026</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔬 LAPS</h1>
  <p><strong>Linguistic Adaptive Preservation Scoring</strong> — Explainable token-level decision making for code-mixed text rewriting</p>
</div>
""", unsafe_allow_html=True)

# ── Sample sentences ──────────────────────────────────────────────────────────
EXAMPLES = [
    "Kal client meeting postpone ho gayi, kya hum phir se schedule karein?",
    "Server ka debug karna padega, deadline kal hai.",
    "Aaj doctor ne kaha ki medicine leni padegi.",
    "Meeting mein presentation dena hai, feedback lena mat bhoolna.",
    "Project ka final report submit karna hai, koi issue hai kya?",
]

st.markdown("### ✍️ Input Sentence")
col_inp, col_btn = st.columns([5, 1])
with col_inp:
    sentence = st.text_area(
        "Enter a code-mixed sentence (Hinglish/Tanglish):",
        value=EXAMPLES[0],
        height=90,
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("Analyse ▶", use_container_width=True, type="primary")

with st.expander("💡 Try an example"):
    for i, ex in enumerate(EXAMPLES):
        if st.button(f"Example {i+1}", key=f"ex_{i}"):
            sentence = ex

# ─────────────────────────────────────────────────────────────────────────────
# Run LAPS
# ─────────────────────────────────────────────────────────────────────────────
if analyse or sentence:
    with st.spinner("Running LAPS pipeline..."):
        result = run(
            sentence=sentence,
            register=register,
            llm_backend=llm_backend,
            llm_model=llm_model,
            weights=weights,
        )

    tokens       = result["tokens"]
    preserve_list = result["preserve_list"]
    rewrite      = result["rewrite_result"]
    explanations = result["explanations"]
    cmi_info     = result.get("cmi_info", {})

    # Monolingual English Alert
    if result.get("is_monolingual"):
        st.info(
            "ℹ️ **Monolingual English Input Detected (Code-Mixing Index CMI = 0.0%)**\n\n"
            "This sentence is 100% monolingual English. LAPS is a code-mixed preservation framework designed for multilingual text (e.g. Hinglish/Tanglish). "
            "Since this input is already pure English, no code-mixed loanword preservation or translation is required."
        )

    # ── Token visualisation ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Token Decision Map")
    pills_html = ""
    for tok in tokens:
        d = tok.get("decision", "NATIVE")
        sc = f" ({tok['laps_score']:.2f})" if tok["laps_score"] is not None else ""
        cls = "pill-preserve" if d == "PRESERVE" else ("pill-rewrite" if d == "REWRITE" else "pill-native")
        pills_html += f'<span class="{cls}">{tok["token"]}{sc}</span> '
    st.markdown(f'<div style="line-height:2.2;">{pills_html}</div>', unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <small>
    <span class='pill-preserve'>PRESERVE</span> – kept in original form &nbsp;
    <span class='pill-rewrite'>REWRITE</span> – translated by LLM &nbsp;
    <span class='pill-native'>NATIVE</span> – native Hindi, not scored
    </small>
    """, unsafe_allow_html=True)

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown("---")
    cand_tokens  = [t for t in tokens if t["laps_score"] is not None]
    n_preserve   = sum(1 for t in tokens if t.get("decision") == "PRESERVE")
    n_rewrite    = sum(1 for t in tokens if t.get("decision") == "REWRITE")
    n_native     = sum(1 for t in tokens if t.get("decision") == "NATIVE")
    avg_score    = (sum(t["laps_score"] for t in cand_tokens) / len(cand_tokens)) if cand_tokens else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PRESERVE", n_preserve)
    m2.metric("REWRITE",  n_rewrite)
    m3.metric("NATIVE",   n_native)
    m4.metric("Avg Score", f"{avg_score:.3f}")

    # ── Feature radar for candidate tokens ──────────────────────────────────
    if cand_tokens:
        st.markdown("---")
        st.markdown("### 📊 Feature Scores per Token")

        radar_tokens = [t for t in cand_tokens if t.get("features")]
        if radar_tokens:
            sel_tok = st.selectbox(
                "Select token to inspect:",
                options=[t["token"] for t in radar_tokens],
            )
            sel = next(t for t in radar_tokens if t["token"] == sel_tok)
            feats = sel["features"]

            cats   = ["BF", "DS", "SI", "CC", "AF", "TB"]
            values = [feats.get(c, 0) for c in cats]
            values_closed = values + [values[0]]
            cats_closed   = cats + [cats[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill='toself',
                fillcolor='rgba(99,102,241,0.2)',
                line=dict(color='#818cf8', width=2),
                name=sel_tok,
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='#1e293b',
                    radialaxis=dict(visible=True, range=[0, 1],
                                    color='#64748b', gridcolor='#334155'),
                    angularaxis=dict(color='#94a3b8', gridcolor='#334155'),
                ),
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#94a3b8', family='Inter'),
                margin=dict(t=30, b=30, l=30, r=30),
                showlegend=False,
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Bar chart of all token scores
        df_scores = pd.DataFrame([
            {"Token": t["token"], "LAPS Score": t["laps_score"],
             "Decision": t.get("decision", "?")}
            for t in cand_tokens if t["laps_score"] is not None
        ])
        color_map = {"PRESERVE": "#4ade80", "REWRITE": "#f87171"}
        fig2 = go.Figure()
        for dec, color in color_map.items():
            sub = df_scores[df_scores["Decision"] == dec]
            fig2.add_trace(go.Bar(
                x=sub["Token"], y=sub["LAPS Score"],
                name=dec, marker_color=color,
            ))
        threshold_val = {"casual": 0.35, "business": 0.50, "academic": 0.65, "social": 0.40}.get(register, 0.5)
        fig2.add_hline(y=threshold_val, line_dash="dot", line_color="#f59e0b",
                       annotation_text=f"θ={threshold_val} ({register})",
                       annotation_font_color="#f59e0b")
        fig2.update_layout(
            paper_bgcolor='#0d1117', plot_bgcolor='#1e293b',
            font=dict(color='#94a3b8', family='Inter'),
            legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
            xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155', range=[0, 1]),
            height=320, margin=dict(t=20, b=20),
            barmode='group',
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Rewritten output ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✨ Rewritten Output")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Original (code-mixed):**")
        st.info(sentence)
    with col_b:
        st.markdown("**LAPS-guided rewrite:**")
        st.success(rewrite["rewritten"])

    st.markdown(f"**Preserved tokens:** `{', '.join(preserve_list) if preserve_list else 'None'}`")

    # ── Explanation table ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 Token-Level Explanations")

    rows = []
    for ex in explanations:
        rows.append({
            "Token":    ex["token"],
            "Decision": ex["decision"],
            "Score":    f"{ex['score']:.3f}" if ex["score"] is not None else "—",
            "θ (threshold)": f"{ex['threshold']:.2f}" if ex["threshold"] is not None else "—",
            "Primary Reason": ex["primary"],
        })

    df_exp = pd.DataFrame(rows)

    def _row_style(row):
        if row["Decision"] == "PRESERVE":
            return ['background-color:#1e3a2f; color:#4ade80'] * len(row)
        if row["Decision"] == "REWRITE":
            return ['background-color:#3b1f1f; color:#f87171'] * len(row)
        return ['color:#94a3b8'] * len(row)

    st.dataframe(
        df_exp.style.apply(_row_style, axis=1),
        use_container_width=True,
        height=300,
    )
