"""
LAPS Interactive Demonstration - Streamlit Application
======================================================
Explainable Token-Level Decision Making for Code-Mixed Text Rewriting.
Run: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.pipeline import run
from src.laps.threshold_engine import available_registers, REGISTER_THRESHOLDS
from src.llm.explainer import format_explanation_table

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LAPS - Linguistic Adaptive Preservation Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Professional CSS Styles (No Emojis)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & Main Container */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
}

/* Header Banner */
.hero-container {
    background: linear-gradient(135deg, #111827 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.5;
}

/* Status Badges & Pills */
.pill-preserve {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid #059669;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}

.pill-uncertain {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid #d97706;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}

.pill-rewrite {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid #dc2626;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}

.pill-native {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(51, 65, 85, 0.4);
    color: #94a3b8;
    border: 1px solid #475569;
    font-size: 0.82rem;
    font-weight: 400;
    margin: 3px;
}

.tag-cmi {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}

/* Card Containers */
.panel-card {
    background: #131d31;
    border: 1px solid #24344d;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.panel-title {
    color: #f1f5f9;
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

/* Output Display Box */
.rewrite-panel {
    background: #0f172a;
    border: 1px solid #3b82f6;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    color: #f8fafc;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-top: 0.25rem;
}

/* Metric Container Customization */
div[data-testid="stMetric"] {
    background-color: #131d31;
    border: 1px solid #24344d;
    border-radius: 8px;
    padding: 0.85rem 1rem;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0d1322;
    border-right: 1px solid #1e293b;
}

/* Button Styling */
.stButton > button {
    border-radius: 6px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Curated Evaluation Sentences
# -----------------------------------------------------------------------------
SAMPLE_SENTENCES = [
    {
        "label": "Academic - Viva Link Query",
        "text": "hi! Kisi ke pass viva calender ki link h kya?",
        "register": "academic",
    },
    {
        "label": "Business - Client Meeting Postponement",
        "text": "Kal client meeting postpone ho gayi, kya hum phir se schedule karein?",
        "register": "business",
    },
    {
        "label": "Technical - Server Debugging in Production",
        "text": "Server ka debug karna padega, production mein crash ho raha hai.",
        "register": "business",
    },
    {
        "label": "Academic - Machine Learning Model Accuracy",
        "text": "Machine learning model ka accuracy improve karo, deadline kal hai.",
        "register": "academic",
    },
    {
        "label": "Casual - Wi-Fi Password Request",
        "text": "Wi-Fi ka password kya hai? Phone connect nahi ho raha.",
        "register": "casual",
    },
    {
        "label": "Social - Instagram Story Post",
        "text": "Instagram pe story daalni hai, filter kaun sa lagaun?",
        "register": "social",
    },
]

# Initialize Session State
if "sentence_input" not in st.session_state:
    st.session_state.sentence_input = SAMPLE_SENTENCES[0]["text"]
if "selected_register" not in st.session_state:
    st.session_state.selected_register = "academic"

# -----------------------------------------------------------------------------
# Sidebar Configuration Panel
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Pipeline Configuration")
    
    # Register Selector
    reg_options = available_registers()
    current_reg_idx = reg_options.index(st.session_state.selected_register) if st.session_state.selected_register in reg_options else 0
    
    register = st.selectbox(
        "Communicative Register",
        options=reg_options,
        index=current_reg_idx,
        help="Sets the baseline preservation threshold (theta_R) based on communication formality."
    )
    st.session_state.selected_register = register

    theta_current = REGISTER_THRESHOLDS.get(register, 0.50)
    st.caption(f"Baseline Preservation Threshold (theta): **{theta_current:.2f}**")

    st.markdown("---")
    st.markdown("### Linguistic Scoring Weights")
    
    # Weight Presets
    preset = st.selectbox(
        "Weight Configuration Mode",
        ["Default Balanced", "Optimized (w* Business)", "Domain Specificity Priority", "Custom"],
        index=0
    )

    if preset == "Optimized (w* Business)":
        def_w1, def_w2, def_w3, def_w4, def_w5, def_w6 = 0.15, 0.30, 0.20, 0.15, 0.10, 0.10
    elif preset == "Domain Specificity Priority":
        def_w1, def_w2, def_w3, def_w4, def_w5, def_w6 = 0.10, 0.40, 0.20, 0.10, 0.10, 0.10
    else:
        def_w1, def_w2, def_w3, def_w4, def_w5, def_w6 = 0.20, 0.20, 0.20, 0.15, 0.15, 0.10

    w1 = st.slider("w1 - Borrowing Frequency (BF)", 0.0, 1.0, def_w1, 0.05)
    w2 = st.slider("w2 - Domain Specificity (DS)",  0.0, 1.0, def_w2, 0.05)
    w3 = st.slider("w3 - Semantic Importance (SI)", 0.0, 1.0, def_w3, 0.05)
    w4 = st.slider("w4 - Context Consistency (CC)", 0.0, 1.0, def_w4, 0.05)
    w5 = st.slider("w5 - Audience Familiarity (AF)", 0.0, 1.0, def_w5, 0.05)
    w6 = st.slider("w6 - Translation Benefit (TB)", 0.0, 1.0, def_w6, 0.05)

    w_sum = w1 + w2 + w3 + w4 + w5 + w6
    if abs(w_sum - 1.0) > 0.01:
        st.warning(f"Total weight sum = {w_sum:.2f} (Recommended sum = 1.00)")
    else:
        st.success(f"Weights normalized (Sum = {w_sum:.2f})")

    weights = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6}

    st.markdown("---")
    st.markdown("### Downstream Rewriter Backend")
    llm_backend = st.selectbox(
        "LLM Backend Engine",
        ["mock", "openai", "ollama"],
        index=0,
        help="'mock' operates fully offline with rule-based grammar preservation."
    )
    llm_model = None
    if llm_backend == "openai":
        llm_model = st.text_input("OpenAI Model Name", "gpt-4o-mini")
    elif llm_backend == "ollama":
        llm_model = st.text_input("Local Ollama Model", "llama3")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b; line-height:1.4;'>"
        "<strong>LAPS Research Framework</strong><br>"
        "Knowledge Graph: 2,907 Nodes | 2,940 Edges<br>"
        "VIT University - Semester Project 2026"
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Main Header Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">LAPS: Linguistic Adaptive Preservation Scoring</div>
    <div class="hero-subtitle">
        Explainable Token-Level Decision Making for Code-Mixed Text Rewriting via Bilingual Knowledge Graphs and Soft Sigmoid Thresholding
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Input & Sample Selection
# -----------------------------------------------------------------------------
st.markdown("#### Input Sentence")

# Sample Picker Row
sample_cols = st.columns(len(SAMPLE_SENTENCES))
for idx, sample in enumerate(SAMPLE_SENTENCES):
    if sample_cols[idx].button(f"Sample {idx+1}", key=f"btn_sample_{idx}", use_container_width=True):
        st.session_state.sentence_input = sample["text"]
        st.session_state.selected_register = sample["register"]
        st.rerun()

col_text, col_run = st.columns([5, 1])
with col_text:
    input_text = st.text_area(
        "Code-Mixed Sentence (Hinglish):",
        value=st.session_state.sentence_input,
        height=85,
        label_visibility="collapsed",
        key="main_input_area"
    )
with col_run:
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    execute_pipeline = st.button("Analyze Sentence", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Pipeline Execution
# -----------------------------------------------------------------------------
if input_text:
    with st.spinner("Processing through LAPS linguistic pipeline..."):
        result = run(
            sentence=input_text,
            register=register,
            llm_backend=llm_backend,
            llm_model=llm_model,
            weights=weights,
        )

    tokens        = result["tokens"]
    preserve_list = result["preserve_list"]
    rewrite       = result["rewrite_result"]
    explanations  = result["explanations"]
    cmi_info      = result.get("cmi_info", {})
    is_mono       = result.get("is_monolingual", False)

    # Monolingual English Alert
    if is_mono:
        st.info(
            "Monolingual English Input Detected (Code-Mixing Index CMI = 0.0%). "
            "The input is already fully monolingual. No code-mixed preservation constraints or translations are required."
        )

    # -------------------------------------------------------------------------
    # Section 1: Token Decision Map
    # -------------------------------------------------------------------------
    st.markdown("---")
    col_hdr, col_cmi = st.columns([3, 2])
    with col_hdr:
        st.markdown("#### Token Decision Map")
    with col_cmi:
        cmi_val = cmi_info.get("cmi", 0.0)
        mode_label = cmi_info.get("mode", "CODE_MIXED")
        st.markdown(
            f"<div style='text-align:right; font-size:0.85rem; color:#94a3b8; padding-top:4px;'>"
            f"Language Mode: <span style='color:#38bdf8; font-weight:600;'>{mode_label}</span> | "
            f"CMI: <span style='color:#38bdf8; font-weight:600;'>{cmi_val:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    pills_html = ""
    for tok in tokens:
        dec = tok.get("decision", "NATIVE")
        prob = tok.get("probability")
        prob_str = f" P={prob:.2f}" if prob is not None else ""
        sc = f" [{tok['laps_score']:.2f}{prob_str}]" if tok.get("laps_score") is not None else ""
        border_flag = " *" if tok.get("is_borderline") else ""

        if dec == "PRESERVE":
            cls = "pill-preserve"
        elif dec == "UNCERTAIN":
            cls = "pill-uncertain"
        elif dec == "REWRITE":
            cls = "pill-rewrite"
        else:
            cls = "pill-native"

        pills_html += f'<span class="{cls}">{tok["token"]}{sc}{border_flag}</span> '

    st.markdown(f'<div style="line-height:2.6; margin: 0.5rem 0 1rem 0;">{pills_html}</div>', unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style="font-size:0.78rem; color:#94a3b8; display:flex; gap:1.5rem; flex-wrap:wrap;">
        <div><span class="pill-preserve">PRESERVE</span> Retain original loanword (P &ge; 0.70)</div>
        <div><span class="pill-uncertain">UNCERTAIN</span> Borderline candidate; preserved with flag (0.50 &le; P &lt; 0.70)</div>
        <div><span class="pill-rewrite">REWRITE</span> Eligible for natural translation (P &lt; 0.50)</div>
        <div><span class="pill-native">NATIVE</span> Native grammar token (non-candidate)</div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Section 2: Summary Metrics
    # -------------------------------------------------------------------------
    st.markdown("---")
    candidates = [t for t in tokens if t.get("laps_score") is not None]
    n_pres = sum(1 for t in tokens if t.get("decision") == "PRESERVE")
    n_unc  = sum(1 for t in tokens if t.get("decision") == "UNCERTAIN")
    n_rew  = sum(1 for t in tokens if t.get("decision") == "REWRITE")
    n_nat  = sum(1 for t in tokens if t.get("decision") == "NATIVE")
    
    probs = [t["probability"] for t in candidates if t.get("probability") is not None]
    avg_p = sum(probs) / len(probs) if probs else 0.0
    avg_sc = sum(t["laps_score"] for t in candidates) / len(candidates) if candidates else 0.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Preserved", n_pres)
    m2.metric("Uncertain", n_unc)
    m3.metric("Rewritten", n_rew)
    m4.metric("Native Tokens", n_nat)
    m5.metric("Avg LAPS Score", f"{avg_sc:.3f}")
    m6.metric("Avg P(Preserve)", f"{avg_p:.3f}")

    # -------------------------------------------------------------------------
    # Section 3: Dual Output Comparison
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Rewritten Output Comparison")
    
    col_orig, col_rew = st.columns(2)
    with col_orig:
        st.markdown("**Original Input (Code-Mixed)**")
        st.markdown(f'<div class="rewrite-panel" style="border-color:#475569; border-left-color:#64748b;">{input_text}</div>', unsafe_allow_html=True)
    with col_rew:
        st.markdown("**LAPS-Guided English Rewrite**")
        st.markdown(f'<div class="rewrite-panel">{rewrite["rewritten"]}</div>', unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:0.85rem; color:#94a3b8; margin-top:0.6rem;'>"
        f"Preserved Entities: <span class='mono' style='color:#34d399;'>{', '.join(preserve_list) if preserve_list else 'None'}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # Section 4: Graph-Enriched XML Prompt Inspector
    # -------------------------------------------------------------------------
    xml_prompt = rewrite.get("xml_prompt", "")
    if xml_prompt and xml_prompt != "Monolingual English - No rewriting required.":
        with st.expander("Inspect Graph-Enriched XML Prompt (Structured LLM Constraint)"):
            st.code(xml_prompt, language="xml")

    # -------------------------------------------------------------------------
    # Section 5: Feature Analysis & Visualizations
    # -------------------------------------------------------------------------
    if candidates:
        st.markdown("---")
        st.markdown("#### Feature Analysis and Probability Distribution")

        tab_bars, tab_radar = st.tabs(["Score and Probability Distribution", "Hexagonal Feature Radar"])

        with tab_bars:
            df_chart = pd.DataFrame([
                {
                    "Token": t["token"],
                    "LAPS Score": t["laps_score"],
                    "P(Preserve)": t.get("probability") or 0.0,
                    "Decision": t.get("decision", "NATIVE")
                }
                for t in candidates
            ])

            color_discrete = {
                "PRESERVE": "#10b981",
                "UNCERTAIN": "#f59e0b",
                "REWRITE": "#ef4444"
            }

            fig_bar = go.Figure()

            # Bars for LAPS Score grouped by decision
            for dec_label, col_hex in color_discrete.items():
                subset = df_chart[df_chart["Decision"] == dec_label]
                if not subset.empty:
                    fig_bar.add_trace(go.Bar(
                        x=subset["Token"],
                        y=subset["LAPS Score"],
                        name=f"{dec_label} (Score)",
                        marker_color=col_hex,
                        opacity=0.85,
                    ))

            # Scatter Line for P(Preserve)
            fig_bar.add_trace(go.Scatter(
                x=df_chart["Token"],
                y=df_chart["P(Preserve)"],
                mode='lines+markers',
                name='P(Preserve)',
                line=dict(color='#a855f7', width=2.5, dash='dot'),
                marker=dict(size=8, color='#c084fc'),
            ))

            # Reference Threshold Lines
            fig_bar.add_hline(
                y=theta_current,
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text=f"Register Baseline theta={theta_current:.2f} ({register})",
                annotation_font_color="#f59e0b",
                annotation_position="top left"
            )
            fig_bar.add_hline(
                y=0.70,
                line_dash="dash",
                line_color="#10b981",
                annotation_text="P=0.70 (Preserve Boundary)",
                annotation_font_color="#10b981",
                annotation_position="top right"
            )
            fig_bar.add_hline(
                y=0.50,
                line_dash="dash",
                line_color="#fbbf24",
                annotation_text="P=0.50 (Uncertain Boundary)",
                annotation_font_color="#fbbf24",
                annotation_position="bottom right"
            )

            fig_bar.update_layout(
                paper_bgcolor='#0b0f19',
                plot_bgcolor='#111827',
                font=dict(color='#94a3b8', family='Inter'),
                legend=dict(bgcolor='#131d31', bordercolor='#24344d'),
                xaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#f1f5f9')),
                yaxis=dict(gridcolor='#1e293b', range=[0, 1.05], tickfont=dict(color='#f1f5f9')),
                height=370,
                margin=dict(t=25, b=25, l=40, r=40),
                barmode='group',
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab_radar:
            radar_candidates = [t for t in candidates if t.get("features")]
            if radar_candidates:
                col_sel, col_plot = st.columns([1, 3])
                with col_sel:
                    selected_token_name = st.selectbox(
                        "Select Candidate Token:",
                        options=[t["token"] for t in radar_candidates],
                        key="radar_token_selector"
                    )
                    selected_tok = next(t for t in radar_candidates if t["token"] == selected_token_name)
                    st.markdown(
                        f"<div class='panel-card' style='margin-top:1rem;'>"
                        f"<div class='panel-title'>Token Details</div>"
                        f"<strong>Decision:</strong> {selected_tok.get('decision')}<br>"
                        f"<strong>Score:</strong> {selected_tok.get('laps_score'):.3f}<br>"
                        f"<strong>P(Preserve):</strong> {selected_tok.get('probability'):.3f}<br>"
                        f"<strong>Domain:</strong> {selected_tok.get('domain', 'General')}<br>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col_plot:
                    f_dict = selected_tok["features"]
                    categories = ["BF", "DS", "SI", "CC", "AF", "TB"]
                    vals = [f_dict.get(c, 0.0) for c in categories]
                    vals_closed = vals + [vals[0]]
                    cats_closed = categories + [categories[0]]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals_closed,
                        theta=cats_closed,
                        fill='toself',
                        fillcolor='rgba(59, 130, 246, 0.25)',
                        line=dict(color='#3b82f6', width=2),
                        name=selected_token_name,
                    ))

                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor='#111827',
                            radialaxis=dict(visible=True, range=[0, 1], color='#64748b', gridcolor='#1e293b'),
                            angularaxis=dict(color='#94a3b8', gridcolor='#1e293b'),
                        ),
                        paper_bgcolor='#0b0f19',
                        plot_bgcolor='#0b0f19',
                        font=dict(color='#94a3b8', family='Inter'),
                        margin=dict(t=20, b=20, l=30, r=30),
                        showlegend=False,
                        height=340,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

    # -------------------------------------------------------------------------
    # Section 6: Structured Explanation Table & Export
    # -------------------------------------------------------------------------
    st.markdown("---")
    col_tbl_hdr, col_tbl_dl = st.columns([3, 1])
    with col_tbl_hdr:
        st.markdown("#### Token-Level Explanations and Audit Trail")

    rows_data = []
    for ex in explanations:
        prob_val = ex.get("probability")
        sc_val = ex.get("score")
        th_val = ex.get("threshold")
        rows_data.append({
            "Token":          ex["token"],
            "Decision":       ex["decision"],
            "LAPS Score":     f"{sc_val:.3f}" if sc_val is not None else "-",
            "P(Preserve)":   f"{prob_val:.3f}" if prob_val is not None else "-",
            "Threshold":      f"{th_val:.2f}" if th_val is not None else "-",
            "Borderline":     "Yes" if ex.get("is_borderline") else "No",
            "Domain":         ex.get("domain", "-"),
            "Primary Justification": ex["primary"],
        })

    df_export = pd.DataFrame(rows_data)

    with col_tbl_dl:
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Report",
            data=csv_bytes,
            file_name=f"laps_explanation_{register}.csv",
            mime="text/csv",
            use_container_width=True
        )

    def _style_row(row):
        dec = row["Decision"]
        if dec == "PRESERVE":
            return ['background-color: rgba(16, 185, 129, 0.12); color: #34d399'] * len(row)
        elif dec == "UNCERTAIN":
            return ['background-color: rgba(245, 158, 11, 0.12); color: #fbbf24'] * len(row)
        elif dec == "REWRITE":
            return ['background-color: rgba(239, 68, 68, 0.12); color: #f87171'] * len(row)
        return ['color: #94a3b8'] * len(row)

    st.dataframe(
        df_export.style.apply(_style_row, axis=1),
        use_container_width=True,
        height=300,
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem; padding: 1rem 0;'>"
    "LAPS: Linguistic Adaptive Preservation Scoring | School of Computer Science and Engineering, VIT University"
    "</div>",
    unsafe_allow_html=True
)
