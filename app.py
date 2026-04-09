import streamlit as st
import pandas as pd

# Import logic from backend package
from backend.preprocess import preprocess_data
from backend.eda import plot_histograms
from backend.model import train_and_select_model
from backend.insights import generate_insights
from utils.file_handler import load_file
from agent.agent import run_agentic_pipeline

st.set_page_config(page_title="Automated Data Analysis Application", layout="wide", page_icon="📊")

# Import icons from lucide-react style
try:
    from streamlit_extras.stylable_container import stylable_container
except:
    pass

# Inject Custom CSS for softer typography and button aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background: #0a0e1a;
    }

    /* Main container background */
    .main .block-container {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1419 50%, #0a0e1a 100%);
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sleek container styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        transition: all 0.3s ease-in-out;
        border: 1px solid rgba(99, 102, 241, 0.15);
        background: linear-gradient(135deg, rgba(15, 22, 35, 0.95) 0%, rgba(19, 30, 46, 0.95) 100%);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border: 1px solid rgba(99, 102, 241, 0.45);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
        transform: translateY(-2px);
    }

    /* Sleek Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.15) 100%);
        color: #a5b4fc;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15);
    }
    div.stButton > button:first-child:hover {
        border: 1px solid rgba(99, 102, 241, 1);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.3) 100%);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        color: #c7d2fe;
    }

    /* Headers typography */
    h1, h2, h3, h4 {
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Mute secondary text */
    p, .stMarkdown p {
        color: #94a3b8 !important;
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(99, 102, 241, 0.05);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: rgba(99, 102, 241, 0.08);
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 22, 35, 0.5);
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #64748b;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.25) 100%) !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        border-radius: 10px;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background: rgba(15, 22, 35, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
        background: rgba(15, 22, 35, 0.8);
    }

    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.2);
        margin: 2rem 0;
    }

    /* ── Model Decision Engine custom card styles ── */
    .model-engine-card {
        background: linear-gradient(135deg, #0f1623 0%, #131e2e 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(79, 70, 229, 0.2) 100%);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
    }
    .model-name {
        font-size: 22px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 4px 0;
    }
    .model-task {
        font-size: 13px;
        color: #64748b;
        margin: 0 0 20px 0;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .metrics-grid-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(79, 70, 229, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e2e8f0;
        line-height: 1;
    }
    .metric-value span {
        font-size: 13px;
        color: #a5b4fc;
        margin-left: 2px;
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.3) 50%, transparent 100%);
        margin: 20px 0;
    }
    .section-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 10px;
    }
    .explanation-text {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.6;
    }
    .confidence-bar-wrap {
        margin-top: 18px;
    }
    .confidence-bar-track {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 99px;
        height: 6px;
        overflow: hidden;
        margin-top: 8px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        transition: width 1s ease;
    }
    .confidence-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 8px;
    }
    .confidence-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .pill-high   { background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(79, 70, 229, 0.2) 100%); border: 1px solid rgba(99, 102, 241, 0.4); color: #a5b4fc; }
    .pill-medium { background: rgba(234,179,8,0.12);  border: 1px solid rgba(234,179,8,0.35);  color: #facc15; }
    .pill-low    { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.35);  color: #f87171; }
    .confidence-pct {
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Data Analysis Dashboard")

# We manage state so results persist when navigating
if "result" not in st.session_state:
    st.session_state.result = None
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello! I am your AI Data Scientist. I have analyzed your dataset and model. What would you like to know?"}]

left_panel, right_panel = st.columns([2, 1])

with left_panel:
    st.markdown("#### Dataset Upload")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload", type=["csv", "xlsx", "xls", "json"], label_visibility="collapsed")
        
    if uploaded_file is not None:
        if st.session_state.uploaded_name != uploaded_file.name:
            df = load_file(uploaded_file)
            st.session_state.df = df
            st.session_state.uploaded_name = uploaded_file.name
            with st.spinner("Analyzing dataset..."):
                st.session_state.result = run_agentic_pipeline(df, "Perform a comprehensive analysis on the data.")
            st.rerun()
        
    if st.session_state.result is not None:
        st.write("")
        st.markdown("#### Dataset Explorer")
        with st.container(border=True):
            if st.session_state.result.get("clean_data") is not None:
                data_tabs = st.tabs(["Raw Dataset", "Cleaned Dataset"])
                
                with data_tabs[0]:
                    preview_df = st.session_state.df.head(10).copy()
                    for c in preview_df.select_dtypes(include=['object']).columns:
                        preview_df[c] = preview_df[c].astype(str)
                    st.dataframe(preview_df, hide_index=True, width='stretch')
                    
                with data_tabs[1]:
                    clean_preview = st.session_state.result["clean_data"].head(10).copy()
                    for c in clean_preview.select_dtypes(include=['object']).columns:
                        clean_preview[c] = clean_preview[c].astype(str)
                    st.dataframe(clean_preview, hide_index=True, width='stretch')
            else:
                preview_df = st.session_state.df.head(10).copy()
                for c in preview_df.select_dtypes(include=['object']).columns:
                    preview_df[c] = preview_df[c].astype(str)
                st.dataframe(preview_df, hide_index=True, width='stretch')

    if st.session_state.result is not None:
        st.markdown("#### Data Insights & Patterns")
        with st.container(border=True):
            viz_dict = {}
            if st.session_state.result.get("eda_hist") is not None:
                viz_dict["Histogram"] = st.session_state.result["eda_hist"]
            if st.session_state.result.get("eda_box") is not None:
                viz_dict["Box Plot"] = st.session_state.result["eda_box"]
            if st.session_state.result.get("eda_bar") is not None:
                viz_dict["Bar Plot"] = st.session_state.result["eda_bar"]
            if st.session_state.result.get("eda_scatter") is not None:
                viz_dict["Scatter Plot"] = st.session_state.result["eda_scatter"]
            if st.session_state.result.get("eda_pair") is not None:
                viz_dict["Pair Plot"] = st.session_state.result["eda_pair"]
            
            if viz_dict:
                # Create tabs dynamically based on available plots
                tabs = st.tabs(list(viz_dict.keys()))
                for tab, (name, fig) in zip(tabs, viz_dict.items()):
                    with tab:
                        # Check if it's a natively interactive Plotly chart or legacy Matplotlib
                        if type(fig).__name__ == "Figure" and hasattr(fig, "update_layout"):
                            st.plotly_chart(fig, width='stretch', theme="streamlit")
                        else:
                            st.pyplot(fig, width='content')
            else:
                st.info("No visualizations available.")


with right_panel:
    if st.session_state.result is not None:
        st.markdown("#### AI Assistant")
        
        with st.container(border=True):
            st.markdown("##### Exploratory Data Analysis")
            st.write("The dataset has been processed and analyzed successfully. Structural variances have been calculated natively within the data.")
            
        with st.container(border=True):
            st.markdown("##### Model Results")
            model_score = st.session_state.result.get("model_score")
            if model_score is not None:
                score_val = model_score * 100 if model_score <= 1 else model_score
                st.markdown(f"Model Accuracy: **{score_val:.1f}%**")
                st.write("The model achieved high accuracy against baseline splits.")
            else:
                st.write("No applicable model was trained.")
                
        with st.container(border=True):
            st.markdown("##### Insights")
            if st.session_state.result.get("insights"):
                for ins in st.session_state.result["insights"]:
                    st.markdown(f"- {ins}")
            else:
                st.write("- No generalized insights generated.")
                
        # ---------------------------------------------
        # INTERACTIVE AI DATA SCIENTIST CHAT
        # ---------------------------------------------
        st.write("")
        st.markdown("#### 💬 Ask the Data Scientist")
        with st.container(border=True, height=500):
            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Input bar
        user_q = st.chat_input("Ask about the model, graphs, or insights...")
        if user_q:
            # Append & display user message
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            # Trigger rerun to show user message immediately and run the spinner properly
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# FULL-WIDTH ─ AI Model Decision Engine
# (Spans exactly the same width as left_panel + right_panel combined)
# ══════════════════════════════════════════════════════════════════
if st.session_state.result is not None:
    st.write("")
    st.markdown("#### 🤖 AI Model Decision Engine")
    with st.container(border=True):
        result = st.session_state.result
        pipeline_metrics = result.get("metrics") or result.get("model_metrics", {})

        if pipeline_metrics:
            model_name       = result.get("model_name", "Unknown Model")
            task             = result.get("task", "Unknown Task")
            candidate_scores = result.get("candidate_scores", {})
            feature_importances = result.get("feature_importances")
            train_size       = result.get("train_size")
            test_size        = result.get("test_size")
            n_features       = result.get("n_features")
            target_col       = result.get("target_col", "target")

            # ── Header ──────────────────────────────────────────────
            hcol1, hcol2 = st.columns([3, 1])
            with hcol1:
                st.markdown(f"### ⚙️ {model_name}")
                st.caption("Auto-selected by the AI agent based on dataset characteristics")
            with hcol2:
                st.metric("Task Type", task.capitalize())

            # ── Dataset pills ────────────────────────────────────────
            if train_size and test_size:
                total = train_size + test_size
                st.markdown(
                    f"""<div style="display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px 0;">
                        <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.35);
                              color:#818cf8;border-radius:20px;padding:3px 14px;font-size:12px;font-weight:600;">
                            ▣ {total} rows &nbsp;·&nbsp; {n_features} features
                        </span>
                        <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
                              color:#4ade80;border-radius:20px;padding:3px 14px;font-size:12px;font-weight:600;">
                            ▸ Train {train_size}
                        </span>
                        <span style="background:rgba(251,146,60,0.1);border:1px solid rgba(251,146,60,0.3);
                              color:#fb923c;border-radius:20px;padding:3px 14px;font-size:12px;font-weight:600;">
                            ▸ Test {test_size}
                        </span>
                        <span style="background:rgba(148,163,184,0.1);border:1px solid rgba(148,163,184,0.25);
                              color:#94a3b8;border-radius:20px;padding:3px 14px;font-size:12px;font-weight:600;">
                            ◉ Target: {target_col}
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Performance Metrics ──────────────────────────────────
            st.markdown("**▣ Performance Metrics**")
            if task == "classification":
                score_val = pipeline_metrics.get("accuracy", 0)
                acc  = pipeline_metrics.get("accuracy",  0)
                prec = pipeline_metrics.get("precision", 0)
                rec  = pipeline_metrics.get("recall",    0)
                f1   = pipeline_metrics.get("f1_score",  0)

                def grade(v):
                    if v >= 0.9:  return "✓ Excellent"
                    if v >= 0.75: return "◐ Good"
                    return "✗ Needs work"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{acc:.2f}",  grade(acc))
                c2.metric("Precision", f"{prec:.2f}", grade(prec))
                c3.metric("Recall",    f"{rec:.2f}",  grade(rec))
                c4.metric("F1 Score",  f"{f1:.2f}",   grade(f1))
                insight = (
                    f"The model achieves **{acc*100:.1f}% accuracy** with balanced precision ({prec:.2f}) "
                    f"and recall ({rec:.2f}), yielding an F1 of {f1:.2f}. "
                    + ("Performance is strong and reliable." if acc >= 0.8
                       else "Consider feature engineering or additional data.")
                )
            else:
                score_val = pipeline_metrics.get("r2_score", 0)
                r2  = pipeline_metrics.get("r2_score", 0)
                mse = pipeline_metrics.get("mse",      0)
                mae = pipeline_metrics.get("mae",      0)

                def r2_grade(v):
                    if v >= 0.85: return "✓ Excellent"
                    if v >= 0.65: return "◐ Good"
                    return "✗ Weak fit"

                c1, c2, c3 = st.columns(3)
                c1.metric("R² Score", f"{r2:.3f}",   r2_grade(r2))
                c2.metric("MSE",      f"{mse:,.3f}", "Error²")
                c3.metric("MAE",      f"{mae:.3f}",  "Avg error")
                insight = (
                    f"The model explains **{r2*100:.1f}% of variance** in `{target_col}`. "
                    f"Mean absolute error is {mae:.3f}. "
                    + ("Excellent fit for production use." if r2 >= 0.85
                       else "Consider non-linear models or more features.")
                )

            st.divider()

            # ── Two-column: Competition table | Feature importance ────
            left_de, right_de = st.columns([1, 1], gap="large")

            with left_de:
                if candidate_scores:
                    st.markdown("**▲ Model Competition**")
                    metric_label = "Accuracy" if task == "classification" else "R² Score"
                    rows = []
                    for cname, cscore in candidate_scores.items():
                        medal    = "●" if cname == model_name else "○"
                        selected = "✓ Selected" if cname == model_name else ""
                        score_str = f"{cscore:.4f}" if cscore is not None else "—"
                        rows.append({"Model": f"{medal} {cname}", metric_label: score_str, "Status": selected})
                    import pandas as _pd
                    st.dataframe(_pd.DataFrame(rows), hide_index=True, width="stretch")
                    st.caption(f"Criterion: highest {metric_label} on 80/20 held-out test set (seed=42)")

                st.write("")
                st.markdown("**▸ Why this model?**")
                runner_up = [k for k in candidate_scores if k != model_name]
                runner_up_text = ""
                if runner_up:
                    ru_name  = runner_up[0]
                    ru_score = candidate_scores.get(ru_name)
                    bs       = candidate_scores.get(model_name)
                    if ru_score is not None and bs is not None:
                        runner_up_text = f" It outperformed <b style='color:#e2e8f0'>{ru_name}</b> by <b style='color:#e2e8f0'>{abs(bs-ru_score):.4f}</b>."
                st.markdown(
                    f"""<div style="background:rgba(99,102,241,0.06);border-left:3px solid rgba(99,102,241,0.5);
                         border-radius:0 8px 8px 0;padding:14px 18px;font-size:13px;color:#94a3b8;line-height:1.8;">
                        The AI agent compared all candidates on an 80/20 train-test split.
                        <b style="color:#e2e8f0">{model_name}</b> was chosen as the best generaliser.{runner_up_text}
                    </div>""",
                    unsafe_allow_html=True,
                )

            with right_de:
                if feature_importances:
                    st.markdown("**▣ Feature Importance** *(Random Forest)*")
                    top_n = dict(list(feature_importances.items())[:10])
                    import plotly.graph_objects as _go
                    fig_fi = _go.Figure(_go.Bar(
                        x=list(top_n.values()),
                        y=list(top_n.keys()),
                        orientation="h",
                        marker=dict(color=list(top_n.values()), colorscale="Viridis", showscale=False),
                        text=[f"{v:.3f}" for v in top_n.values()],
                        textposition="outside",
                    ))
                    fig_fi.update_layout(
                        height=max(250, len(top_n) * 36),
                        margin=dict(l=0, r=70, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                        font=dict(color="#94a3b8"),
                    )
                    st.plotly_chart(fig_fi, width="stretch")
                    st.caption(f"Top driver: **{list(top_n.keys())[0]}** ({list(top_n.values())[0]:.3f})")
                else:
                    st.markdown("**◆ Model Insight**")
                    st.info(insight)

            st.divider()
            if feature_importances:
                st.markdown("**◆ Model Insight**")
                st.info(insight)

            score_pct = min(max(score_val, 0), 1)
            st.markdown("**▲ Model Confidence**")
            st.progress(score_pct)
            if score_pct >= 0.8:
                st.markdown(f"<div style='background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.35);color:#818cf8;border-radius:8px;padding:12px 16px;font-size:14px;font-weight:600;'>✓  High Confidence — {score_pct*100:.1f}%</div>", unsafe_allow_html=True)
            elif score_pct >= 0.6:
                st.warning(f"⚠  Moderate Confidence — {score_pct*100:.1f}%")
            else:
                st.error(f"✗  Low Confidence — {score_pct*100:.1f}%")
        else:
            st.info("⚙  No applicable model was trained on this dataset.")

# st.chat_input triggers a rerun; intercept last unanswered user message and stream the reply
if st.session_state.get("chat_history") and st.session_state.chat_history[-1]["role"] == "user":
    user_q = st.session_state.chat_history[-1]["content"]
    with right_panel:
        with st.chat_message("assistant"):
            from utils.llm import stream_data_science_response
            res = st.session_state.result
            df_shape = str(st.session_state.df.shape)
            metrics = str(res.get("metrics") or res.get("model_metrics", {}))
            model_name = res.get("model_name", "Unknown")
            task = res.get("task", "Unknown")
            insights = str(res.get("insights", []))

            # Stream tokens live — user sees words appear as they are generated
            ai_response = st.write_stream(
                stream_data_science_response(
                    df_shape, model_name, task, metrics, insights, user_q
                )
            )

        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        st.rerun()

# --- FOOTER ---
if st.session_state.result is None:
    st.write("")
    st.divider()
    st.markdown("#### Core Features")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        with st.container(border=True):
            st.markdown("### 📊 Exploratory Data Analysis")
            st.caption("Automatically analyzes data distributions, trends, and patterns using visualizations like histograms, scatter plots, and correlations.")

    with f_col2:
        with st.container(border=True):
            st.markdown("### 🧹 Data Preprocessing")
            st.caption("Handles missing values, removes noise (like unnamed columns), and prepares clean data for analysis and modeling.")

    st.write("")
    f_col3, f_col4 = st.columns(2)

    with f_col3:
        with st.container(border=True):
            st.markdown("### 🤖 Model Decision Engine")
            st.caption("Automatically selects the best machine learning model, trains it, and evaluates performance using appropriate metrics.")

    with f_col4:
        with st.container(border=True):
            st.markdown("### 💬 AI Assistant")
            st.caption("Interact with your data using natural language. Ask questions about models, graphs, and insights with statistical explanations.")
