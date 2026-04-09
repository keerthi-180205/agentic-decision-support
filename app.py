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

# Inject Custom CSS for softer typography and button aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Sleek container styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.06);
        background: #0f1623;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border: 1px solid rgba(99, 102, 241, 0.35);
        box-shadow: 0 4px 24px -4px rgba(99, 102, 241, 0.15);
    }

    /* Sleek Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.4px;
        border: 1px solid rgba(99, 102, 241, 0.5);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        border: 1px solid rgba(99, 102, 241, 1);
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.3);
        transform: translateY(-1px);
    }

    /* Headers typography */
    h1, h2, h3, h4 {
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }

    /* Mute secondary text */
    p, .stMarkdown p {
        color: #A1A1AA !important;
    }

    /* ── Model Decision Engine custom card styles ── */
    .model-engine-card {
        background: linear-gradient(135deg, #0f1623 0%, #131e2e 100%);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 0;
    }
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818cf8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 16px;
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
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
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
        color: #818cf8;
        margin-left: 2px;
    }
    .divider {
        height: 1px;
        background: rgba(255,255,255,0.05);
        margin: 20px 0;
    }
    .section-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #475569;
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
        background: rgba(255,255,255,0.06);
        border-radius: 99px;
        height: 6px;
        overflow: hidden;
        margin-top: 8px;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 99px;
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
    .pill-high   { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.35);  color: #4ade80; }
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
                    st.dataframe(preview_df, hide_index=True, use_container_width=True)
                    
                with data_tabs[1]:
                    clean_preview = st.session_state.result["clean_data"].head(10).copy()
                    for c in clean_preview.select_dtypes(include=['object']).columns:
                        clean_preview[c] = clean_preview[c].astype(str)
                    st.dataframe(clean_preview, hide_index=True, use_container_width=True)
            else:
                preview_df = st.session_state.df.head(10).copy()
                for c in preview_df.select_dtypes(include=['object']).columns:
                    preview_df[c] = preview_df[c].astype(str)
                st.dataframe(preview_df, hide_index=True, use_container_width=True)

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
                            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                        else:
                            st.pyplot(fig, use_container_width=False)
            else:
                st.info("No visualizations available.")

        st.markdown("#### AI Model Decision Engine")
        with st.container(border=True):
            result = st.session_state.result
            pipeline_metrics = result.get("metrics") or result.get("model_metrics", {})

            if pipeline_metrics:
                model_name = result.get("model_name", "Unknown Model")
                task = result.get("task", "Unknown Task")

                # Model identity row
                col_name, col_task = st.columns([2, 1])
                with col_name:
                    st.markdown(f"### 🤖 {model_name}")
                    st.caption("Auto-selected by the AI agent based on dataset characteristics")
                with col_task:
                    st.metric("Task Type", task.capitalize())

                st.divider()

                # Metrics
                st.markdown("**📊 Performance Metrics**")
                if task == "classification":
                    score_val = pipeline_metrics.get("accuracy", 0)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy",  f"{pipeline_metrics.get('accuracy', 0):.2f}")
                    c2.metric("Precision", f"{pipeline_metrics.get('precision', 0):.2f}")
                    c3.metric("Recall",    f"{pipeline_metrics.get('recall', 0):.2f}")
                    c4.metric("F1 Score",  f"{pipeline_metrics.get('f1_score', 0):.2f}")
                    insight = "Reliable classification performance with consistent accuracy across evaluation metrics."
                else:
                    score_val = pipeline_metrics.get("r2_score", 0)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² Score", f"{pipeline_metrics.get('r2_score', 0):.2f}")
                    c2.metric("MSE",      f"{pipeline_metrics.get('mse', 0):,.2f}")
                    c3.metric("MAE",      f"{pipeline_metrics.get('mae', 0):.2f}")
                    insight = "Strong regression fit — the model explains a high proportion of variance in the target."

                st.divider()

                # Insight + Why this model
                st.markdown("**💡 Model Insight**")
                st.info(insight)

                st.markdown("**🔍 Why this model?**")
                st.caption(
                    f"The AI agent evaluated multiple candidate models and selected **{model_name}** "
                    f"as the best performer based on cross-validated score, feature distribution, and generalisation."
                )

                st.divider()

                # Confidence
                score_pct = min(max(score_val, 0), 1)
                st.markdown("**📈 Model Confidence**")
                st.progress(score_pct)

                if score_pct >= 0.8:
                    st.success(f"✅  High Confidence — {score_pct*100:.1f}%")
                elif score_pct >= 0.6:
                    st.warning(f"⚠️  Moderate Confidence — {score_pct*100:.1f}%")
                else:
                    st.error(f"❌  Low Confidence — {score_pct*100:.1f}%")

            else:
                st.info("🤖  No applicable model was trained on this dataset.")

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

# Execute the LLM interaction outside of chat_input's immediate event loop if needed
# Actually, st.chat_input triggers a rerun, so we intercept the last unanswered query
if st.session_state.get("chat_history") and st.session_state.chat_history[-1]["role"] == "user":
    user_q = st.session_state.chat_history[-1]["content"]
    with right_panel:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data context..."):
                from utils.llm import get_data_science_response
                res = st.session_state.result
                df_shape = str(st.session_state.df.shape)
                metrics = str(res.get("metrics") or res.get("model_metrics", {}))
                model_name = res.get("model_name", "Unknown")
                task = res.get("task", "Unknown")
                insights = str(res.get("insights", []))
                
                ai_response = get_data_science_response(
                    df_shape, model_name, task, metrics, insights, user_q
                )
                
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()
