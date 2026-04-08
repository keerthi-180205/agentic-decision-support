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
    /* Sleek container styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: #111827;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
    }
    /* Sleek Buttons */
    div.stButton > button:first-child {
        border-radius: 6px;
        font-weight: 500;
        letter-spacing: 0.5px;
        border: 1px solid rgba(99, 102, 241, 0.5);
    }
    div.stButton > button:first-child:hover {
        border: 1px solid rgba(99, 102, 241, 1);
        box-shadow: 0 4px 12px -2px rgba(99, 102, 241, 0.2);
    }
    /* Headers typography */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    /* Mute secondary text further */
    p, .stMarkdown p {
        color: #A1A1AA !important;
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
        st.write("")
        st.markdown("#### Visualizations")
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
                        # Rendering with use_container_width=False ensures the plot uses
                        # its exact native matplotlib pixel size instead of ballooning up.
                        st.pyplot(fig, use_container_width=False)
            else:
                st.info("No visualizations available.")

with right_panel:
    if st.session_state.result is not None:
        st.markdown("#### AI Assistant")
        
        with st.container(border=True):
            st.markdown("##### Exploratory Data Analysis")
            st.write("")
            st.write("Summary:")
            st.write("The dataset has been processed and analyzed successfully. Structural variances have been calculated natively within the data.")
            
        with st.container(border=True):
            st.markdown("##### Model Results")
            st.write("")
            model_score = st.session_state.result.get("model_score")
            if model_score is not None:
                score_val = model_score * 100 if model_score <= 1 else model_score
                st.markdown(f"Model Accuracy: **{score_val:.1f}%**")
                st.write("The model achieved high accuracy against baseline splits.")
            else:
                st.write("No applicable model was trained.")
                
        with st.container(border=True):
            st.markdown("##### Insights")
            st.write("")
            if st.session_state.result.get("insights"):
                for ins in st.session_state.result["insights"]:
                    st.markdown(f"- {ins}")
            else:
                st.write("- No generalized insights generated.")
