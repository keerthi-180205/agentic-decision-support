import streamlit as st
import pandas as pd

from backend.preprocess import preprocess_data
from backend.eda import plot_histograms, plot_correlation, show_basic_info
from backend.model import train_and_select_model
from backend.insights import generate_insights
from utils.file_handler import load_file
from agent.agent import run_agentic_pipeline as run_agent

st.set_page_config(page_title="Automated Data Analysis Application", layout="wide", page_icon="🚀")

st.title("🚀 Data Analysis & Insights Pipeline")
st.markdown("Upload your dataset and ask questions to automatically process, analyze, and model your data.")

uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    df = load_file(uploaded_file)

    if df is None or df.empty:
        st.error("Failed to load file. Please upload a valid CSV, Excel, or JSON file.")
        st.stop()

    st.subheader("📂 Dataset Preview")
    preview_df = df.head().copy()
    for col in preview_df.select_dtypes(include=['object']).columns:
        preview_df[col] = preview_df[col].astype(str)
    st.dataframe(preview_df, use_container_width=True)

    query = st.text_input("💬 Ask your data question (e.g. 'predict sales', 'visualize data')")

    if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
        st.markdown("---")

        with st.spinner("Agent is analyzing your request..."):
            result = run_agent(df, query)

        # 1. Cleaned Data
        if result.get("clean_data") is not None:
            st.subheader("🧹 1. Cleaned Data")
            clean_preview = result["clean_data"].head().copy()
            for col in clean_preview.select_dtypes(include=['object']).columns:
                clean_preview[col] = clean_preview[col].astype(str)
            st.dataframe(clean_preview, use_container_width=True)

        # 2. Basic Info
        if result.get("basic_info") is not None:
            st.subheader("📋 2. Dataset Info")
            info = result["basic_info"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", info["shape"][0])
                st.metric("Columns", info["shape"][1])
            with col2:
                st.write("**Column Names:**")
                st.write(", ".join(info["columns"]))
            st.write("**Summary Statistics:**")
            st.dataframe(info["summary"].T, use_container_width=True)

        # 3. Histograms
        if result.get("eda_hist") is not None:
            st.subheader("📊 3. Histograms")
            st.pyplot(result["eda_hist"], use_container_width=True)

        # 4. Model Performance
        score = result.get("model_score")
        if score is not None and score != -float("inf"):
            st.subheader("🤖 5. Model Performance")
            st.metric(label="Best Model Score", value=f"{score:.4f}")

        # 6. Insights
        if result.get("insights"):
            st.subheader("💡 6. Actionable Insights")
            for ins in result["insights"]:
                st.write(f"- {ins}")
