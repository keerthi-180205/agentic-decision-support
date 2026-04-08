import streamlit as st
import pandas as pd

# Import logic from backend package
from backend.preprocess import preprocess_data
from backend.eda import plot_histograms
from backend.model import train_and_select_model
from backend.insights import generate_insights

st.set_page_config(page_title="Automated Data Analysis Application", layout="wide", page_icon="🚀")

st.title("🚀 Data Analysis & Insights Pipeline")
st.markdown("Upload your dataset and ask questions to automatically process, analyze, and model your data.")

# 1. Add file upload:
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    # 2. Load file:
    df = load_file(uploaded_file)
    
    # Safely display preview without PyArrow ArrowTypeErrors by stringifying ONLY for the UI
    st.subheader("Dataset Preview")
    
    # Create a safe copy for the preview
    preview_df = df.head().copy()
    for col in preview_df.select_dtypes(include=['object']).columns:
        preview_df[col] = preview_df[col].astype(str)
        
    st.dataframe(preview_df)
    
    # 3. Add chatbot input:
    query = st.text_input("Ask your data question")
    
    if st.button("Run Analysis", use_container_width=True, type="primary"):
        st.markdown("---")
        
        # 6. Add agent message:
        st.info("Agent is analyzing your request...")
        
        # 4. Call agent:
        result = run_agent(df, query)
        
        # 5. Display output:
        
        # Cleaned data
        if result.get("clean_data") is not None:
            st.subheader("🧹 1. Cleaned Data")
            clean_preview = result["clean_data"].head().copy()
            for col in clean_preview.select_dtypes(include=['object']).columns:
                clean_preview[col] = clean_preview[col].astype(str)
            st.dataframe(clean_preview)
            
        # Graphs (if available)
        if result.get("eda") is not None:
            st.subheader("📊 2. Exploratory Data Analysis")
            st.pyplot(result["eda"])
            
        # Model score (if available)
        if result.get("model_score") is not None:
            st.subheader("🤖 3. Model Performance")
            st.metric(label="Best Model Score", value=f"{result['model_score']:.4f}")
            
        # Insights (list)
        if result.get("insights"):
            st.subheader("💡 4. Actionable Insights")
            for ins in result["insights"]:
                st.write(f"- {ins}")
