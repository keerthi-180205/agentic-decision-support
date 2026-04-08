import streamlit as st
import pandas as pd

# Import logic from backend package
from backend.preprocess import preprocess_data
from backend.eda import plot_histograms
from backend.model import train_and_select_model
from backend.insights import generate_insights
from utils.file_handler import load_file

st.set_page_config(page_title="Automated Data Analysis Application", layout="wide", page_icon="🚀")

st.title("🚀 Data Analysis & Insights Pipeline")
st.markdown("Upload your dataset to automatically preprocess it, perform EDA, build a model, and generate insights.")

# 1. File uploader (CSV, Excel, JSON)
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    # Read the dataset based on file extension
    file_name = uploaded_file.name
    
    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format!")
            st.stop()
            
        # Clean up empty columns (like Unnamed: 2) from Excel
        df = df.dropna(axis=1, how='all')
        
        # Safely fix PyArrow mixed-type crashes by setting non-NaN objects to strings
        for col in df.select_dtypes(include='object').columns:
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].astype(str)
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
        
    # 2. Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # 3. Button: "Run Analysis"
    if st.button("Run Analysis", use_container_width=True, type="primary"):
        st.markdown("---")
        
        # Step A: Preprocess Data
        st.subheader("🧹 1. Cleaned Data")
        with st.spinner("Preprocessing data (handling missing values, encoding, etc.)..."):
            df_clean = preprocess_data(df)
            st.dataframe(df_clean.head())
        
        # Step B: Exploratory Data Analysis (EDA)
        st.subheader("📊 2. Exploratory Data Analysis")
        with st.spinner("Generating plots and visualizing distributions..."):
            fig = plot_histograms(df_clean)
            if fig:
                st.pyplot(fig)
            else:
                st.info("No plot could be generated.")
                
        # Step C: Data Splitting (Last column assumes target)
        target = df_clean.columns[-1]
        X = df_clean.drop(columns=[target])
        y = df_clean[target]
        
        # Step D: Train and Select Model
        st.subheader("🤖 3. Model Performance")
        with st.spinner(f"Training models to predict '{target}'..."):
            model, score = train_and_select_model(X, y)
            st.success("Model trained successfully!")
            st.metric(label=f"Best Model Score (Predicting: {target})", value=f"{score:.4f}")
            
        # Step E: Generate Business Insights
        st.subheader("💡 4. Actionable Insights")
        with st.spinner("Extracting insights based on the trained model and data..."):
            insights = generate_insights(df_clean, model)
            if insights and isinstance(insights, list):
                for ins in insights:
                    st.write(f"- {ins}")
            else:
                st.info("No specific insights could be generated.")
