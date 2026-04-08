import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------
# 1. BASIC INFO
# -------------------------------
def show_basic_info(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "summary": df.describe(include='all')
    }

# -------------------------------
# 2. HISTOGRAMS
# -------------------------------
def plot_histograms(df):
    df = df.copy()
    
    # Convert boolean -> numeric (True -> 1, False -> 0)
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    
    # If NO numeric columns, return None
    if len(numeric_cols) == 0:
        return None
        
    # Prevent DecompressionBombError if too many columns were one-hot encoded
    if len(numeric_cols) > 20: 
        numeric_cols = numeric_cols[:20] # Limit to max 20 histograms
        
    # Create subplots (Max 3 per row)
    n_cols = len(numeric_cols)
    n_grid_cols = min(n_cols, 3)
    n_rows = int(np.ceil(n_cols / n_grid_cols))
    
    fig, axes = plt.subplots(n_rows, n_grid_cols, figsize=(n_grid_cols * 5, n_rows * 4))
    
    # Flatten axes array for simple iteration (or wrap in list if only 1)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # Plot histogram for each numeric column
    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[idx])
        axes[idx].set_title(f"Histogram of {col}")
        
    # Remove unused subplots
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])
        
    plt.tight_layout()
    return fig

# -------------------------------
# 3. CORRELATION HEATMAP
# -------------------------------
def plot_correlation(df):
    df = df.copy()
    
    # Convert boolean -> numeric (True -> 1, False -> 0)
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        
    # Select numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    # Prevent DecompressionBombError if too many columns
    if numeric_df.shape[1] > 30:
        numeric_df = numeric_df.iloc[:, :30]
        
    # If less than 2 numeric columns, return None
    if numeric_df.shape[1] < 2:
        return None
        
    # Compute correlation
    corr = numeric_df.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    
    return fig