import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# -------------------------------
# BASIC INFO
# -------------------------------
def show_basic_info(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "summary": df.describe(include='all')
    }


# -------------------------------
# HISTOGRAMS
# -------------------------------
def plot_histograms(df):
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) == 0:
        return None

    n_cols = len(numeric_cols)
    n_rows = int(np.ceil(n_cols / 3))
    n_grid_cols = min(n_cols, 3)

    fig, axes = plt.subplots(n_rows, n_grid_cols, figsize=(n_grid_cols * 5, n_rows * 4))

    # Handle single plot case
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[idx])
        axes[idx].set_title(f"Histogram of {col}")

    # Remove unused subplots
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig


# -------------------------------
# CORRELATION HEATMAP
# -------------------------------
def plot_correlation(df):
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cleaner heatmap (less clutter)
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()

    return fig