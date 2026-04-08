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
# HELPER: CLEAN NUMERIC DATA
# -------------------------------
def get_clean_numeric_df(df):
    df = df.copy()

    # Convert boolean → int
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    numeric_df = df.select_dtypes(include='number')

    # Remove zero variance
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

    # Remove ID-like columns
    numeric_df = numeric_df.loc[:, ~numeric_df.columns.str.contains("id|ID|Id", case=False)]

    # 🚨 RELAX THIS: keep columns with some variation
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 2]

    # 🚨 FALLBACK: if nothing left, use original numeric columns
    if numeric_df.shape[1] == 0:
        numeric_df = df.select_dtypes(include='number')

    # Limit columns (for UI)
    if numeric_df.shape[1] > 6:
        numeric_df = numeric_df.iloc[:, :6]

    return numeric_df

# -------------------------------
# 2. HISTOGRAMS
# -------------------------------
def plot_histograms(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] == 0:
        return None

    cols = numeric_df.columns

    n_cols = len(cols)
    n_grid_cols = min(n_cols, 3)
    n_rows = int(np.ceil(n_cols / n_grid_cols))

    fig, axes = plt.subplots(n_rows, n_grid_cols, figsize=(n_grid_cols * 5, n_rows * 4))

    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, col in enumerate(cols):
        sns.histplot(numeric_df[col].dropna(), kde=True, ax=axes[idx])

        # Clean title
        clean_name = str(col)[:25]
        axes[idx].set_title(clean_name)

    # Remove empty plots
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig

# -------------------------------
# 3. CORRELATION HEATMAP
# -------------------------------
def plot_correlation(df):
    numeric_df = get_clean_numeric_df(df)

    # Need at least 2 features
    if numeric_df.shape[1] < 2:
        return None

    # Skip meaningless correlation (too unique data)
    if numeric_df.nunique().mean() > 0.8 * len(df):
        return None

    corr = numeric_df.corr()

    # Clean NaN rows/cols
    corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if corr.shape[0] < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()

    return fig

# -------------------------------
# 4. SCATTER PLOT
# -------------------------------
def plot_scatter(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] < 2:
        return None

    x_col, y_col = numeric_df.columns[0], numeric_df.columns[1]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=numeric_df, x=x_col, y=y_col, ax=ax, alpha=0.6)
    ax.set_title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    return fig

# -------------------------------
# 5. PAIRPLOT
# -------------------------------
def plot_pairplot(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] < 2:
        return None

    cols = numeric_df.columns[:4].tolist()
    # Reduce height of each subplot to make it more compact
    pair_fig = sns.pairplot(numeric_df[cols].dropna(), height=1.5, aspect=1.2)
    plt.tight_layout()
    return pair_fig.fig

# -------------------------------
# 6. BOXPLOT
# -------------------------------
def plot_boxplot(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] == 0:
        return None

    width = min(8, max(5, numeric_df.shape[1] * 1.0))
    fig, ax = plt.subplots(figsize=(width, 4))
    sns.boxplot(data=numeric_df, ax=ax)
    ax.set_title("Boxplot — Distribution & Outliers")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# -------------------------------
# 7. BAR PLOT (CATEGORICAL)
# -------------------------------
def plot_barplot(df):
    # Try to find categorical columns first, then fallback to low cardinality numerics
    cat_df = df.select_dtypes(include=['object', 'category'])
    if cat_df.shape[1] == 0:
        cat_df = df.loc[:, df.nunique() <= 10]
        
    if cat_df.shape[1] == 0:
        return None
        
    cols = cat_df.columns[:4].tolist() # Limit to first 4 categorical features
    if len(cols) == 0:
        return None

    n_cols = len(cols)
    n_grid_cols = min(n_cols, 2)
    n_rows = int(np.ceil(n_cols / n_grid_cols))

    fig, axes = plt.subplots(n_rows, n_grid_cols, figsize=(n_grid_cols * 5, n_rows * 4))

    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, col in enumerate(cols):
        val_counts = df[col].value_counts().nlargest(10) # Top 10 categories
        
        # Use a single color for seaborn 0.13+ or standard hue config
        sns.barplot(x=val_counts.index, y=val_counts.values, ax=axes[idx], color='steelblue')
        
        clean_name = str(col)[:25]
        axes[idx].set_title(f"{clean_name} (Top 10)")
        axes[idx].tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig