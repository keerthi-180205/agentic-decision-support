import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    # RELAX THIS: keep columns with some variation
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 2]

    # FALLBACK: if nothing left, use original numeric columns
    if numeric_df.shape[1] == 0:
        numeric_df = df.select_dtypes(include='number')

    # Limit columns (for UI speed)
    if numeric_df.shape[1] > 6:
        numeric_df = numeric_df.iloc[:, :6]

    return numeric_df

# -------------------------------
# UI THEME CONFIGURATION
# -------------------------------
def configure_layout(fig):
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font_color="#E5E7EB",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="closest"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#374151", zerolinecolor="#374151")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#374151", zerolinecolor="#374151")
    return fig

# -------------------------------
# 2. HISTOGRAMS
# -------------------------------
def plot_histograms(df):
    numeric_df = get_clean_numeric_df(df)
    if numeric_df.shape[1] == 0:
        return None

    cols = numeric_df.columns[:6].tolist()
    n_cols = len(cols)
    n_grid_cols = min(n_cols, 2)
    n_rows = int(np.ceil(n_cols / n_grid_cols))

    fig = make_subplots(rows=n_rows, cols=n_grid_cols, subplot_titles=cols)

    for idx, col in enumerate(cols):
        r = (idx // n_grid_cols) + 1
        c = (idx % n_grid_cols) + 1
        fig.add_trace(go.Histogram(x=numeric_df[col].dropna(), name=col, marker_color="#6366F1", opacity=0.8, marker_line_color="white", marker_line_width=1), row=r, col=c)

    fig.update_layout(title="Distributions & Frequencies", showlegend=False, height=max(400, 250 * n_rows))
    return configure_layout(fig)

# -------------------------------
# 3. CORRELATION HEATMAP
# -------------------------------
def plot_correlation(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] < 2:
        return None

    if numeric_df.nunique().mean() > 0.8 * len(df):
        return None

    corr = numeric_df.corr().dropna(axis=0, how='all').dropna(axis=1, how='all')

    if corr.shape[0] < 2:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu', zmin=-1, zmax=1
    ))
    fig.update_layout(title="Correlation Matrix", height=500)
    return configure_layout(fig)

# -------------------------------
# 4. SCATTER PLOT
# -------------------------------
def plot_scatter(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] < 2:
        return None

    x_col, y_col = numeric_df.columns[0], numeric_df.columns[1]

    fig = px.scatter(numeric_df, x=x_col, y=y_col, opacity=0.7, color_discrete_sequence=["#8B5CF6"])
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(title=f"Relationship Profile: {x_col} vs {y_col}", height=450)
    return configure_layout(fig)

# -------------------------------
# 5. PAIRPLOT
# -------------------------------
def plot_pairplot(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] < 2:
        return None

    cols = numeric_df.columns[:4].tolist()
    fig = px.scatter_matrix(numeric_df, dimensions=cols, color_discrete_sequence=["#8B5CF6"], opacity=0.6)
    fig.update_traces(diagonal_visible=True, marker_line_color="white", marker_line_width=1)
    fig.update_layout(title="Multivariate Interaction Matrix", height=600)
    return configure_layout(fig)

# -------------------------------
# 6. BOXPLOT
# -------------------------------
def plot_boxplot(df):
    numeric_df = get_clean_numeric_df(df)

    if numeric_df.shape[1] == 0:
        return None

    fig = go.Figure()
    for col in numeric_df.columns[:8]:
        fig.add_trace(go.Box(
            y=numeric_df[col].dropna(), 
            name=col, 
            fillcolor="#8B5CF6",
            line_color="white",
            line_width=1,
            marker_color="#8B5CF6", 
            marker=dict(line=dict(color="white", width=1))
        ))

    fig.update_layout(title="Outlier Detection & Ranges", showlegend=False, height=450)
    return configure_layout(fig)

# -------------------------------
# 7. BAR PLOT (CATEGORICAL)
# -------------------------------
def plot_barplot(df):
    cat_df = df.select_dtypes(include=['object', 'category'])
    if cat_df.shape[1] == 0:
        cat_df = df.loc[:, df.nunique() <= 10]
        
    if cat_df.shape[1] == 0:
        return None
        
    cols = cat_df.columns[:2].tolist()
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)

    for idx, col in enumerate(cols):
        val_counts = df[col].value_counts().nlargest(10)
        fig.add_trace(go.Bar(x=val_counts.index, y=val_counts.values, marker_color="#8B5CF6", name=col, marker_line_color="white", marker_line_width=1), row=1, col=idx+1)

    fig.update_layout(title="Highest Frequency Categories", showlegend=False, height=400)
    return configure_layout(fig)