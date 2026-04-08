import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df):
    df = df.copy()

    df = df.dropna(axis=1, how='all')

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x)

    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors='coerce')
        # Only apply if majority of values converted successfully (>50%)
        if converted.notna().sum() > len(df) * 0.5:
            df[col] = converted

    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        modes = df[col].mode()
        if len(modes) > 0:
            df[col] = df[col].fillna(modes[0])
        else:
            df[col] = df[col].fillna("Unknown")

    return df


def encode_categorical(df):
    df = df.copy()

    # Protect target column (last column) from encoding
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    
    # Filter out high-cardinality columns (like ID, Names) to prevent MemoryError
    cat_cols = [col for col in cat_cols if X[col].nunique() <= 20]
    
    # Drop the high-cardinality columns that we aren't encoding (since they will break models)
    high_card_cols = [c for c in X.select_dtypes(include='object').columns.tolist() if c not in cat_cols]
    X = X.drop(columns=high_card_cols)

    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Convert any bool columns from get_dummies to int
    bool_cols = X.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    df = pd.concat([X, y], axis=1)
    return df


def scale_features(df):
    df = df.copy()

    if len(df.columns) < 2:
        return df

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).copy()
    y = df[[target_col]]

    num_cols = X.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    df = pd.concat([X, y], axis=1)
    return df


def preprocess_data(df):
    df = df.copy()
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = scale_features(df)
    return df
