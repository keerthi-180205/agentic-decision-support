import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    df = df.copy()

    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')

    # Safely flatten complex objects (lists, dicts) from JSON into strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x)

    # Convert boolean to numeric: True -> 1, False -> 0
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Convert numeric-like strings safely
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Fill numerical columns with mean
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Fill categorical columns with mode
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

    # Detect categorical columns (object dtype)
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if len(cat_cols) > 0:
        # Apply one-hot encoding
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def scale_features(df):
    df = df.copy()

    # Ensure there's a target column to separate
    if len(df.columns) < 2:
        return df

    # Treat last column as target
    target_col = df.columns[-1]
    
    # Separate X and y
    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    # Scale only numeric columns in X using StandardScaler
    num_cols = X.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Recombine X and y
    df = pd.concat([X, y], axis=1)

    return df


def preprocess_data(df):
    df = df.copy()

    # Call functions in order
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = scale_features(df)

    return df