import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df):
    """
    Fill missing values:
    - Numerical columns → fill with mean
    - Categorical columns → fill with mode
    """
    df = df.copy()

    # Fill numerical columns with mean
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if not df[col].empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical(df):
    """
    One-hot encode all categorical (object) columns.
    Uses drop_first=True to avoid multicollinearity.
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def scale_features(df):
    """
    Scale only numerical feature columns using StandardScaler.
    The last column is treated as the target and is excluded from scaling.
    Handles the case where there are no numerical feature columns safely.
    """
    df = df.copy()

    # Separate target (last column) from features
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # Identify numeric feature columns only (not the target)
    num_cols = X.select_dtypes(include='number').columns.tolist()

    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Combine features and target back
    df = pd.concat([X, y], axis=1)

    return df


def preprocess_data(df):
    """
    Master preprocessing pipeline.
    Calls all steps in order:
      1. Handle missing values
      2. Encode categorical columns
      3. Scale numerical feature columns (excludes target)

    Returns the fully cleaned and preprocessed DataFrame.
    """
    df = df.copy()

    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = scale_features(df)

    return df
