import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

def preprocess_data(df):
    df = df.copy()

    summary = {
        "numerical_columns": [],
        "categorical_columns": [],
        "scaled_columns": [],
        "onehot_encoded": [],
        "ordinal_encoded": [],
        "missing_values_handled": True,
        "dropped_columns": [],
        "explanation": []
    }

    # ── STEP 0: FIX DIRTY VALUES (🔥 IMPORTANT) ──
    df.replace(["ERROR", "UNKNOWN", "nan"], np.nan, inplace=True)

    # Try converting numeric-like columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # ── STEP 1: CLEANING ──
    df = df.drop_duplicates()

    drop_cols = []
    for col in df.columns:
        col_lower = str(col).lower()

        if "unnamed" in col_lower:
            drop_cols.append(col)

        elif df[col].nunique() == len(df) and any(x in col_lower for x in ["id", "uuid", "code"]):
            drop_cols.append(col)

    df = df.drop(columns=drop_cols)
    summary["dropped_columns"] = drop_cols

    # ── STEP 2: COLUMN TYPES ──
    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # ── STEP 3: MISSING VALUES ──
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

    # ── STEP 4: TARGET DETECTION ──
    target_names = ['target', 'label', 'churn']
    target_col = df.columns[-1]

    for col in df.columns:
        if str(col).lower() in target_names:
            target_col = col
            break

    if df[target_col].nunique() > 20:
        for col in df.columns:
            if df[col].nunique() <= 2:
                target_col = col
                break

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if target_col in num_cols: num_cols.remove(target_col)
    if target_col in cat_cols: cat_cols.remove(target_col)

    summary["numerical_columns"] = num_cols
    summary["categorical_columns"] = cat_cols

    # ── STEP 5: SPLIT ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── STEP 6: ENCODING ──
    ordinal_cols = []
    nominal_cols = []

    for col in cat_cols:
        unique_count = X_train[col].nunique()

        if unique_count <= 5:
            ordinal_cols.append(col)
        elif unique_count <= 20:
            nominal_cols.append(col)
        else:
            summary["dropped_columns"].append(col)

    summary["ordinal_encoded"] = ordinal_cols
    summary["onehot_encoded"] = nominal_cols

    X_train_transformed = pd.DataFrame(index=X_train.index)
    X_test_transformed = pd.DataFrame(index=X_test.index)

    # numerical
    for col in num_cols:
        X_train_transformed[col] = X_train[col]
        X_test_transformed[col] = X_test[col]

    # ordinal encoding
    if ordinal_cols:
        ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_transformed[ordinal_cols] = ord_enc.fit_transform(X_train[ordinal_cols].astype(str))
        X_test_transformed[ordinal_cols] = ord_enc.transform(X_test[ordinal_cols].astype(str))

    # one-hot encoding
    if nominal_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        train_ohe = ohe.fit_transform(X_train[nominal_cols].astype(str))
        test_ohe = ohe.transform(X_test[nominal_cols].astype(str))

        ohe_cols = ohe.get_feature_names_out(nominal_cols)

        X_train_transformed = pd.concat([
            X_train_transformed,
            pd.DataFrame(train_ohe, columns=ohe_cols, index=X_train.index)
        ], axis=1)

        X_test_transformed = pd.concat([
            X_test_transformed,
            pd.DataFrame(test_ohe, columns=ohe_cols, index=X_test.index)
        ], axis=1)

    # ── STEP 7: SCALING ──
    if num_cols:
        scaler = StandardScaler()
        X_train_transformed[num_cols] = scaler.fit_transform(X_train_transformed[num_cols])
        X_test_transformed[num_cols] = scaler.transform(X_test_transformed[num_cols])

    summary["scaled_columns"] = num_cols

    # ── STEP 8: EXPLANATION ──
    if num_cols:
        summary["explanation"].append(
            f"StandardScaler applied to numerical columns: {num_cols}."
        )

    if ordinal_cols:
        summary["explanation"].append(
            f"Ordinal encoding applied to: {ordinal_cols} (low unique values)."
        )

    if nominal_cols:
        summary["explanation"].append(
            f"One-hot encoding applied to: {nominal_cols}."
        )

    if drop_cols:
        summary["explanation"].append(
            f"Dropped columns: {drop_cols} (ID-like or irrelevant)."
        )

    return X_train_transformed, X_test_transformed, y_train, y_test, summary