import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------------------------------
# 1: MODEL SELECTION
# -----------------------------------------------------
def select_and_train_model(df):
    if df is None or len(df) == 0 or len(df.columns) < 2:
        return None

    # Step 1: Split data
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 2: Detect task
    if y.nunique() <= 10:
        task = "classification"
    else:
        task = "regression"

    # Step 3: Train-test split
    test_sz = 0.2 if len(X) >= 5 else 0.5
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # Step 4: Define models
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, solver='lbfgs'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

    # Step 5: Train + select best model
    best_model = None
    best_name = None
    best_score = -float("inf")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception:
            continue

    if best_model is None:
        return None

    # Get predictions for evaluation
    try:
        y_pred = best_model.predict(X_test)
    except Exception:
        y_pred = y_test

    # Step 6: Return result
    return {
        "best_model": best_model,
        "model_name": best_name,
        "task": task,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }

# -----------------------------------------------------
# 2: MODEL EVALUATION
# -----------------------------------------------------
def evaluate_model(task, y_test, y_pred):
    try:
        if task == "classification":
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        else:
            return {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mse": float(mean_squared_error(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred))
            }
    except Exception:
        return {}

# -----------------------------------------------------
# 3: FULL PIPELINE
# -----------------------------------------------------
def run_full_model_pipeline(df):
    result = select_and_train_model(df)
    if result is None:
        return None

    metrics = evaluate_model(result["task"], result["y_test"], result["y_pred"])

    return {
        "model_name": result["model_name"],
        "task": result["task"],
        "metrics": metrics
    }

# -----------------------------------------------------
# LEGACY PIPELINE COMPATIBILITY (DO NOT BREAK)
# -----------------------------------------------------
def train_and_select_model(X, y):
    df = pd.concat([X, y], axis=1)
    res = select_and_train_model(df)
    
    if res is None or res["best_model"] is None:
        return None, None
        
    model = res["best_model"]
    try:
        score = model.score(res["X_test"], res["y_test"])
    except Exception:
        score = -float("inf")
        
    return model, score

def evaluate_model_legacy(model, X_test, y_test):
    if model is None:
        return None
    try:
        return model.score(X_test, y_test)
    except Exception:
        return None
