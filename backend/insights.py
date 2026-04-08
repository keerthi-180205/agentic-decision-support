import numpy as np
import pandas as pd


def generate_insights(df, model):
    if model is None or len(df.columns) < 2:
        return ["Not enough data to extract insights."]

    insights = []

    target_name = df.columns[-1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = list(X.columns)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Match feature count safely
        if len(importances) == len(feature_names):
            top_indices = np.argsort(importances)[-3:]
            for i in reversed(top_indices):
                score = importances[i]
                if score > 0:
                    insights.append(
                        f"Feature '{feature_names[i]}' appears to be one of the strongest drivers influencing the outcome (importance: {score:.2f})."
                    )
    else:
        X_numeric = X.select_dtypes(include='number')

        if pd.api.types.is_bool_dtype(y) or y.dtype == 'object':
            try:
                y_numeric = y.astype('category').cat.codes
            except Exception:
                y_numeric = y
        else:
            y_numeric = y

        if not X_numeric.empty:
            correlations = X_numeric.apply(
                lambda col: col.corr(y_numeric) if col.std() != 0 else 0
            )
            top_features = correlations.abs().nlargest(3).index
            for feature in top_features:
                c = correlations[feature]
                if pd.notna(c) and abs(c) >= 0.3:
                    direction = "higher" if c > 0 else "lower"
                    insights.append(
                        f"An increase in '{feature}' is associated with a {direction} value of the target variable (correlation: {c:.2f})."
                    )

    if not insights:
        insights.append(f"No strongly influential features found for predicting '{target_name}'.")

    insights.append("Suggestion: Focus on the top influential features identified above.")
    insights.append("Suggestion: Consider removing low-impact features to improve model efficiency.")

    return insights
