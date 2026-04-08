import numpy as np


def generate_insights(df, model):
    """
    Generate human-readable insights from the trained model.
    Assumes the last column of df is the target variable.
    Returns: list of insight strings
    """
    insights = []

    # Last column is target, all others are features
    feature_names = list(df.columns[:-1])
    target_name = df.columns[-1]

    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1]

    # --- Feature Importance (for tree-based models like RandomForest) ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-3:]  # top 3 features by importance

        for i in reversed(top_indices):  # highest importance first
            insights.append(
                f"Feature '{feature_names[i]}' strongly influences the target '{target_name}' "
                f"(importance: {importances[i]:.2f})"
            )

        low_indices = [i for i in range(len(feature_names)) if i not in top_indices]
        if low_indices:
            low_features = ", ".join([feature_names[i] for i in low_indices])
            insights.append(f"Features with low impact on '{target_name}': {low_features}")

    # --- Correlation with target (for all models) ---
    else:
        # Fix 3: skip constant columns to avoid NaN from corr()
        correlations = X.apply(lambda col: col.corr(y) if col.std() != 0 else 0)
        corr_threshold = 0.3

        # Fix 4: show only top 3 by absolute correlation
        top_features = correlations.abs().nlargest(3).index
        for feature in top_features:
            corr = correlations[feature]
            if abs(corr) >= corr_threshold:
                insights.append(
                    f"Feature '{feature}' strongly influences the target '{target_name}' "
                    f"(correlation: {corr:.2f})"
                )
            else:
                insights.append(
                    f"Feature '{feature}' has low impact on the target '{target_name}' "
                    f"(correlation: {corr:.2f})"
                )

    # --- Decision Suggestions ---
    insights.append(
        "Decision Suggestion: Focus on the top influential features to improve model performance and decision quality."
    )
    insights.append(
        "Decision Suggestion: Consider removing low-impact features to simplify the model and reduce noise."
    )

    return insights
