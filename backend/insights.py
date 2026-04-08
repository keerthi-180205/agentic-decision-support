import numpy as np
import pandas as pd

def generate_insights(df, model):
    """
    Generate simple insights and suggestions based on feature importance or correlation.
    """
    if model is None or len(df.columns) < 2:
        return ["Not enough data to extract insights."]

    insights = []
    
    # Last column is target
    feature_names = list(df.columns[:-1])
    target_name = df.columns[-1]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Use feature importance if available (e.g. tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            # Top 3 features
            top_indices = np.argsort(importances)[-3:]
            for i in reversed(top_indices):
                score = importances[i]
                if score > 0:
                    insights.append(
                        f"Feature '{feature_names[i]}' strongly influences '{target_name}' (importance: {score:.2f})"
                    )
    else:
        # Fallback to correlation for linear models
        X_numeric = X.select_dtypes(include=[np.number, bool])
        
        # Clean target for correlation
        if pd.api.types.is_bool_dtype(y) or y.dtype == 'object':
            try:
                y_numeric = y.astype('category').cat.codes
            except:
                y_numeric = y
        else:
            y_numeric = y
            
        if not X_numeric.empty:
            correlations = X_numeric.apply(lambda col: col.corr(y_numeric) if col.std() != 0 else 0)
            corr_threshold = 0.3
            
            # Top 3 correlated features
            top_features = correlations.abs().nlargest(3).index
            for feature in top_features:
                c = correlations[feature]
                if pd.notna(c) and abs(c) >= corr_threshold:
                    val_type = "positively" if c > 0 else "negatively"
                    insights.append(
                        f"Feature '{feature}' {val_type} influences '{target_name}' (correlation: {c:.2f})"
                    )

    if not insights:
        insights.append(f"No strongly influential features found for predicting '{target_name}'.")

    # simple insights + suggestions
    insights.append("Suggestion: Focus on the top influential features identified above.")
    insights.append("Suggestion: Consider removing low-impact features to improve model efficiency.")

    return insights
