from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


def detect_problem_type(y):
    """
    Detect whether the problem is classification or regression.
    If unique values in y < 10 → classification
    Else → regression
    """
    unique_values = y.nunique()
    if y.dtype == "object" or y.nunique() < 10:
        return "classification"
    else:
        return "regression"


def train_and_select_model(X, y):
    """
    Train multiple models based on problem type and return the best one.
    Returns: (best_model, best_score)
    """
    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "classification":
        models = [
            LogisticRegression(max_iter=1000, solver='lbfgs'),
            RandomForestClassifier(n_estimators=100, random_state=42),
        ]
    else:
        models = [
            LinearRegression(),
            RandomForestRegressor(n_estimators=100, random_state=42),
        ]

    best_model = None
    best_score = -float("inf")

    for model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    Returns: model score (float)
    """
    return model.score(X_test, y_test)
