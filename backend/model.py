from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def detect_problem_type(y):
    if y.dtype == "object" or y.nunique() < 10:
        return "classification"
    return "regression"

def train_and_select_model(X, y):
    if len(X) == 0 or len(y) == 0:
        return None, None

    problem_type = detect_problem_type(y)

    test_size = 0.2 if len(X) >= 5 else 0.5
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

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
        try:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    return best_model, best_score


def evaluate_model(model, X_test, y_test):
    if model is None:
        return None
    try:
        return model.score(X_test, y_test)
    except Exception:
        return None
