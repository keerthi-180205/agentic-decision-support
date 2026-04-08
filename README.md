# 🤖 Agentic Decision Support System

A hackathon project that uses machine learning to automatically analyze datasets, train the best-fit model, and generate human-readable insights to support smarter decision-making.

---

## 📌 Project Overview

This system takes any structured CSV dataset as input and:
- Automatically detects whether the problem is **classification** or **regression**
- Trains and compares multiple ML models to find the **best performer**
- Generates **actionable insights** based on feature importance and correlations
- Displays results through an **interactive web UI**

---

## 👥 Team Structure (4 Members)

| File | Role | Responsibility |
|---|---|---|
| `preprocess.py` | Data Engineer | Data cleaning, encoding, train/test split |
| `eda.py` | Data Analyst | Exploratory analysis & visualizations |
| `model.py` + `insights.py` | ML Engineer | Model training, selection & insight generation |
| `app.py` | Frontend/UI | Streamlit interface that ties everything together |

---

## 📁 Project Structure

```
agentic-decision-support/
├── app.py              # Streamlit web application (UI entry point)
├── model.py            # ML model training & selection
├── insights.py         # Feature importance & insight generation
├── preprocess.py       # Data preprocessing & cleaning
├── eda.py              # Exploratory data analysis
├── utils.py            # Shared helper functions
├── data/
│   └── sample.csv      # Sample dataset
└── requirements.txt    # Project dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/agentic-decision-support.git
cd agentic-decision-support
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧠 ML Module — `model.py`

### `detect_problem_type(y)`
Automatically determines if the task is classification or regression.
- `< 10 unique values` or `object dtype` → **Classification**
- `≥ 10 unique values` → **Regression**

### `train_and_select_model(X, y)`
Trains multiple models and returns the best one.
- **Classification:** `LogisticRegression`, `RandomForestClassifier`
- **Regression:** `LinearRegression`, `RandomForestRegressor`
- Returns: `(best_model, best_score)`

### `evaluate_model(model, X_test, y_test)`
Scores the model on test data.
- Returns: `float` score

---

## 💡 Insights Module — `insights.py`

### `generate_insights(df, model)`
Generates human-readable insights from the trained model.
- Uses **feature importance** for tree-based models (RandomForest)
- Falls back to **correlation** for linear models
- Shows **top 3 most influential features** only (UI-friendly)
- Appends 2 decision suggestions
- Returns: `list` of insight strings

---

## 🔗 How Modules Connect

```
CSV Data
   ↓
preprocess.py  →  cleaned X, y
   ↓
model.py       →  best_model, score
   ↓
insights.py    →  list of insights
   ↓
app.py         →  displays everything in UI
```

---

## 📦 Dependencies

| Package | Version | Used In |
|---|---|---|
| pandas | 2.3.3 | preprocess, eda, insights |
| numpy | 2.4.2 | insights |
| scikit-learn | 1.8.0 | model |
| matplotlib | 3.10.8 | eda |
| seaborn | 0.13.2 | eda |
| streamlit | 1.54.0 | app |

---

## 📄 License

This project was built as part of a hackathon. Open for educational use.