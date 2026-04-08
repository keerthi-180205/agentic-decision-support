from backend.preprocess import preprocess_data
from backend.eda import plot_histograms, plot_correlation, show_basic_info
from backend.model import train_and_select_model
from backend.insights import generate_insights
from agent.interpreter import interpret_query
from agent.planner import decide_plan

def run_agentic_pipeline(df, query=""):
    # 1. basic info on ORIGINAL data (readable column names)
    basic_info = show_basic_info(df)

    # 2. preprocess data
    clean_data = preprocess_data(df)

    # 3. interpret query
    intent = interpret_query(query)

    # 4. decide plan
    workflow = decide_plan(clean_data, intent)

    fig_hist = None
    fig_corr = None
    score = None
    insights = []

    if workflow == "eda_only":
        fig_hist = plot_histograms(clean_data)
        fig_corr = plot_correlation(clean_data)

    elif workflow == "full_pipeline":
        target = clean_data.columns[-1]
        X = clean_data.drop(columns=[target])
        y = clean_data[target]

        model, best_score = train_and_select_model(X, y)

        if best_score is not None and best_score != -float("inf"):
            score = best_score

        if model is not None:
            insights = generate_insights(clean_data, model)

        fig_hist = plot_histograms(clean_data)
        fig_corr = plot_correlation(clean_data)

    return {
        "clean_data": clean_data,
        "basic_info": basic_info,
        "eda_hist": fig_hist,
        "eda_corr": fig_corr,
        "model_score": score,
        "insights": insights
    }
