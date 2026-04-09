from backend.preprocess import preprocess_data
from backend.eda import plot_histograms, plot_correlation, show_basic_info, plot_barplot, plot_boxplot, plot_scatter, plot_pairplot
from backend.model import train_and_select_model, run_full_model_pipeline
from backend.insights import generate_insights
from agent.interpreter import interpret_query
from agent.planner import decide_plan

def run_agentic_pipeline(df, query=""):
    # 1. basic info on ORIGINAL data (readable column names)
    basic_info = show_basic_info(df)

    # 2. preprocess data
    import pandas as pd
    prep_result = preprocess_data(df)
    
    prep_summary = {}
    if isinstance(prep_result, tuple) and len(prep_result) == 5:
        X_train, X_test, y_train, y_test, prep_summary = prep_result
        # Recombine train and test into a single unified dataframe for downstream UI & EDA
        X_clean = pd.concat([X_train, X_test]).sort_index()
        y_clean = pd.concat([y_train, y_test]).sort_index()
        clean_data = pd.concat([X_clean, y_clean], axis=1)
    else:
        clean_data = prep_result
    # 3. interpret query
    intent = interpret_query(query)

    # 4. decide plan
    workflow = decide_plan(clean_data, intent)

    fig_hist = None
    fig_corr = None
    fig_bar = None
    fig_box = None
    fig_scatter = None
    fig_pair = None
    score = None
    insights = []
    
    metrics = {}
    model_name = ""
    task_type = ""

    if workflow == "eda_only":
        fig_hist = plot_histograms(clean_data)
        fig_corr = plot_correlation(clean_data)
        fig_bar = plot_barplot(clean_data)
        fig_box = plot_boxplot(clean_data)
        fig_scatter = plot_scatter(clean_data)
        fig_pair = plot_pairplot(clean_data)

    elif workflow == "full_pipeline":
        target = clean_data.columns[-1]
        X = clean_data.drop(columns=[target])
        y = clean_data[target]

        model, best_score = train_and_select_model(X, y)

        if best_score is not None and best_score != -float("inf"):
            score = best_score

        if model is not None:
            insights = generate_insights(clean_data, model)
            
        full_result = run_full_model_pipeline(clean_data)
        if full_result:
            metrics = full_result.get("metrics", {})
            model_name = full_result.get("model_name", "")
            task_type = full_result.get("task", "")

        fig_hist = plot_histograms(clean_data)
        fig_corr = plot_correlation(clean_data)
        fig_bar = plot_barplot(clean_data)
        fig_box = plot_boxplot(clean_data)
        fig_scatter = plot_scatter(clean_data)
        fig_pair = plot_pairplot(clean_data)

    return {
        "clean_data": clean_data,
        "preprocessing_summary": prep_summary,
        "basic_info": basic_info,
        "eda_hist": fig_hist,
        "eda_corr": fig_corr,
        "eda_bar": fig_bar,
        "eda_box": fig_box,
        "eda_scatter": fig_scatter,
        "eda_pair": fig_pair,
        "model_score": score,
        "insights": insights,
        "metrics": metrics,
        "model_name": model_name,
        "task": task_type
    }
