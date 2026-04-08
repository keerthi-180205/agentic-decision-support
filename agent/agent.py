from backend.preprocess import preprocess_data
from backend.eda import plot_histograms
from backend.model import train_and_select_model
from backend.insights import generate_insights
from agent.interpreter import interpret_query
from agent.planner import decide_plan

def run_agentic_pipeline(df, query=""):
    """
    Main agent logic combining preprocessing, interpretation, and execution.
    """
    # 1. preprocess data
    clean_data = preprocess_data(df)
    
    # 2. interpret query
    intent = interpret_query(query)
    
    # 3. decide plan
    workflow = decide_plan(clean_data, intent)
    
    # Initialize returns
    fig = None
    score = None
    insights = []
    
    # Execute workflow
    if workflow == "eda_only":
        fig = plot_histograms(clean_data)
        
    elif workflow == "full_pipeline":
        # Split target (last column)
        target = clean_data.columns[-1]
        X = clean_data.drop(columns=[target])
        y = clean_data[target]
        
        # Run model
        model, score = train_and_select_model(X, y)
        
        # Generate insights
        if model is not None:
            insights = generate_insights(clean_data, model)
            
        # Run EDA
        fig = plot_histograms(clean_data)
        
    return {
        "clean_data": clean_data,
        "eda": fig,
        "model_score": score,
        "insights": insights
    }
