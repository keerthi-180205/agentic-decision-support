def decide_plan(df, intent: str = "auto") -> str:
    """
    Decide the workflow based on the intent and dataset size.
    """
    if intent == "eda":
        return "eda_only"
    if intent == "model":
        return "full_pipeline"
        
    # Auto decision: small dataset -> eda_only, else full_pipeline
    if len(df) < 50:
        return "eda_only"
        
    return "full_pipeline"
