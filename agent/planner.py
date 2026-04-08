def decide_plan(df, intent: str = "auto") -> str:
    if intent == "eda":
        return "eda_only"
    if intent == "model":
        return "full_pipeline"

    # Auto decision: less than 5 rows -> eda_only, else full_pipeline
    if len(df) < 5:
        return "eda_only"

    return "full_pipeline"
