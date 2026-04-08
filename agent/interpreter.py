def interpret_query(query: str) -> str:
    """
    Interpret the natural language query into an intent.
    """
    if not query or not isinstance(query, str):
        return "auto"
        
    q = query.lower().strip()
    
    if "predict" in q:
        return "model"
    elif "visualize" in q or "graph" in q:
        return "eda"
        
    return "auto"
