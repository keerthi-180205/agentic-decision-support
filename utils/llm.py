import ollama

def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama LLM and retrieves the response.
    """
    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: Unable to connect to local Ollama. Please ensure Ollama is installed, running locally, and the 'llama3' model is pulled. Details: {str(e)}"

def get_data_science_response(df_shape, model_name, task, metrics, insights, user_question: str) -> str:
    """
    Submits a context-aware question to the local LLM adopting an expert data scientist persona.
    """
    system_prompt = """
You are a professional Data Scientist and Statistical Analyst.

STRICT BEHAVIOR RULES:
- ONLY answer questions related to data analysis, statistics, machine learning, or the given dataset
- DO NOT answer general knowledge or unrelated questions
- If a question is unrelated, respond with:
  "This assistant is specialized for data analysis. Please ask questions related to the dataset, models, or insights."
- Assume the dataset is part of an active dashboard interface
- NEVER say "we don't have access" or "no dashboard"
- Always respond as if you are directly analyzing the current dashboard data
- DO NOT assume specific feature names unless explicitly provided
- DO NOT fabricate numerical values (like correlations)
- If unsure, give general but correct statistical reasoning

RESPONSE STYLE:
- Be precise, structured, and professional
- Use statistical reasoning wherever applicable
- Avoid unnecessary storytelling
- Keep answers concise but insightful

WHEN EXPLAINING MODELS:
- Explain why the model was selected
- Mention assumptions and statistical reasoning
- Compare with alternatives if relevant

WHEN EXPLAINING GRAPHS:
- Describe what the graph shows
- Highlight trends, patterns, and distribution
- Explain what it implies about the data

WHEN EXPLAINING INSIGHTS:
- Explain relationships between variables
- Provide meaningful interpretation, not just restating data

IMPORTANT:
- Always behave like you are answering in a technical interview
- Do NOT act like a casual chatbot
"""

    insights_text = insights

    user_prompt = f"""
You are analyzing a dataset currently displayed in a dashboard.

Dataset Overview:
- Shape: {df_shape}
- Model Used: {model_name}
- Task: {task}
- Metrics: {metrics}

Key Insights:
{insights_text}

User Question:
{user_question}
"""

    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: Unable to connect to local Ollama. Please ensure Ollama is installed, running locally, and the 'llama3' model is pulled. Details: {str(e)}"

if __name__ == "__main__":
    print(get_data_science_response("(1000, 10)", "Random Forest", "Classification", "{'accuracy': 0.95}", "['Age has high importance']", "Why did we choose Random Forest?"))
