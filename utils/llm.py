import ollama

MODEL = "qwen3:0.6b"

def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama LLM and retrieves the response.
    """
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,  # Skip chain-of-thought for speed
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ Ollama error: {str(e)}"


def get_data_science_response(df_shape, model_name, task, metrics, insights, user_question: str) -> str:
    """
    Submits a context-aware question to the local LLM adopting an expert data scientist persona.
    Returns the full response as a string.
    """
    system_prompt = """You are a professional Data Scientist and Statistical Analyst.

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
- Keep answers concise but insightful (3-5 sentences max)"""

    user_prompt = f"""Dataset Overview:
- Shape: {df_shape}
- Model Used: {model_name}
- Task: {task}
- Metrics: {metrics}

Key Insights:
{insights}

User Question:
{user_question}"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            think=False,  # Skip chain-of-thought for speed
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ Ollama error: {str(e)}"


def stream_data_science_response(df_shape, model_name, task, metrics, insights, user_question: str):
    """
    Generator: streams the LLM response token by token for use with st.write_stream().
    """
    system_prompt = """You are a professional Data Scientist and Statistical Analyst.

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
- Keep answers concise but insightful (3-5 sentences max)"""

    user_prompt = f"""Dataset Overview:
- Shape: {df_shape}
- Model Used: {model_name}
- Task: {task}
- Metrics: {metrics}

Key Insights:
{insights}

User Question:
{user_question}"""

    try:
        stream = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            think=False,
            stream=True,
        )
        for chunk in stream:
            token = chunk['message']['content']
            if token:
                yield token
    except Exception as e:
        yield f"⚠️ Ollama error: {str(e)}"


if __name__ == "__main__":
    print(get_data_science_response(
        "(1000, 10)", "Random Forest", "Classification",
        "{'accuracy': 0.95}", "['Age has high importance']",
        "Why did we choose Random Forest?"
    ))
