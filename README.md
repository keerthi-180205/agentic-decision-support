# Autonomous Data Analytics System

An AI-driven system that automatically analyzes datasets, selects the most suitable machine learning model, and generates actionable insights without manual intervention.

Developed during a hackathon as a collaborative project.

---

## Problem Statement

Data analysis and model selection require significant manual effort and expertise. Beginners often struggle to preprocess data, choose the right model, and interpret results effectively.

---

## System Capabilities

- Automatically detects dataset type (classification or regression)  
- Selects the most suitable machine learning model  
- Trains and evaluates multiple models  
- Generates performance metrics  
- Produces human-readable insights  
- Provides an interactive web interface for user interaction  
- Integrates chatbot-based interaction using Ollama  

---

## My Contributions

- Implemented automated model selection logic (classification vs regression)  
- Built model training and evaluation pipeline using Scikit-learn  
- Developed insight generation module using feature importance and correlations  
- Integrated Ollama-based chatbot for interactive insights  
- Contributed to backend integration with Streamlit interface  

---

## Team Contributions

- **Keerthi N (ML Engineer):** Designed and implemented automated model selection pipeline, training workflow, insight generation, and integrated Ollama-based chatbot for interactive insights  
- **Nithish S Gowda (Data Engineer):** Handled data preprocessing, cleaning, and feature engineering  
- **Chinmay CP (Data Analyst):** Performed exploratory data analysis and visualization  
- **Tejas Gupta (Frontend Developer):** Built Streamlit interface and integrated backend components  

---

## Tech Stack

- Language: Python  
- Libraries: Pandas, NumPy, Scikit-learn  
- Visualization: Matplotlib, Seaborn  
- Interface: Streamlit  
- LLM Integration: Ollama  

---

## System Architecture
```
CSV Data
↓
preprocess.py → cleaned X, y
↓
model.py → best_model, score
↓
insights.py → generated insights
↓
app.py → Streamlit UI
↓
Ollama → chatbot-based interaction

```
---

## Output

- Automatically selected best model for given dataset  
- Model performance metrics (accuracy / regression score)  
- Generated insights highlighting key influencing features  
- Interactive chatbot-based insights using Ollama  

### 1. Dataset Upload & Preview
![Dataset Upload](assets/ui.png)<img width="1215" height="826" alt="Screenshot 2026-04-11 200940" src="https://github.com/user-attachments/assets/262b0fa4-122c-4d49-985c-c7a15d6012a9" />


### 2. Data Processing Pipeline
![Processing](assets/processing.png)<img width="1196" height="830" alt="Screenshot 2026-04-11 201030" src="https://github.com/user-attachments/assets/bcc4200c-4023-444c-991a-3d981cdfb034" />


### 3. Data Visualization
![Visualization](assets/graph.png)<img width="1236" height="717" alt="Screenshot 2026-04-11 201044" src="https://github.com/user-attachments/assets/e75c0d8c-af2d-4e39-b6d0-361fc9fb209a" />


### 4. Model Performance
![Model Output](assets/model.png)<img width="1802" height="671" alt="Screenshot 2026-04-11 201149" src="https://github.com/user-attachments/assets/47550608-90a5-408b-a4ad-6d31a543d057" />


### 5. AI Chatbot Interaction
![Chatbot](assets/chat.png)<img width="622" height="552" alt="Screenshot 2026-04-11 201612" src="https://github.com/user-attachments/assets/5e025d66-e962-49b6-985d-6e66ec9b6fe2" />


---

## Setup & Installation
### 1. Clone the repository
```bash
git clone https://github.com/your-username/autonomous-data-analytics-system.git
cd autonomous-data-analytics-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```
---

## Model Selection & Training Pipeline

### detect_problem_type(y)
Determines whether the task is classification or regression based on the target variable.

### train_and_select_model(X, y)
- Trains multiple models:
  - Classification: Logistic Regression, Random Forest Classifier  
  - Regression: Linear Regression, Random Forest Regressor  
- Selects the best model based on performance score  

### evaluate_model(model, X_test, y_test)
- Evaluates the selected model on test data  
- Returns performance score  

---

## Insight Generation Module

### generate_insights(df, model)

- Uses feature importance for tree-based models  
- Uses correlation analysis for linear models  
- Identifies top influential features  
- Generates human-readable insights  
- Provides decision-support suggestions  

---

## Dependencies

Core libraries used in the project:

- pandas, numpy → data processing  
- scikit-learn → model training  
- matplotlib, seaborn → visualization  
- streamlit → web interface   
