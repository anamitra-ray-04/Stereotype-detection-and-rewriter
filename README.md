# Stereotype Detection and Rewriting System

A sophisticated system that combines Large Language Models (LLM) and Machine Learning to detect and mitigate stereotypes in text.

## Features

1. **Input Collection**
   - Text input
   - File upload
   - Web scraping

2. **Preprocessing**
   - Text cleaning and normalization
   - Sentence segmentation
   - Tokenization
   - Stopword removal
   - Lemmatization

3. **Stereotype Detection**
   - Hybrid approach using BERT, Logistic Regression and LLM models
   - Multi-type stereotype identification
   - Severity scoring (0-1 scale)
   - Detailed analysis and explanation
   - Stereotype type classification

3. **Bias-free Rewriting**
   - Intelligent text rewriting to remove stereotypes
   - Maintains core message while eliminating bias
   - Grammar and context preservation
   - Factual alternatives to stereotypical statements

4. **User-friendly Interface**
   - Clean Streamlit web interface
   - Real-time analysis
   - Clear visualization of results
   - Side-by-side model comparisons

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Unix/macOS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the Streamlit interface
2. Enter text in the input area
3. Click "Analyze and Rewrite"
4. View results from both models:
   - LLM Analysis with severity scores and explanations
   - Logistic Regression probabilities and confidence
5. If stereotypes are detected, view the bias-free rewritten version

## Project Structure

```
stereotype-detection-and-rewriter/
├── app.py                  # Streamlit interface
├── detector.py            # Core detection and rewriting logic
├── requirements.txt       # Project dependencies
├── model-log-reg.pkl     # Trained logistic regression model
└── vectorizer-log-reg.pkl # Text vectorizer for ML model
```

## Requirements

- Python 3.8+
- Streamlit
- Python-dotenv
- Requests
- Scikit-learn
- Joblib
- Groq API access
