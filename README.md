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
```text
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
4. View results from different models:
   - LLM Analysis with severity scores and explanations
   - Logistic Regression probabilities and confidence
   - BERT probabilities and confidence
5. If stereotypes are detected, view the bias-free rewritten version

## Project Structure

```
stereotype-detection-and-rewriter/
├── app.py                     # Streamlit interface (UI)
├── detector.py                # Core detection & rewriting logic (LLM, logreg, BERT)
├── requirements.txt           # Project dependencies (BERT extras appended)
├── model-log-reg.pkl          # Trained logistic regression model (binary classifier)
├── vectorizer-log-reg.pkl     # Text vectorizer used by logistic regression
├── transformers/              # Local transformers model files (tokenizer/model) 
└── README.md                  # This file
```

## Requirements

- Python 3.8+
- Streamlit
- python-dotenv
- requests
- scikit-learn
- joblib
- Groq API access (set `GROQ_API_KEY` in `.env`)
- Optional for BERT: `transformers`, `safetensors`, `torch`
