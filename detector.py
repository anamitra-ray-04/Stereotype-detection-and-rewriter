from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import json
import requests
import os
import joblib

class StereotypeDetector:
    def __init__(self, api_key: str):
        """
        Initialize the stereotype detector with Groq API and Logistic Regression model.
        
        Args:
            api_key (str): The API key for Groq
        """
        load_dotenv()
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Load logistic regression model and vectorizer
        model_path = os.path.join(os.path.dirname(__file__), "model-log-reg.pkl")
        vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer-log-reg.pkl")
        
        try:
            self.log_reg_model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            print(f"Warning: Could not load logistic regression model: {str(e)}")
            self.log_reg_model = None
            self.vectorizer = None

    def detect_stereotype(self, text: str) -> Dict[str, Any]:
        """
        Detect stereotypes in the given text using LLM.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, Any]: Detection results including stereotype presence,
                           severity, and explanation
        """
        prompt = f"""Carefully analyze this text for any stereotypes, biases, or prejudices. Consider all types:
        gender, racial, ethnic, cultural, religious, age-related, occupational, etc.

        Text: "{text}"

        Respond ONLY with a JSON object in this exact format:
        {{
            "has_stereotype": true/false,
            "severity_score": <float 0-1>,
            "explanation": "<detailed explanation>",
            "stereotype_type": "<type of stereotype if present, otherwise null>"
        }}

        Severity scoring guide:
        0.0-0.2: No stereotype/Very mild implications
        0.2-0.4: Mild stereotyping, subtle bias
        0.4-0.6: Moderate stereotyping, clear bias
        0.6-0.8: Strong stereotyping, concerning bias
        0.8-1.0: Severe stereotyping, highly problematic

        Be especially vigilant for:
        - Gender roles and expectations
        - Cultural or ethnic generalizations
        - Age-related assumptions
        - Professional or occupational stereotypes
        - Religious or belief-based prejudices
        - Socioeconomic stereotypes
        
        If unsure, lean towards identifying potential stereotypes rather than missing them.
        Provide specific, detailed explanation about why something is or isn't stereotypical.
        
        Return ONLY the JSON object, no additional text or explanation."""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "groq/compound",
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()  # Raise exception for non-200 status codes
            
            result = json.loads(response.json()["choices"][0]["message"]["content"])
            return {
                "success": True,
                "has_stereotype": result["has_stereotype"],
                "severity_score": result["severity_score"],
                "explanation": result["explanation"],
                "stereotype_type": result.get("stereotype_type")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def predict_with_logreg(self, text: str) -> Dict[str, Any]:
        """
        Predict stereotype using logistic regression model.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, Any]: Prediction results including probability
        """
        if self.log_reg_model is None or self.vectorizer is None:
            return {
                "success": False,
                "error": "Logistic regression model not loaded"
            }
            
        try:
            # Transform text using the vectorizer
            X = self.vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.log_reg_model.predict(X)[0]
            probabilities = self.log_reg_model.predict_proba(X)[0]
            
            # Get probability for predicted class
            confidence = probabilities.max()
            
            return {
                "success": True,
                "has_stereotype": bool(prediction),
                "prediction": int(prediction),  # 0 or 1
                "confidence": float(confidence),  # Probability score
                "probabilities": {
                    "no_stereotype": float(probabilities[0]),
                    "stereotype": float(probabilities[1])
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def rewrite_text(self, text: str) -> str:
        """
        Rewrite text to remove stereotypes using LLM.
        
        Args:
            text (str): The text to rewrite
            
        Returns:
            str: Rewritten text without stereotypes
        """
        prompt = f"""Extract the phrase from the given sentence that contains the stereotype. Rewrite only this phrase to completely remove any stereotypes, biases, or prejudices while maintaining the core message and being gramatically and semantically correct.
        Make these changes:
        1. Remove generalizations about groups
        2. Use inclusive and neutral language
        3. Focus on individual characteristics rather than group stereotypes
        4. Maintain the original meaning but remove biased assumptions
        5. Use factual statements instead of stereotypical ones

        Text: "{text}"

        Provide ONLY the rewritten text, no explanations or other text."""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "groq/compound",
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()  # Raise exception for non-200 status codes
            
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error in rewriting: {str(e)}"