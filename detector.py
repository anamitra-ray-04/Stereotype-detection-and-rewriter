from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import json
import requests
import os
import joblib

# Optional: transformers-based BERT pipeline will be loaded if the package
# and local model files exist. Loading is guarded so missing package/model
# does not break existing functionality.

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

        # Try to load local transformers model (BERT) from ./transformers
        self.bert_pipeline = None
        try:
            # Import inside try so missing package won't break the app
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

            model_dir = os.path.join(os.path.dirname(__file__), "transformers")

            # Use from_pretrained with the local directory. If files are present
            # this should load the tokenizer and model.
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)

            # Create a text-classification pipeline. device=-1 forces CPU usage
            self.bert_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1
            )
        except Exception as e:
            # Keep behavior non-breaking: log warning and leave bert_pipeline as None
            print(f"Warning: Could not load BERT model or transformers package: {str(e)}")
            self.bert_pipeline = None

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

    def predict_with_bert(self, text: str) -> Dict[str, Any]:
        """
        Predict stereotype using the local transformers (BERT) model.

        This method is optional: if the transformers package or model files
        aren't available, it returns success=False with an error message.
        The returned structure mirrors `predict_with_logreg` where possible.
        """
        if self.bert_pipeline is None:
            return {
                "success": False,
                "error": "BERT model not loaded"
            }

        try:
            # Request all class scores so we can map probabilities consistently
            raw = self.bert_pipeline(text, truncation=True, return_all_scores=True)

            # raw is typically a list with one element (for the input string)
            if isinstance(raw, list) and len(raw) > 0:
                scores_list = raw[0]
            else:
                scores_list = raw

            # scores_list is expected to be a list of dicts: [{"label":..., "score":...}, ...]
            # Try to order by numeric label index if labels are like LABEL_0, LABEL_1
            try:
                ordered = sorted(
                    scores_list,
                    key=lambda x: int(''.join(filter(str.isdigit, x.get('label', '0'))))
                )
                probabilities = [float(item.get('score', 0.0)) for item in ordered]
            except Exception:
                # Fallback: use given order
                probabilities = [float(item.get('score', 0.0)) for item in scores_list]

            # If binary, assume index 0 -> no_stereotype, index 1 -> stereotype
            if len(probabilities) == 2:
                no_st_prob = probabilities[0]
                st_prob = probabilities[1]
                prediction = 1 if st_prob > no_st_prob else 0
                confidence = max(no_st_prob, st_prob)

                return {
                    "success": True,
                    "has_stereotype": bool(prediction),
                    "prediction": int(prediction),
                    "confidence": float(confidence),
                    "probabilities": {
                        "no_stereotype": float(no_st_prob),
                        "stereotype": float(st_prob)
                    }
                }

            # Non-binary or unexpected output: return generic probability list
            pred_idx = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
            confidence = float(max(probabilities))
            return {
                "success": True,
                "has_stereotype": bool(pred_idx == 1),
                "prediction": pred_idx,
                "confidence": confidence,
                "probabilities": {
                    "class_probabilities": probabilities
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
    
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