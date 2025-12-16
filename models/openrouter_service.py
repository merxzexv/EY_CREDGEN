import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

class OpenRouterService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY") # Fallback for smooth transition if user only has one key set
        self.model_name = "google/gemma-3-27b-it:free"
        
        if self.api_key:
            logger.info("OpenRouter Service initialized.")
        else:
            logger.warning("No OPENROUTER_API_KEY found. Agent will likely fail.")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_response(
        self, 
        user_message: str, 
        system_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a response using OpenRouter (Google Gemma 2), mapped to the backend schema.
        """
        if not self.api_key:
             return {
                "message": "AI Service not configured (Missing API Key).",
                "suggestions": [],
                "worker": "none",
                "intent": "error"
            }

        # prompt construction to enforce JSON output for the frontend
        json_instruction = """
You must respond in strict JSON format.
Output schema:
{
  "response": "Your natural language response here (can include markdown)",
  "suggestions": ["Short follow-up option 1", "Short follow-up option 2", "Option 3"]
}
"""
        full_system_prompt = f"{system_prompt}\n{json_instruction}"

        # Construct payload
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "CredGen AI"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": full_system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=30 
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse the JSON content from the LLM
                try:
                    # Sometimes LLMs wrap JSON in markdown code blocks
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean_content)
                    
                    return {
                        "message": parsed.get("response", ""),
                        "suggestions": parsed.get("suggestions", []),
                        "worker": "none",
                        "intent": "openrouter_response",
                        "status": "success"
                    }
                except json.JSONDecodeError:
                    # Fallback if strict JSON fails
                    logger.warning(f"OpenRouter returned invalid JSON: {content[:100]}...")
                    return {
                        "message": content, # Return raw text
                        "suggestions": [],
                        "worker": "none",
                        "intent": "openrouter_response",
                        "status": "success_text_only"
                    }
            else:
                logger.error(f"OpenRouter API Error: {response.status_code} - {response.text}")
                return {
                    "message": "I'm having trouble connecting to the bank's AI system.",
                    "intent": "error",
                    "worker": "none"
                }
                
        except Exception as e:
            logger.error(f"OpenRouter Request Failed: {e}")
            return {
                "message": "Service unavailable.",
                "intent": "error",
                "worker": "none"
            }
