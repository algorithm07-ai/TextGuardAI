import os
import aiohttp
import logging
from typing import List, Dict, Any
import json
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class DeepSeekIntegration:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    async def analyze_with_deepseek(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text using DeepSeek API.
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a spam detection expert. Analyze the given text and determine if it's spam or not. Provide a confidence score and explanation."
                        },
                        {
                            "role": "user",
                            "content": f"Analyze this text for spam: {text}"
                        }
                    ],
                    "temperature": 0.3
                }
                
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error: {error_text}")
                        raise Exception(f"DeepSeek API error: {response.status}")
                        
                    result = await response.json()
                    analysis = result['choices'][0]['message']['content']
                    
                    # Parse the analysis to extract key information
                    is_spam = "spam" in analysis.lower()
                    confidence = self._extract_confidence(analysis)
                    
                    return {
                        "text": text,
                        "is_spam": is_spam,
                        "confidence": confidence,
                        "analysis": analysis
                    }
                    
        except Exception as e:
            logger.error(f"Error in DeepSeek analysis: {str(e)}")
            raise
            
    async def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in parallel using DeepSeek API.
        """
        try:
            tasks = [self.analyze_with_deepseek(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out any errors and log them
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing text {i}: {str(result)}")
                else:
                    valid_results.append(result)
                    
            return valid_results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            raise
            
    def _extract_confidence(self, analysis: str) -> float:
        """
        Extract confidence score from the analysis text.
        """
        try:
            # Look for percentage or decimal in the text
            import re
            confidence_match = re.search(r'(\d+(?:\.\d+)?%?)\s*(?:confidence|probability|likelihood)', analysis.lower())
            if confidence_match:
                confidence_str = confidence_match.group(1)
                # Convert percentage to decimal if needed
                if '%' in confidence_str:
                    return float(confidence_str.strip('%')) / 100
                return float(confidence_str)
            return 0.5  # Default confidence if not found
        except:
            return 0.5  # Default confidence if extraction fails 