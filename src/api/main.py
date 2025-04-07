from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
from datetime import datetime
import uuid
from typing import List, Optional
from dotenv import load_dotenv

from src.core.data_processor import DataProcessor
from src.core.integration import DeepSeekIntegration
from src.utils.tier_config import TierConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TextGuard AI",
    description="AI-powered text classification and spam detection API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
tier_config = TierConfig()
data_processor = DataProcessor()
deepseek_integration = DeepSeekIntegration()

class TextRequest(BaseModel):
    text: str
    analysis_type: Optional[str] = "spam"

class BatchTextRequest(BaseModel):
    texts: List[str]
    analysis_type: Optional[str] = "spam"

@app.get("/")
async def root():
    return {
        "name": "TextGuard AI",
        "version": "1.0.0",
        "description": "AI-powered text classification and spam detection API",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/tools")
async def get_tools():
    return {
        "tools": [
            {
                "name": "text_classifier",
                "description": "Classifies text as spam or not spam",
                "parameters": {
                    "text": "string",
                    "analysis_type": "string (optional)"
                }
            }
        ]
    }

@app.post("/classify")
async def classify_text(request: TextRequest, api_key: str = Depends(tier_config.verify_api_key)):
    try:
        # Check rate limit
        if not tier_config.check_rate_limit(api_key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Process text
        result = await deepseek_integration.analyze_with_deepseek(request.text)
        
        # Release request count
        tier_config.release_request(api_key)
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_classify")
async def batch_classify(request: BatchTextRequest, api_key: str = Depends(tier_config.verify_api_key)):
    try:
        # Check rate limit
        if not tier_config.check_rate_limit(api_key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Process texts
        results = await deepseek_integration.batch_analyze(request.texts)
        
        # Release request count
        tier_config.release_request(api_key)
        
        return results
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usage")
async def get_usage(api_key: str = Depends(tier_config.verify_api_key)):
    try:
        usage = tier_config.get_usage_stats(api_key)
        return usage
    except Exception as e:
        logger.error(f"Error getting usage stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 