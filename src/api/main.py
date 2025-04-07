from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
import sys
from datetime import datetime
import uuid
from typing import List, Optional
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
async def classify_text(request: TextRequest):
    try:
        # Here you would implement the actual text classification logic
        return {
            "text": request.text,
            "is_spam": False,  # Placeholder
            "confidence": 0.95,  # Placeholder
            "analysis_type": request.analysis_type,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_classify")
async def batch_classify(request: BatchTextRequest):
    try:
        results = []
        for text in request.texts:
            results.append({
                "text": text,
                "is_spam": False,  # Placeholder
                "confidence": 0.95,  # Placeholder
                "analysis_type": request.analysis_type,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4())
            })
        return {"results": results}
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 