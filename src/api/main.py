from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="TextGuard AI API",
    description="API for text classification and spam detection",
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
        "description": "Text classification and spam detection API"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/classify")
async def classify_text(request: TextRequest):
    try:
        # Here you would implement the actual text classification logic
        return {
            "text": request.text,
            "is_spam": False,  # Placeholder
            "confidence": 0.95,  # Placeholder
            "analysis_type": request.analysis_type
        }
    except Exception as e:
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
                "analysis_type": request.analysis_type
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 