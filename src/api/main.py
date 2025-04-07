import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.core import DeepSeekMCPClient
from src.utils import get_tier_config

# Initialize FastAPI app
app = FastAPI(
    title="TextGuard AI API",
    description="API for text analysis and spam detection using DeepSeek MCP",
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

# Initialize MCP client
client = DeepSeekMCPClient(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
)

class TextRequest(BaseModel):
    text: str
    tier: Optional[str] = "free"
    options: Optional[Dict[str, Any]] = None

class BatchRequest(BaseModel):
    texts: list[str]
    tier: Optional[str] = "free"
    options: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "TextGuard AI API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/tiers")
async def get_tiers():
    """Get available API tiers and their configurations."""
    return get_tier_config()

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    """
    Analyze a single text using the specified tier.
    """
    try:
        # Set tier
        client.set_tier(request.tier)
        
        # Process text
        result = await client.process_text(request.text, request.options)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_analyze(request: BatchRequest):
    """
    Analyze multiple texts in parallel using the specified tier.
    """
    try:
        # Set tier
        client.set_tier(request.tier)
        
        # Process texts
        results = await client.batch_process(request.texts, request.options)
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get API usage statistics.
    """
    return client.get_usage_stats() 