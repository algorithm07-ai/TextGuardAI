import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepSeekMCPClient:
    """
    Client for interacting with the DeepSeek API using the MCP protocol.
    """
    
    def __init__(self, api_key: str, tier: str = "free"):
        """
        Initialize the DeepSeek MCP client.
        
        Args:
            api_key: The DeepSeek API key
            tier: The API access tier (free, basic, premium)
        """
        self.api_key = api_key
        self.tier = tier
        self.base_url = "https://api.deepseek.com/v1"
        self.session = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def __aenter__(self):
        """Create aiohttp session when entering context."""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting context."""
        if self.session:
            await self.session.close()
            
    def set_tier(self, tier: str):
        """
        Set the API access tier.
        
        Args:
            tier: The API access tier (free, basic, premium)
        """
        if tier not in ["free", "basic", "premium"]:
            raise ValueError("Tier must be one of: free, basic, premium")
        self.tier = tier
        logger.info(f"API tier set to: {tier}")
        
    async def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text using the DeepSeek API.
        
        Args:
            text: The text to process
            options: Additional options for processing
            
        Returns:
            Dict containing the processing results
        """
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
        # Check cache
        cache_key = f"{text}:{json.dumps(options or {})}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now() - cache_entry["timestamp"]).total_seconds() < self.cache_ttl:
                logger.info("Using cached result")
                return cache_entry["result"]
                
        # Prepare request
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": text}],
            "tier": self.tier
        }
        
        if options:
            payload.update(options)
            
        # Make request
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    raise Exception(f"API error: {response.status} - {error_text}")
                    
                result = await response.json()
                
                # Cache result
                self.cache[cache_key] = {
                    "result": result,
                    "timestamp": datetime.now()
                }
                
                return result
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
            
    async def batch_process(self, texts: List[str], options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple texts in parallel.
        
        Args:
            texts: List of texts to process
            options: Additional options for processing
            
        Returns:
            List of processing results
        """
        tasks = [self.process_text(text, options) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dict containing usage statistics
        """
        return {
            "tier": self.tier,
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat()
        } 