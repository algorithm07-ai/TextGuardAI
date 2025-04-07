import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepSeekMCPError(Exception):
    """Base exception for DeepSeek MCP client errors."""
    pass

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
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
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
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text using the DeepSeek API with retry mechanism.
        
        Args:
            text: The text to process
            options: Additional options for processing
            
        Returns:
            Dict containing the processing results
            
        Raises:
            DeepSeekMCPError: If the API request fails after retries
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
            
        # Make request with retry
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error: {error_text}")
                        raise DeepSeekMCPError(f"API error: {response.status} - {error_text}")
                        
                    result = await response.json()
                    
                    # Cache result
                    self.cache[cache_key] = {
                        "result": result,
                        "timestamp": datetime.now()
                    }
                    
                    return result
                    
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise DeepSeekMCPError(f"Network error after {self.max_retries} attempts: {str(e)}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                raise DeepSeekMCPError(f"Error processing text: {str(e)}")
                
    async def batch_process(self, texts: List[str], options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple texts in parallel with error handling.
        
        Args:
            texts: List of texts to process
            options: Additional options for processing
            
        Returns:
            List of processing results
        """
        tasks = [self.process_text(text, options) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle errors
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing text {i}: {str(result)}")
                processed_results.append({
                    "error": str(result),
                    "text": texts[i],
                    "status": "error"
                })
            else:
                processed_results.append({
                    "result": result,
                    "text": texts[i],
                    "status": "success"
                })
                
        return processed_results
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dict containing usage statistics
        """
        return {
            "tier": self.tier,
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat(),
            "cache_hits": sum(1 for entry in self.cache.values() 
                            if (datetime.now() - entry["timestamp"]).total_seconds() < self.cache_ttl)
        } 