import os
import time
from typing import Dict, Optional
from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class TierConfig:
    def __init__(self):
        load_dotenv()
        self.api_key_header = APIKeyHeader(name="X-API-Key")
        
        # Define tier limits
        self.tier_limits = {
            "free": {
                "requests_per_day": 100,
                "batch_size": 10
            },
            "basic": {
                "requests_per_day": 1000,
                "batch_size": 50
            },
            "premium": {
                "requests_per_day": 10000,
                "batch_size": 100
            }
        }
        
        # Initialize request tracking
        self.request_counts: Dict[str, Dict] = {}
        
    async def verify_api_key(self, api_key: str = Depends(APIKeyHeader(name="X-API-Key"))) -> str:
        """
        Verify the API key and return the tier.
        """
        if not api_key:
            raise HTTPException(status_code=401, detail="API key is required")
            
        # In a real implementation, you would verify the API key against a database
        # For now, we'll use a simple mapping from the environment
        valid_keys = {
            os.getenv("FREE_API_KEY", "free_key"): "free",
            os.getenv("BASIC_API_KEY", "basic_key"): "basic",
            os.getenv("PREMIUM_API_KEY", "premium_key"): "premium"
        }
        
        tier = valid_keys.get(api_key)
        if not tier:
            raise HTTPException(status_code=401, detail="Invalid API key")
            
        return tier
        
    def check_rate_limit(self, tier: str) -> bool:
        """
        Check if the request is within rate limits.
        """
        current_time = time.time()
        
        # Initialize tracking for new API keys
        if tier not in self.request_counts:
            self.request_counts[tier] = {
                "count": 0,
                "reset_time": current_time + 86400  # 24 hours
            }
            
        # Reset counter if 24 hours have passed
        if current_time > self.request_counts[tier]["reset_time"]:
            self.request_counts[tier] = {
                "count": 0,
                "reset_time": current_time + 86400
            }
            
        # Check if under limit
        if self.request_counts[tier]["count"] >= self.tier_limits[tier]["requests_per_day"]:
            return False
            
        return True
        
    def release_request(self, tier: str):
        """
        Increment the request count for the tier.
        """
        if tier in self.request_counts:
            self.request_counts[tier]["count"] += 1
            
    def get_usage_stats(self, tier: str) -> Dict:
        """
        Get usage statistics for the tier.
        """
        if tier not in self.request_counts:
            return {
                "tier": tier,
                "requests_today": 0,
                "requests_remaining": self.tier_limits[tier]["requests_per_day"],
                "reset_time": time.time() + 86400
            }
            
        current_count = self.request_counts[tier]["count"]
        return {
            "tier": tier,
            "requests_today": current_count,
            "requests_remaining": self.tier_limits[tier]["requests_per_day"] - current_count,
            "reset_time": self.request_counts[tier]["reset_time"]
        } 