import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.core import DeepSeekMCPClient, DeepSeekMCPError

@pytest.fixture
def client():
    """Create a test client instance."""
    return DeepSeekMCPClient(api_key="test-key")

@pytest.mark.asyncio
async def test_process_text_success(client):
    """Test successful text processing."""
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Test response",
                    "role": "assistant"
                }
            }
        ]
    }
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = MagicMock(
            return_value=mock_response
        )
        
        result = await client.process_text("Test text")
        assert result == mock_response

@pytest.mark.asyncio
async def test_process_text_retry(client):
    """Test retry mechanism on rate limit."""
    with patch("aiohttp.ClientSession.post") as mock_post:
        # First call returns rate limit
        mock_post.return_value.__aenter__.return_value.status = 429
        mock_post.return_value.__aenter__.return_value.headers = {"Retry-After": "1"}
        
        # Second call succeeds
        mock_success = MagicMock()
        mock_success.status = 200
        mock_success.json = MagicMock(return_value={"result": "success"})
        
        mock_post.side_effect = [
            MagicMock(__aenter__=MagicMock(return_value=mock_post.return_value.__aenter__.return_value)),
            MagicMock(__aenter__=MagicMock(return_value=mock_success))
        ]
        
        result = await client.process_text("Test text")
        assert result == {"result": "success"}
        assert mock_post.call_count == 2

@pytest.mark.asyncio
async def test_process_text_error(client):
    """Test error handling."""
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 500
        mock_post.return_value.__aenter__.return_value.text = MagicMock(
            return_value="Internal Server Error"
        )
        
        with pytest.raises(DeepSeekMCPError):
            await client.process_text("Test text")

@pytest.mark.asyncio
async def test_batch_process(client):
    """Test batch processing."""
    texts = ["Text 1", "Text 2"]
    mock_response = {"result": "success"}
    
    with patch.object(client, "process_text", return_value=mock_response):
        results = await client.batch_process(texts)
        
        assert len(results) == 2
        assert all(r["status"] == "success" for r in results)
        assert all(r["result"] == mock_response for r in results)

def test_set_tier(client):
    """Test tier setting."""
    client.set_tier("premium")
    assert client.tier == "premium"
    
    with pytest.raises(ValueError):
        client.set_tier("invalid_tier")

def test_get_usage_stats(client):
    """Test usage statistics."""
    stats = client.get_usage_stats()
    assert "tier" in stats
    assert "cache_size" in stats
    assert "cache_hits" in stats
    assert "timestamp" in stats 