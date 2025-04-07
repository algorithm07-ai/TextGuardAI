import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "TextGuard AI API"
    assert response.json()["status"] == "operational"

def test_health(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_tiers(client):
    """Test tiers endpoint."""
    response = client.get("/tiers")
    assert response.status_code == 200
    tiers = response.json()
    assert "free" in tiers
    assert "basic" in tiers
    assert "premium" in tiers

@patch("src.core.DeepSeekMCPClient.process_text")
def test_analyze_text_success(mock_process, client):
    """Test successful text analysis."""
    mock_process.return_value = {"result": "success"}
    
    response = client.post(
        "/analyze",
        json={
            "text": "Test text",
            "tier": "free"
        }
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["result"] == {"result": "success"}

@patch("src.core.DeepSeekMCPClient.process_text")
def test_analyze_text_error(mock_process, client):
    """Test text analysis error handling."""
    mock_process.side_effect = Exception("Test error")
    
    response = client.post(
        "/analyze",
        json={
            "text": "Test text",
            "tier": "free"
        }
    )
    
    assert response.status_code == 500
    assert "error" in response.json()

@patch("src.core.DeepSeekMCPClient.batch_process")
def test_batch_analyze_success(mock_batch_process, client):
    """Test successful batch analysis."""
    mock_batch_process.return_value = [
        {"status": "success", "result": {"result": "success"}}
    ]
    
    response = client.post(
        "/batch",
        json={
            "texts": ["Test text"],
            "tier": "free"
        }
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["results"]) == 1

@patch("src.core.DeepSeekMCPClient.batch_process")
def test_batch_analyze_error(mock_batch_process, client):
    """Test batch analysis error handling."""
    mock_batch_process.side_effect = Exception("Test error")
    
    response = client.post(
        "/batch",
        json={
            "texts": ["Test text"],
            "tier": "free"
        }
    )
    
    assert response.status_code == 500
    assert "error" in response.json()

def test_get_stats(client):
    """Test stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "tier" in stats
    assert "cache_size" in stats 