import os
import sys
import uvicorn

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the FastAPI app
from src.api.main import app

if __name__ == "__main__":
    print("Starting TextGuard AI API...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 