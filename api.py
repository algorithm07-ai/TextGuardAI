import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from data_processor import DataProcessor
from train import SimpleClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SMS Spam Classifier API",
    description="An API for classifying SMS messages as spam or ham",
    version="1.0.0"
)

# Initialize model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
processor = None

class Message(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    text: str
    is_spam: bool
    confidence: float
    preprocessed_text: str

@app.on_event("startup")
async def load_model():
    global model, processor
    try:
        logger.info("Loading model and processor...")
        model = SimpleClassifier().to(device)
        model.load_state_dict(torch.load('best_model.pt'))
        model.eval()
        
        processor = DataProcessor()
        # Load a small sample to fit the vectorizer
        df = processor.load_data('SMSSpamCollection')
        processor.prepare_data(df)
        
        logger.info("Model and processor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not load model")

@app.get("/health")
async def health_check():
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy"}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_message(message: Message):
    try:
        # Preprocess the text
        preprocessed_text = processor.preprocess_text(message.text)
        
        # Convert to features
        features = processor.vectorizer.transform([preprocessed_text]).toarray()[0]
        features = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            confidence = probabilities[0][prediction[0]].item()
        
        return ClassificationResponse(
            text=message.text,
            is_spam=bool(prediction[0].item()),
            confidence=confidence,
            preprocessed_text=preprocessed_text
        )
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 