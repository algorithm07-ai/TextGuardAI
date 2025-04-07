# TextGuard AI

TextGuard AI is a powerful text classification and spam detection API that uses advanced AI models to analyze and classify text content. It provides a simple REST API interface for both single text analysis and batch processing.

## Features

- Text classification and spam detection
- Batch processing support
- Tiered API access (Free, Basic, Premium)
- Rate limiting and usage tracking
- Detailed analysis with confidence scores
- Easy integration with existing applications

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/textguard-ai.git
cd textguard-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
FREE_API_KEY=free_key
BASIC_API_KEY=basic_key
PREMIUM_API_KEY=premium_key
```

## Usage

1. Start the server:
```bash
python -m src.api.main
```

2. The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /`: Root endpoint with API information
- `GET /health`: Health check endpoint
- `GET /tools`: List available tools and their parameters
- `POST /classify`: Classify a single text
- `POST /batch_classify`: Classify multiple texts
- `GET /usage`: Get usage statistics

### Example Requests

1. Single Text Classification:
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text here", "analysis_type": "spam"}'
```

2. Batch Classification:
```bash
curl -X POST "http://localhost:8000/batch_classify" \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text 1", "Text 2"], "analysis_type": "spam"}'
```

## API Tiers

### Free Tier
- 100 requests per day
- Maximum batch size: 10 texts
- Basic analysis

### Basic Tier
- 1,000 requests per day
- Maximum batch size: 50 texts
- Detailed analysis

### Premium Tier
- 10,000 requests per day
- Maximum batch size: 100 texts
- Advanced analysis with priority processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 