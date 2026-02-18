# AI Automation Toolkit

Production-ready AI tools for text generation and sentiment analysis. Supports free and low-cost providers.

## Features

### 1. AI Text Generator
- Multiple providers: Ollama (local), HuggingFace, Google Gemini
- Chat mode with conversation history
- Customizable temperature and max tokens
- History saving/loading

### 2. Sentiment Analyzer
- Multiple methods: VADER, TextBlob, HuggingFace
- Batch processing
- File analysis with summary statistics
- JSON export

## Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

Choose based on what you need:

```bash
# For sentiment analysis (VADER)
pip install vaderSentiment

# For sentiment analysis (TextBlob)
pip install textblob
python -m textblob.download_corpora

# For HuggingFace API
pip install requests
```

## Setup

### Ollama (Local, Free)

1. Install Ollama: https://ollama.ai
2. Pull a model:
```bash
ollama pull llama3.2
```

No API key needed!

### HuggingFace (Free Tier)

1. Sign up: https://huggingface.co
2. Get API token: Settings → Access Tokens
3. Set environment variable:
```bash
export HUGGINGFACE_API_KEY=your_token_here
```

### Google Gemini (Free Tier)

1. Get API key: https://makersuite.google.com/app/apikey
2. Set environment variable:
```bash
export GEMINI_API_KEY=your_api_key_here
```

## Usage

### AI Text Generator

**Command Line:**

```bash
# Single prompt (Ollama)
python ai_text_generator.py --prompt "Write a short story about a robot"

# Chat mode (HuggingFace)
python ai_text_generator.py --provider huggingface --chat

# Custom parameters
python ai_text_generator.py --provider gemini --prompt "Explain AI" --max-tokens 200 --temperature 0.5
```

**Python API:**

```python
from ai_text_generator import AITextGenerator

# Initialize
generator = AITextGenerator(provider="ollama", model="llama3.2")

# Generate text
response = generator.generate(
    "Write a product description for wireless headphones",
    max_tokens=300,
    temperature=0.7
)
print(response)

# Chat mode
response = generator.chat("What is AI?")
print(response)

# Save history
generator.save_history("conversation.json")
```

### Sentiment Analyzer

**Command Line:**

```bash
# Single text (VADER - no API needed)
python sentiment_analyzer.py --text "This is amazing! I love it!"

# Analyze file (one text per line)
python sentiment_analyzer.py --file reviews.txt --output results.json

# Use TextBlob
python sentiment_analyzer.py --method textblob --text "Not great, but okay"

# Use HuggingFace (requires API key)
python sentiment_analyzer.py --method huggingface --file tweets.txt
```

**Python API:**

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize
analyzer = SentimentAnalyzer(method="vader")

# Analyze single text
result = analyzer.analyze("This product is terrible!")
print(result)
# Output:
# {
#   "method": "vader",
#   "label": "NEGATIVE",
#   "scores": {"positive": 0.0, "negative": 0.508, "neutral": 0.492, "compound": -0.5574},
#   "confidence": 0.5574
# }

# Batch analysis
texts = ["Great service!", "Awful experience", "It's okay"]
results = analyzer.analyze_batch(texts)

# Get summary
summary = analyzer.get_summary(results)
print(f"Positive: {summary['positive_pct']:.1f}%")
```

## Examples

### Generate Product Descriptions

```python
from ai_text_generator import AITextGenerator

generator = AITextGenerator(provider="ollama")

products = ["Bluetooth Speaker", "Smart Watch", "LED Lamp"]

for product in products:
    description = generator.generate(
        f"Write a compelling 2-sentence product description for {product}",
        max_tokens=100
    )
    print(f"{product}: {description}\n")
```

### Analyze Customer Reviews

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(method="vader")

reviews = [
    "Best purchase ever! Highly recommend.",
    "Stopped working after 2 weeks. Waste of money.",
    "Decent quality for the price."
]

results = analyzer.analyze_batch(reviews)
summary = analyzer.get_summary(results)

print(f"Customer satisfaction: {summary['positive_pct']:.1f}%")
```

### Social Media Monitoring

```python
from sentiment_analyzer import SentimentAnalyzer
import json

# Analyze tweets from file
analyzer = SentimentAnalyzer(method="vader")
results = analyzer.analyze_file("tweets.txt", "sentiment_results.json")

# Filter negative mentions
negative = [r for r in results if r['label'] == 'NEGATIVE' and r['confidence'] > 0.5]

print(f"Found {len(negative)} strongly negative mentions")
```

## Comparison of Methods

### Text Generation

| Provider | Cost | Speed | Quality | Local |
|----------|------|-------|---------|-------|
| Ollama | Free | Fast | Good | ✅ |
| HuggingFace | Free tier | Medium | Good | ❌ |
| Gemini | Free tier | Fast | Excellent | ❌ |

### Sentiment Analysis

| Method | Installation | Speed | Accuracy | Best For |
|--------|-------------|-------|----------|----------|
| VADER | `pip install vaderSentiment` | Very Fast | Good for social media | Short texts, social media |
| TextBlob | `pip install textblob` | Fast | General purpose | Simple sentiment detection |
| HuggingFace | API key required | Medium | High | Production use |

## Tips

- **Use Ollama for development** - Free, fast, no rate limits
- **VADER for social media** - Optimized for tweets, reviews
- **Gemini for quality** - Best free tier output quality
- **Save API costs** - Use local Ollama when possible

## Troubleshooting

**"Connection refused" with Ollama:**
```bash
# Start Ollama service
ollama serve
```

**"Model loading error":**
```bash
# Pull the model first
ollama pull llama3.2
```

**HuggingFace rate limits:**
- Free tier: 1000 requests/day
- Upgrade for more: https://huggingface.co/pricing

## License

MIT

## Author

v0id-lab
