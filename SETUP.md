# Setup Guide

## Quick Start

### 1. Create your `.env` file

```bash
cp env.example .env
```

Then edit `.env` and add your API keys:
- **CEREBRAS_API_KEY**: Get from https://cerebras.ai/
- **OPENAI_API_KEY**: Get from https://platform.openai.com/api-keys

### 2. Install dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Test API connections

```bash
python test_api_connection.py
```

You should see:
```
✅ OpenAI Embeddings API: WORKING
✅ Cerebras LLM API: WORKING
🎉 All API connections working! Ready to build.
```

## What's Configured

### OpenAI Embeddings API
- **Model**: `text-embedding-3-small`
- **Dimensions**: 1536
- **Cost**: ~$0.02 per million tokens
- **Usage**: Converting deal descriptions and queries into vectors for semantic search

### Cerebras LLM API
- **Model**: `llama3.1-8b`
- **Speed**: ~2200 tokens/sec
- **Cost**: $0.10 per million tokens (input + output)
- **Usage**: 
  - Deal reasoning
  - DSPy prompt optimization
  - LLM-as-judge quality evaluation

## API Request Examples

### Embeddings (OpenAI)

```python
from dealgraph.embeddings import EmbeddingEncoder

encoder = EmbeddingEncoder()

# Single text
embedding = encoder.embed_text("US industrial distribution roll-up")
print(embedding.shape)  # (1536,)

# Batch
embeddings = encoder.embed_texts([
    "Healthcare platform acquisition",
    "Software consolidation play"
])
print(len(embeddings))  # 2
```

### LLM (Cerebras)

```python
from openai import OpenAI
from dealgraph.config import settings

client = OpenAI(
    api_key=settings.CEREBRAS_API_KEY,
    base_url=settings.CEREBRAS_BASE_URL
)

response = client.chat.completions.create(
    model="llama3.1-8b",
    messages=[
        {"role": "user", "content": "Analyze this deal..."}
    ],
    temperature=0.7,
    max_tokens=2000
)

print(response.choices[0].message.content)
```

## Troubleshooting

### "CEREBRAS_API_KEY not found"
- Make sure you created `.env` file (not just `env.example`)
- Check that the file contains `CEREBRAS_API_KEY=your-key-here`
- No quotes needed around the key value

### "OPENAI_API_KEY not found"
- Same as above for OpenAI key
- Get key from https://platform.openai.com/api-keys

### Import errors
- Make sure you're in the virtual environment: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -e .`

### API connection fails
- Check your API keys are valid
- Verify you have credits/quota on both platforms
- Check internet connection

## Next Steps

Once the test script passes, you're ready to start building! See:
- **SPECIFICATION.md** for detailed implementation plan
- **ARCHITECTURE.md** for system design
- **PROMPT_OPTIMIZATION.md** for DSPy optimization guide

