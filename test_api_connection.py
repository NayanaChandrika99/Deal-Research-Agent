#!/usr/bin/env python3
"""
Quick test script to verify API connections.

Run this after setting up your .env file to ensure:
1. OpenAI embeddings API works
2. Cerebras LLM API works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dealgraph.config import settings
from dealgraph.embeddings import EmbeddingEncoder


def test_embeddings():
    """Test OpenAI embeddings API."""
    print("=" * 60)
    print("Testing OpenAI Embeddings API")
    print("=" * 60)
    
    try:
        encoder = EmbeddingEncoder()
        print(f"✓ Encoder initialized: {encoder}")
        
        # Test single embedding
        test_text = "US industrial distribution roll-up with multiple add-ons"
        print(f"\nEmbedding test text: '{test_text}'")
        
        embedding = encoder.embed_text(test_text)
        print(f"✓ Embedding generated successfully")
        print(f"  - Shape: {embedding.shape}")
        print(f"  - Dimensions: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        
        # Test batch embeddings
        test_texts = [
            "Healthcare services platform acquisition",
            "Software-as-a-Service roll-up strategy",
            "Manufacturing consolidation play"
        ]
        print(f"\nEmbedding {len(test_texts)} texts in batch...")
        
        embeddings = encoder.embed_texts(test_texts)
        print(f"✓ Batch embeddings generated successfully")
        print(f"  - Count: {len(embeddings)}")
        print(f"  - Each shape: {embeddings[0].shape}")
        
        print("\n✅ OpenAI Embeddings API: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ OpenAI Embeddings API: FAILED")
        print(f"Error: {e}")
        return False


def test_cerebras():
    """Test Cerebras LLM API."""
    print("\n" + "=" * 60)
    print("Testing Cerebras LLM API")
    print("=" * 60)
    
    try:
        from openai import OpenAI
        
        # Cerebras uses OpenAI-compatible API
        client = OpenAI(
            api_key=settings.CEREBRAS_API_KEY,
            base_url=settings.CEREBRAS_BASE_URL
        )
        print(f"✓ Client initialized")
        print(f"  - Model: {settings.DSPY_MODEL}")
        print(f"  - Base URL: {settings.CEREBRAS_BASE_URL}")
        
        # Test simple completion
        print(f"\nSending test prompt...")
        response = client.chat.completions.create(
            model=settings.DSPY_MODEL,
            messages=[
                {"role": "user", "content": "Say 'Hello from Cerebras!' and nothing else."}
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✓ Response received: '{result}'")
        print(f"  - Tokens used: {response.usage.total_tokens}")
        
        print("\n✅ Cerebras LLM API: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Cerebras LLM API: FAILED")
        print(f"Error: {e}")
        return False


def main():
    """Run all API tests."""
    print("\n🔍 DealGraph API Connection Test\n")
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("\nPlease create .env file:")
        print("  1. Copy env.example to .env")
        print("  2. Add your API keys")
        print("  3. Run this script again")
        sys.exit(1)
    
    print(f"✓ Found .env file")
    
    # Validate settings
    try:
        settings.validate()
        print(f"✓ API keys loaded from environment")
    except ValueError as e:
        print(f"❌ {e}")
        print("\nPlease check your .env file and add missing API keys.")
        sys.exit(1)
    
    print()
    
    # Run tests
    embeddings_ok = test_embeddings()
    cerebras_ok = test_cerebras()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"OpenAI Embeddings: {'✅ PASS' if embeddings_ok else '❌ FAIL'}")
    print(f"Cerebras LLM:      {'✅ PASS' if cerebras_ok else '❌ FAIL'}")
    
    if embeddings_ok and cerebras_ok:
        print("\n🎉 All API connections working! Ready to build.")
        sys.exit(0)
    else:
        print("\n⚠️  Some API connections failed. Please check your API keys.")
        sys.exit(1)


if __name__ == "__main__":
    main()

