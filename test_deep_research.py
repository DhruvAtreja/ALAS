#!/usr/bin/env python3
"""
Simple test for Deep Research API functionality
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.deep_research_client import create_deep_research_client, DeepResearchError
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def simple_research_test():
    """Test basic research functionality"""
    
    if not settings.openai.api_key:
        print("❌ OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    
    print("🔍 Testing Deep Research API...")
    print("=" * 50)
    
    try:
        client = create_deep_research_client()
        
        # Simple test query
        query = "What is machine learning?"
        
        print(f"Query: {query}")
        print("Making API call...")
        
        response = await client.research_topic(
            query=query,
            domain="AI/ML",
            depth="fast"  # Use faster model for testing
        )
        
        print("✅ API call successful!")
        print(f"Model used: {response.model}")
        print(f"Content length: {len(response.content)} characters")
        print(f"Cost estimate: ${response.cost_estimate:.4f}")
        
        if response.content and len(response.content) > 100:
            print("\nFirst 200 characters of response:")
            print("-" * 40)
            print(response.content[:200] + "...")
            print("-" * 40)
            return True
        else:
            print("⚠️  Response seems too short or empty")
            print(f"Full response: {response.content}")
            return False
            
    except DeepResearchError as e:
        print(f"❌ Deep Research Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def test_model_availability():
    """Test which models are available"""
    
    print("\n🔧 Testing Model Availability...")
    print("=" * 50)
    
    from openai import AsyncOpenAI
    
    try:
        client = AsyncOpenAI(api_key=settings.openai.api_key)
        
        models_to_test = ["o3", "o4-mini", "gpt-4o", "gpt-4o-mini"]
        
        for model in models_to_test:
            try:
                print(f"Testing {model}... ", end="")
                # Prepare parameters based on model
                params = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}]
                }
                
                # Use correct token parameter for different models
                if model in ["o3", "o4-mini"]:
                    params["max_completion_tokens"] = 100000
                else:
                    params["max_tokens"] = 16384
                
                response = await client.chat.completions.create(**params)
                print("✅ Available")
            except Exception as e:
                error_msg = str(e)
                if "does not exist" in error_msg or "model_not_found" in error_msg:
                    print("❌ Not available")
                else:
                    print(f"⚠️  Error: {error_msg}")
                    print(f"     Full error: {repr(e)}")
                    
    except Exception as e:
        print(f"❌ Could not test models: {e}")


async def main():
    """Run all tests"""
    
    print("🚀 Deep Research API Test")
    print("=" * 60)
    
    # Test model availability first
    await test_model_availability()
    
    # Test basic research
    success = await simple_research_test()
    
    if success:
        print("\n🎉 Deep Research API is working correctly!")
    else:
        print("\n❌ Deep Research API test failed")
        print("\nTroubleshooting tips:")
        print("1. Check your OpenAI API key")
        print("2. Verify you have access to the models")
        print("3. Check your internet connection")
        print("4. Review the error messages above")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 