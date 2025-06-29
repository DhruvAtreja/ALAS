#!/usr/bin/env python3
"""
Example script demonstrating the Deep Research API client
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.deep_research_client import create_deep_research_client
from core.curriculum_generator import create_curriculum_generator
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


async def example_basic_research():
    """Example of basic topic research"""
    print("🔍 Example 1: Basic Topic Research")
    print("=" * 50)
    
    client = create_deep_research_client()
    
    # Research a specific topic
    query = "LangGraph and its applications in AI agent workflows"
    
    try:
        response = await client.research_topic(
            query=query,
            domain="AI Agents",
            depth="comprehensive"
        )
        
        print(f"Research completed for: {query}")
        print(f"Model used: {response.model}")
        print(f"Content length: {len(response.content)} characters")
        print(f"Estimated cost: ${response.cost_estimate:.4f}")
        print(f"Research ID: {response.id}")
        print("\nFirst 500 characters of research:")
        print("-" * 30)
        print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
        print()
        
    except Exception as e:
        print(f"❌ Research failed: {e}")


async def example_curriculum_generation():
    """Example of curriculum generation"""
    print("📚 Example 2: Curriculum Generation")
    print("=" * 50)
    
    generator = create_curriculum_generator()
    
    domain = "Machine Learning Engineering"
    learning_goals = [
        "Understand ML model deployment at scale",
        "Learn MLOps best practices",
        "Master model monitoring and maintenance"
    ]
    
    try:
        topics = await generator.generate_initial_curriculum(
            domain=domain,
            learning_goals=learning_goals,
            breadth_topics=5
        )
        
        print(f"Generated curriculum for: {domain}")
        print(f"Number of topics: {len(topics)}")
        print("\nGenerated Topics:")
        print("-" * 30)
        
        for i, topic in enumerate(topics, 1):
            print(f"{i}. {topic.name}")
            print(f"   Description: {topic.description}")
            print(f"   Depth: {topic.depth} | Priority: {topic.priority}")
            print()
        
    except Exception as e:
        print(f"❌ Curriculum generation failed: {e}")


async def example_parallel_research():
    """Example of parallel research for multiple topics"""
    print("⚡ Example 3: Parallel Research")
    print("=" * 50)
    
    client = create_deep_research_client()
    
    queries = [
        "Fine-tuning large language models with LoRA",
        "Retrieval-Augmented Generation (RAG) architectures", 
        "AI agent memory systems and persistence"
    ]
    
    try:
        print(f"Starting parallel research for {len(queries)} topics...")
        
        results = await client.parallel_research(
            queries=queries,
            domain="AI/ML",
            max_concurrent=2  # Limit concurrent requests
        )
        
        print(f"Completed {len(results)} research tasks")
        print("\nResults Summary:")
        print("-" * 30)
        
        total_cost = 0
        for i, result in enumerate(results):
            print(f"{i+1}. Query: {queries[i][:50]}...")
            print(f"   Content length: {len(result.content)} chars")
            print(f"   Cost: ${result.cost_estimate:.4f}")
            print(f"   Model: {result.model}")
            total_cost += result.cost_estimate or 0
            print()
        
        print(f"Total estimated cost: ${total_cost:.4f}")
        
    except Exception as e:
        print(f"❌ Parallel research failed: {e}")


async def example_training_questions():
    """Example of generating training questions from research"""
    print("❓ Example 4: Training Question Generation")
    print("=" * 50)
    
    client = create_deep_research_client()
    
    # First research a topic
    topic_name = "Python asyncio programming"
    research_response = await client.research_topic(
        query=f"Comprehensive guide to {topic_name}",
        domain="Python Programming",
        depth="fast"  # Use faster model for this example
    )
    
    # Then generate training questions
    try:
        questions = await client.generate_training_questions(
            topic_content=research_response.content,
            topic_name=topic_name,
            num_questions=5
        )
        
        print(f"Generated {len(questions)} training questions for: {topic_name}")
        print("\nSample Questions:")
        print("-" * 30)
        
        for i, q in enumerate(questions[:3], 1):  # Show first 3 questions
            print(f"{i}. Question: {q.get('question', 'N/A')}")
            print(f"   Category: {q.get('category', 'N/A')}")
            print(f"   Difficulty: {q.get('difficulty', 'N/A')}")
            print(f"   Answer: {q.get('answer', 'N/A')[:100]}...")
            print()
        
    except Exception as e:
        print(f"❌ Question generation failed: {e}")


async def example_cost_tracking():
    """Example of cost tracking and monitoring"""
    print("💰 Example 5: Cost Tracking")
    print("=" * 50)
    
    client = create_deep_research_client()
    
    # Set up cost tracking
    total_budget = 5.00  # $5 budget for examples
    current_cost = 0.0
    
    queries = [
        "Brief overview of transformers in NLP",
        "Quick guide to Docker containerization",
        "Basics of REST API design"
    ]
    
    print(f"Budget: ${total_budget:.2f}")
    print("Executing research with cost tracking...")
    
    for i, query in enumerate(queries, 1):
        try:
            response = await client.research_topic(
                query=query,
                domain="Technology",
                depth="fast"  # Use faster/cheaper model
            )
            
            cost = response.cost_estimate or 0
            current_cost += cost
            
            print(f"{i}. Query: {query[:40]}...")
            print(f"   Cost: ${cost:.4f} | Running total: ${current_cost:.4f}")
            
            if current_cost > total_budget:
                print(f"⚠️  Budget exceeded! Stopping execution.")
                break
                
        except Exception as e:
            print(f"❌ Research {i} failed: {e}")
    
    print(f"\nFinal cost: ${current_cost:.4f} / ${total_budget:.2f}")
    print(f"Budget utilization: {(current_cost/total_budget)*100:.1f}%")


async def main():
    """Run all examples"""
    print("🚀 Deep Research API Examples")
    print("=" * 60)
    print()
    
    # Check if API key is configured
    if not settings.openai.api_key:
        print("❌ OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file")
        return
    
    examples = [
        example_basic_research,
        example_curriculum_generation,
        example_parallel_research,
        example_training_questions,
        example_cost_tracking
    ]
    
    for example in examples:
        try:
            await example()
            print()
        except KeyboardInterrupt:
            print("\n⚠️  Example interrupted by user")
            break
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print()
    
    print("✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main()) 