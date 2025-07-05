#!/usr/bin/env python3
"""
Test script for Training Data Generation functionality
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.training_data_generator import create_training_data_generator
from src.core.deep_research_client import Curriculum
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_curriculum_from_file(filename: str = "curriculum_test_results.json") -> Curriculum:
    """Load curriculum from previously generated JSON file"""
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract curriculum data
        curriculum_data = data["curriculum"]
        
        # Convert back to Curriculum object
        curriculum = Curriculum.model_validate(curriculum_data)
        
        logger.info(f"✅ Loaded curriculum with {len(curriculum.topics)} topics from {filename}")
        return curriculum
        
    except FileNotFoundError:
        logger.error(f"❌ Curriculum file '{filename}' not found")
        logger.info("Please run curriculum generation first or provide a valid file path")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load curriculum: {e}")
        raise


async def test_single_topic_generation():
    """Test generating questions for a single topic"""
    
    print("\n🧪 Testing Single Topic Question Generation...")
    print("=" * 60)
    
    try:
        # Load curriculum
        curriculum = load_curriculum_from_file()
        
        # Pick the first topic for testing
        test_topic = curriculum.topics[0]
        
        print(f"Testing with topic: {test_topic.name}")
        print(f"Topic difficulty: {test_topic.difficulty}")
        
        # Create generator with small number of questions for testing
        generator = create_training_data_generator(
            max_concurrent=1, 
            questions_per_topic=10  # Small number for quick testing
        )
        
        # Generate questions for single topic
        topic_data = await generator.generate_topic_questions(test_topic, curriculum.domain)
        
        print(f"✅ Generated {len(topic_data.questions)} questions")
        print(f"Generation time: {topic_data.generation_metadata['generation_duration']:.2f}s")
        
        # Show first few questions
        print("\nSample questions:")
        for i, question in enumerate(topic_data.questions[:3], 1):
            print(f"{i}. [{question.category}] {question.question[:100]}...")
            print(f"   Answer: {question.answer[:150]}...")
            print()
        
        return True, topic_data
        
    except Exception as e:
        logger.error(f"Single topic test failed: {e}")
        return False, None


async def test_curriculum_training_generation():
    """Test generating training data for entire curriculum"""
    
    print("\n📚 Testing Full Curriculum Training Data Generation...")
    print("=" * 60)
    
    try:
        # Load curriculum
        curriculum = load_curriculum_from_file()
        
        # Limit to first few topics for testing to avoid long wait
        test_curriculum = Curriculum(
            domain=curriculum.domain,
            topics=curriculum.topics[:3],  # Only first 3 topics for testing
            metadata=curriculum.metadata
        )
        
        print(f"Testing with {len(test_curriculum.topics)} topics:")
        for topic in test_curriculum.topics:
            print(f"  - {topic.name} ({topic.difficulty})")
        
        # Create generator with test settings
        generator = create_training_data_generator(
            max_concurrent=2,  # Limit concurrency for testing
            questions_per_topic=15  # Moderate number for testing
        )
        
        # Generate training data
        training_data = await generator.generate_curriculum_training_data(test_curriculum)
        
        print(f"\n✅ Training data generation completed!")
        print(f"Successfully processed: {training_data.generation_summary['successful_topics']} topics")
        print(f"Failed topics: {training_data.generation_summary['failed_topics']}")
        print(f"Total questions: {training_data.total_questions}")
        print(f"Total time: {training_data.generation_summary['total_duration_seconds']:.2f}s")
        print(f"Rate: {training_data.generation_summary['questions_per_minute']:.1f} questions/minute")
        
        # Save test results
        json_file = generator.save_training_data(training_data, "test_training_data.json")
        jsonl_file = generator.export_for_openai_finetuning(training_data, "test_openai_training.jsonl")
        
        print(f"\n💾 Files saved:")
        print(f"  - JSON format: {json_file}")
        print(f"  - OpenAI JSONL format: {jsonl_file}")
        
        return True, training_data
        
    except Exception as e:
        logger.error(f"Curriculum training generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_rate_limiting():
    """Test rate limiting with multiple concurrent requests"""
    
    print("\n⚡ Testing Rate Limiting...")
    print("=" * 60)
    
    try:
        # Load curriculum
        curriculum = load_curriculum_from_file()
        
        # Create multiple generators with different concurrency limits
        test_cases = [
            {"max_concurrent": 1, "name": "Sequential (1 concurrent)"},
            {"max_concurrent": 3, "name": "Moderate (3 concurrent)"},
        ]
        
        for test_case in test_cases:
            print(f"\nTesting {test_case['name']}...")
            
            # Use only 2 topics and 5 questions for quick testing
            test_curriculum = Curriculum(
                domain=curriculum.domain,
                topics=curriculum.topics[:2],
                metadata=curriculum.metadata
            )
            
            generator = create_training_data_generator(
                max_concurrent=test_case['max_concurrent'],
                questions_per_topic=5
            )
            
            start_time = asyncio.get_event_loop().time()
            training_data = await generator.generate_curriculum_training_data(test_curriculum)
            end_time = asyncio.get_event_loop().time()
            
            duration = end_time - start_time
            print(f"  Duration: {duration:.2f}s")
            print(f"  Questions generated: {training_data.total_questions}")
            print(f"  Rate: {training_data.total_questions / duration:.1f} questions/second")
        
        return True
        
    except Exception as e:
        logger.error(f"Rate limiting test failed: {e}")
        return False


def analyze_test_results(filename: str = "test_training_data.json"):
    """Analyze the generated training data"""
    
    print(f"\n📊 Analyzing Training Data from {filename}...")
    print("=" * 60)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        training_data = data["training_data"]
        
        print(f"Domain: {training_data['domain']}")
        print(f"Total topics: {len(training_data['topics'])}")
        print(f"Total questions: {training_data['total_questions']}")
        
        # Analyze by category
        category_counts = {}
        difficulty_counts = {}
        
        for topic in training_data['topics']:
            print(f"\nTopic: {topic['topic_name']}")
            print(f"  Questions generated: {len(topic['questions'])}")
            
            for question in topic['questions']:
                category = question['category']
                difficulty = question['difficulty']
                
                category_counts[category] = category_counts.get(category, 0) + 1
                difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        print(f"\n📈 Category Distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / training_data['total_questions']) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print(f"\n📊 Difficulty Distribution:")
        for difficulty, count in sorted(difficulty_counts.items()):
            percentage = (count / training_data['total_questions']) * 100
            print(f"  {difficulty}: {count} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


async def main():
    """Run all training data generation tests"""
    
    print("🚀 Training Data Generation Test Suite")
    print("=" * 80)
    
    if not settings.openai.api_key:
        print("❌ OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    
    # Check if curriculum file exists
    if not Path("curriculum_test_results.json").exists():
        print("❌ No curriculum file found!")
        print("Please run the curriculum generation test first to create curriculum_test_results.json")
        return False
    
    results = []
    
    # Test 1: Single topic generation
    print("\n" + "="*80)
    single_success, _ = await test_single_topic_generation()
    results.append(("Single Topic Generation", single_success))
    
    # Test 2: Full curriculum generation (limited)
    print("\n" + "="*80)
    curriculum_success, _ = await test_curriculum_training_generation()
    results.append(("Curriculum Training Generation", curriculum_success))
    
    # Test 3: Rate limiting
    print("\n" + "="*80)
    rate_limit_success = await test_rate_limiting()
    results.append(("Rate Limiting", rate_limit_success))
    
    # Test 4: Result analysis
    print("\n" + "="*80)
    if curriculum_success:
        analysis_success = analyze_test_results()
        results.append(("Result Analysis", analysis_success))
    
    # Summary
    print("\n" + "="*80)
    print("🏁 Test Results Summary")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Training data generation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 