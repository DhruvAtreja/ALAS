#!/usr/bin/env python3
"""
Test script for model evaluation functionality
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.evaluator import create_model_evaluator
from src.core.training_data_generator import CurriculumTrainingData
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_training_data_from_file(filename: str = "training_data_python_programming_20250630_021709.json") -> CurriculumTrainingData:
    """Load training data from JSON file"""
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract training data
        training_data_dict = data["training_data"]
        
        # Convert back to CurriculumTrainingData object
        training_data = CurriculumTrainingData.model_validate(training_data_dict)
        
        logger.info(f"✅ Loaded training data with {training_data.total_questions} questions from {filename}")
        return training_data
        
    except FileNotFoundError:
        logger.error(f"❌ Training data file '{filename}' not found")
        logger.info("Please run training data generation first or provide a valid file path")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        raise


async def test_single_question_evaluation():
    """Test evaluating a single question"""
    
    print("\n🧪 Testing Single Question Evaluation...")
    print("=" * 60)
    
    try:
        # Load training data
        training_data = load_training_data_from_file()
        
        # Get first question for testing
        first_topic = training_data.topics[0]
        first_question = first_topic.questions[0]
        
        print(f"Testing question: {first_question.question[:100]}...")
        print(f"From topic: {first_topic.topic_name}")
        
        # Create evaluator
        evaluator = create_model_evaluator(
            model_to_test="gpt-4.1-2025-04-14",
            max_concurrent=1
        )
        
        # Get model answer
        model_answer = await evaluator.get_model_answer(
            first_question.question, 
            first_topic.topic_name
        )
        
        print(f"\n📝 Model Answer ({len(model_answer)} chars):")
        print("-" * 40)
        print(model_answer[:300] + "..." if len(model_answer) > 300 else model_answer)
        print("-" * 40)
        
        print(f"\n🎯 Ideal Answer ({len(first_question.answer)} chars):")
        print("-" * 40)
        print(first_question.answer[:300] + "..." if len(first_question.answer) > 300 else first_question.answer)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        logger.error(f"Single question evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_topic_evaluation():
    """Test evaluating all questions for a single topic"""
    
    print("\n📚 Testing Topic Evaluation...")
    print("=" * 60)
    
    try:
        # Load training data
        training_data = load_training_data_from_file()
        
        # Use first topic for testing
        test_topic = training_data.topics[0]
        
        print(f"Testing topic: {test_topic.topic_name}")
        print(f"Questions to evaluate: {len(test_topic.questions)}")
        
        # Create evaluator
        evaluator = create_model_evaluator(
            model_to_test="gpt-4.1-2025-04-14",
            max_concurrent=2  # Limit for testing
        )
        
        # Get all model responses for this topic
        model_responses = []
        for question in test_topic.questions:
            try:
                model_answer = await evaluator.get_model_answer(
                    question.question, 
                    test_topic.topic_name
                )
                
                from src.core.evaluator import ModelResponse
                response = ModelResponse(
                    question_id=question.id,
                    question=question.question,
                    model_answer=model_answer,
                    ideal_answer=question.answer,
                    topic_id=question.topic_id,
                    topic_name=test_topic.topic_name,
                    category=question.category,
                    difficulty=question.difficulty
                )
                model_responses.append(response)
                
                print(f"  ✅ Got response for: {question.question[:50]}...")
                
            except Exception as e:
                print(f"  ❌ Failed for question: {question.question[:50]}... - {e}")
                continue
        
        print(f"\n📊 Successfully got {len(model_responses)} model responses")
        
        # Evaluate the topic
        print("🔍 Running evaluation with Deep Research API...")
        topic_results = await evaluator.evaluate_topic_responses(
            test_topic.topic_id,
            model_responses
        )
        
        print(f"\n🎉 Topic Evaluation Results:")
        print(f"  Topic: {topic_results.topic_name}")
        print(f"  Accuracy: {topic_results.accuracy:.1%} ({topic_results.correct_answers}/{topic_results.total_questions})")
        print(f"  Correct answers: {topic_results.correct_answers}")
        print(f"  Incorrect answers: {topic_results.incorrect_answers}")
        
        # Show some sample results
        print(f"\n📋 Sample Evaluation Results:")
        for i, result in enumerate(topic_results.results[:3], 1):
            status = "✅ CORRECT" if result.is_correct else "❌ INCORRECT"
            print(f"{i}. {status}")
            print(f"   Question: {result.question[:80]}...")
            print(f"   Explanation: {result.explanation[:100]}...")
            print()
        
        return True, topic_results
        
    except Exception as e:
        logger.error(f"Topic evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_full_evaluation():
    """Test full evaluation pipeline"""
    
    print("\n🚀 Testing Full Evaluation Pipeline...")
    print("=" * 60)
    
    try:
        # Load training data
        training_data = load_training_data_from_file()
        
        # Create limited training data for testing (only use first topic)
        from src.core.training_data_generator import CurriculumTrainingData
        test_training_data = CurriculumTrainingData(
            domain=training_data.domain,
            curriculum_metadata=training_data.curriculum_metadata,
            topics=training_data.topics[:1],  # Only first topic
            total_questions=len(training_data.topics[0].questions),
            generation_summary=training_data.generation_summary
        )
        
        print(f"Testing on limited dataset:")
        print(f"  Domain: {test_training_data.domain}")
        print(f"  Topics: {len(test_training_data.topics)}")
        print(f"  Questions: {test_training_data.total_questions}")
        
        # Create evaluator
        evaluator = create_model_evaluator(
            model_to_test="gpt-4.1-2025-04-14",
            max_concurrent=2
        )
        
        # Run full evaluation
        evaluation_summary = await evaluator.evaluate_training_data(test_training_data)
        
        print(f"\n🎯 Full Evaluation Results:")
        print(f"  Model tested: {evaluation_summary.model_tested}")
        print(f"  Overall accuracy: {evaluation_summary.overall_accuracy:.1%}")
        print(f"  Topics evaluated: {evaluation_summary.total_topics}")
        print(f"  Total questions: {evaluation_summary.total_questions}")
        
        # Show category performance
        print(f"\n📊 Performance by Category:")
        for category, stats in evaluation_summary.category_performance.items():
            accuracy = stats['accuracy']
            correct = stats['correct']
            total = stats['total']
            print(f"  {category}: {accuracy:.1%} ({correct}/{total})")
        
        # Show difficulty performance
        print(f"\n📈 Performance by Difficulty:")
        for difficulty, stats in evaluation_summary.difficulty_performance.items():
            accuracy = stats['accuracy']
            correct = stats['correct']
            total = stats['total']
            print(f"  {difficulty}: {accuracy:.1%} ({correct}/{total})")
        
        # Save results
        results_file = evaluator.save_evaluation_results(evaluation_summary, "test_evaluation_results.json")
        print(f"\n💾 Results saved to: {results_file}")
        
        return True, evaluation_summary
        
    except Exception as e:
        logger.error(f"Full evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def analyze_evaluation_results(filename: str = "test_evaluation_results.json"):
    """Analyze saved evaluation results"""
    
    print(f"\n📊 Analyzing Evaluation Results from {filename}...")
    print("=" * 60)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        eval_data = data["evaluation_results"]
        
        print(f"Model: {eval_data['model_tested']}")
        print(f"Domain: {eval_data['domain']}")
        print(f"Overall accuracy: {eval_data['overall_accuracy']:.1%}")
        print(f"Total questions: {eval_data['total_questions']}")
        
        # Analyze topic performance
        print(f"\n📚 Topic Performance:")
        for topic in eval_data['topic_results']:
            print(f"  {topic['topic_name']}: {topic['accuracy']:.1%} ({topic['correct_answers']}/{topic['total_questions']})")
        
        # Show some correct/incorrect examples
        print(f"\n✅ Sample Correct Answers:")
        correct_count = 0
        for topic in eval_data['topic_results']:
            for result in topic['results']:
                if result['is_correct'] and correct_count < 2:
                    print(f"  Q: {result['question'][:60]}...")
                    print(f"  A: {result['model_answer'][:100]}...")
                    print(f"  Why: {result['explanation'][:80]}...")
                    print()
                    correct_count += 1
        
        print(f"\n❌ Sample Incorrect Answers:")
        incorrect_count = 0
        for topic in eval_data['topic_results']:
            for result in topic['results']:
                if not result['is_correct'] and incorrect_count < 2:
                    print(f"  Q: {result['question'][:60]}...")
                    print(f"  A: {result['model_answer'][:100]}...")
                    print(f"  Why: {result['explanation'][:80]}...")
                    print()
                    incorrect_count += 1
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


async def main():
    """Run all evaluation tests"""
    
    print("🚀 Model Evaluation Test Suite")
    print("=" * 80)
    
    if not settings.openai.api_key:
        print("❌ OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    
    # Check if training data file exists
    training_data_file = "training_data_python_programming_20250630_021709.json"
    if not Path(training_data_file).exists():
        print("❌ No training data file found!")
        print(f"Please run training data generation first to create {training_data_file}")
        print("Or update the filename in the script to match your training data file")
        return False
    
    results = []
    
    # Test 1: Single question evaluation
    print("\n" + "="*80)
    single_success = await test_single_question_evaluation()
    results.append(("Single Question Evaluation", single_success))
    
    # Test 2: Topic evaluation
    print("\n" + "="*80)
    topic_success, _ = await test_topic_evaluation()
    results.append(("Topic Evaluation", topic_success))
    
    # Test 3: Full evaluation pipeline
    print("\n" + "="*80)
    full_success, _ = await test_full_evaluation()
    results.append(("Full Evaluation Pipeline", full_success))
    
    # Test 4: Results analysis
    print("\n" + "="*80)
    if full_success:
        analysis_success = analyze_evaluation_results()
        results.append(("Results Analysis", analysis_success))
    
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
        print("\n🎉 All tests passed! Model evaluation is working correctly.")
        print("\nNext steps:")
        print("- Try evaluating different models (gpt-4o, gpt-4o-mini, etc.)")
        print("- Run evaluation on larger training datasets")
        print("- Compare performance across different topics and categories")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 