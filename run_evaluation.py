#!/usr/bin/env python3
"""
Standalone script to evaluate model performance on training data
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.evaluator import create_model_evaluator
from src.core.training_data_generator import CurriculumTrainingData
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def evaluate_model_on_training_data(
    training_data_file: str,
    model_to_test: str = "gpt-4.1-2025-04-14",
    output_file: Optional[str] = None,
    max_concurrent: int = 3,
    max_topics: Optional[int] = None
):
    """
    Evaluate a model's performance on training data
    
    Args:
        training_data_file: Path to training data JSON file
        model_to_test: Model to evaluate (e.g., gpt-4.1-2025-04-14)
        output_file: Output filename (auto-generated if None)
        max_concurrent: Maximum concurrent API requests
        max_topics: Maximum number of topics to evaluate (None for all)
    """
    
    try:
        # Load training data
        logger.info(f"Loading training data from {training_data_file}")
        with open(training_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        training_data_dict = data.get("training_data", data)
        training_data = CurriculumTrainingData.model_validate(training_data_dict)
        
        # Limit topics if specified
        if max_topics and max_topics < len(training_data.topics):
            logger.info(f"Limiting evaluation to first {max_topics} topics")
            limited_topics = training_data.topics[:max_topics]
            limited_questions = sum(len(topic.questions) for topic in limited_topics)
            
            training_data = CurriculumTrainingData(
                domain=training_data.domain,
                curriculum_metadata=training_data.curriculum_metadata,
                topics=limited_topics,
                total_questions=limited_questions,
                generation_summary=training_data.generation_summary
            )
        
        logger.info(f"✅ Loaded training data: {training_data.domain}")
        logger.info(f"  - Topics to evaluate: {len(training_data.topics)}")
        logger.info(f"  - Total questions: {training_data.total_questions}")
        logger.info(f"  - Model to test: {model_to_test}")
        logger.info(f"  - Max concurrent requests: {max_concurrent}")
        
        # Create evaluator
        evaluator = create_model_evaluator(
            model_to_test=model_to_test,
            max_concurrent=max_concurrent
        )
        
        # Run evaluation
        logger.info("🚀 Starting model evaluation...")
        evaluation_summary = await evaluator.evaluate_training_data(training_data)
        
        # Save results
        results_file = evaluator.save_evaluation_results(evaluation_summary, output_file)
        logger.info(f"📄 Results saved to: {results_file}")
        
        # Print summary
        logger.info(f"\n🎉 Evaluation completed!")
        logger.info(f"  Model: {evaluation_summary.model_tested}")
        logger.info(f"  Domain: {evaluation_summary.domain}")
        logger.info(f"  Overall accuracy: {evaluation_summary.overall_accuracy:.1%}")
        logger.info(f"  Questions evaluated: {evaluation_summary.total_questions}")
        logger.info(f"  Duration: {evaluation_summary.evaluation_metadata['total_duration_seconds']:.2f}s")
        
        # Show category performance
        if evaluation_summary.category_performance:
            logger.info(f"\n📊 Performance by Category:")
            for category, stats in evaluation_summary.category_performance.items():
                accuracy = stats['accuracy']
                correct = stats['correct']
                total = stats['total']
                logger.info(f"  {category}: {accuracy:.1%} ({correct}/{total})")
        
        # Show difficulty performance
        if evaluation_summary.difficulty_performance:
            logger.info(f"\n📈 Performance by Difficulty:")
            for difficulty, stats in evaluation_summary.difficulty_performance.items():
                accuracy = stats['accuracy']
                correct = stats['correct']
                total = stats['total']
                logger.info(f"  {difficulty}: {accuracy:.1%} ({correct}/{total})")
        
        # Show topic performance
        logger.info(f"\n📚 Performance by Topic:")
        for topic_result in evaluation_summary.topic_results:
            accuracy = topic_result.accuracy
            correct = topic_result.correct_answers
            total = topic_result.total_questions
            logger.info(f"  {topic_result.topic_name}: {accuracy:.1%} ({correct}/{total})")
        
        return results_file
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        raise


def main():
    """CLI interface for model evaluation"""
    
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py training_data.json
  python run_evaluation.py training_data.json --model gpt-4.1-2025-04-14
  python run_evaluation.py training_data.json --concurrent 2 --max-topics 2
  python run_evaluation.py training_data.json --output my_eval_results.json
        """
    )
    
    parser.add_argument(
        "training_data_file",
        help="Path to training data JSON file"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1-2025-04-14",
        help="Model to evaluate (default: gpt-4.1-2025-04-14)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output filename (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=3,
        help="Maximum concurrent API requests (default: 3)"
    )
    
    parser.add_argument(
        "--max-topics",
        type=int,
        help="Maximum number of topics to evaluate (default: all topics)"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: evaluate only first topic with limited concurrency"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.training_data_file).exists():
        print(f"❌ Training data file not found: {args.training_data_file}")
        sys.exit(1)
    
    if not settings.openai.api_key:
        print("❌ OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    
    # Adjust settings for test mode
    if args.test_mode:
        print("🧪 Running in test mode (first topic only, limited concurrency)")
        args.max_topics = 1
        args.concurrent = 2
    
    # Run evaluation
    try:
        result = asyncio.run(evaluate_model_on_training_data(
            training_data_file=args.training_data_file,
            model_to_test=args.model,
            output_file=args.output,
            max_concurrent=args.concurrent,
            max_topics=args.max_topics
        ))
        
        print(f"\n✅ Success! Evaluation results saved to: {result}")
        
        # Provide next steps
        print(f"\nNext steps:")
        print(f"- Review detailed results in {result}")
        print(f"- Compare with other models using --model parameter")
        print(f"- Analyze incorrect answers to identify improvement areas")
        print(f"- Use results for curriculum refinement or fine-tuning decisions")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 