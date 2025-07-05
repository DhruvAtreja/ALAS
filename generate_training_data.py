#!/usr/bin/env python3
"""
Standalone script to generate training data from curriculum
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.training_data_generator import create_training_data_generator
from src.core.deep_research_client import Curriculum
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def generate_training_data_from_curriculum(
    curriculum_file: str,
    output_file: Optional[str] = None,
    questions_per_topic: int = 10,
    max_concurrent: int = 3,
    export_openai: bool = True
):
    """
    Generate training data from a curriculum file
    
    Args:
        curriculum_file: Path to curriculum JSON file
        output_file: Output filename (auto-generated if None)
        questions_per_topic: Number of questions to generate per topic
        max_concurrent: Maximum concurrent API requests
        export_openai: Whether to export OpenAI fine-tuning format
    """
    
    try:
        # Load curriculum
        logger.info(f"Loading curriculum from {curriculum_file}")
        with open(curriculum_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        curriculum_data = data.get("curriculum", data)
        curriculum = Curriculum.model_validate(curriculum_data)
        
        logger.info(f"✅ Loaded curriculum: {curriculum.domain}")
        logger.info(f"  - Topics: {len(curriculum.topics)}")
        logger.info(f"  - Questions per topic: {questions_per_topic}")
        logger.info(f"  - Max concurrent requests: {max_concurrent}")
        logger.info(f"  - Total questions to generate: {len(curriculum.topics) * questions_per_topic}")
        
        # Create training data generator
        generator = create_training_data_generator(
            max_concurrent=max_concurrent,
            questions_per_topic=questions_per_topic
        )
        
        # Generate training data
        logger.info("🚀 Starting training data generation...")
        training_data = await generator.generate_curriculum_training_data(curriculum)
        print("training_data")
        print(training_data)
        
        # Save results
        json_filename = generator.save_training_data(training_data, output_file)
        print("json_filename")
        print(json_filename)
        
        if export_openai:
            openai_filename = json_filename.replace('.json', '_openai.jsonl')
            generator.export_for_openai_finetuning(training_data, openai_filename)
            logger.info(f"📄 OpenAI format exported to: {openai_filename}")
        
        # Print summary
        summary = training_data.generation_summary
        logger.info(f"\n🎉 Training data generation completed!")
        logger.info(f"  ✅ Successful topics: {summary['successful_topics']}")
        logger.info(f"  ❌ Failed topics: {summary['failed_topics']}")
        logger.info(f"  📊 Total questions: {training_data.total_questions}")
        logger.info(f"  ⏱️  Total time: {summary['total_duration_seconds']:.2f}s")
        logger.info(f"  ⚡ Generation rate: {summary['questions_per_minute']:.1f} questions/minute")
        
        if summary['failed_topics'] > 0:
            logger.warning(f"⚠️  {summary['failed_topics']} topics failed to generate:")
            for failed in summary['failed_topic_details']:
                logger.warning(f"    - {failed['topic_name']}: {failed['error']}")
        
        return json_filename
        
    except Exception as e:
        logger.error(f"❌ Training data generation failed: {e}")
        raise


def main():
    """CLI interface for training data generation"""
    
    parser = argparse.ArgumentParser(
        description="Generate training data from curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_training_data.py curriculum_test_results.json
  python generate_training_data.py curriculum.json --questions 30 --concurrent 2
  python generate_training_data.py curriculum.json --output my_training_data.json
        """
    )
    
    parser.add_argument(
        "curriculum_file",
        help="Path to curriculum JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output filename (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=50,
        help="Number of questions to generate per topic (default: 50)"
    )
    
    parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=3,
        help="Maximum concurrent API requests (default: 3)"
    )
    
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Skip exporting OpenAI fine-tuning format"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: only process first 3 topics with 10 questions each"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.curriculum_file).exists():
        print(f"❌ Curriculum file not found: {args.curriculum_file}")
        sys.exit(1)
    
    if not settings.openai.api_key:
        print("❌ OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    
    # Adjust settings for test mode
    if args.test_mode:
        print("🧪 Running in test mode (first 3 topics, 10 questions each)")
        
        # Load and limit curriculum for testing
        with open(args.curriculum_file, 'r') as f:
            data = json.load(f)
        
        curriculum_data = data.get("curriculum", data)
        curriculum_data["topics"] = curriculum_data["topics"][:1]
        
        # Save test curriculum
        test_file = "test_curriculum.json"
        with open(test_file, 'w') as f:
            json.dump({"curriculum": curriculum_data}, f, indent=2)
        
        args.curriculum_file = test_file
        args.questions = 10
        args.concurrent = 2
    
    # Run generation
    try:
        result = asyncio.run(generate_training_data_from_curriculum(
            curriculum_file=args.curriculum_file,
            output_file=args.output,
            questions_per_topic=args.questions,
            max_concurrent=args.concurrent,
            export_openai=not args.no_openai
        ))
        
        print(f"\n✅ Success! Training data saved to: {result}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 