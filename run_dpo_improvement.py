#!/usr/bin/env python3
"""
DPO Improvement Script

This script processes evaluation results, creates DPO training data for incorrect answers,
performs Direct Preference Optimization fine-tuning, and re-evaluates the improved model.

Usage:
    python run_dpo_improvement.py --evaluation-file evaluation_results_finetuned_test.json \
                                  --training-data-file training_data_python_programming_20250630_021709.json \
                                  --domain "Python Programming"
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.dpo_improvement import create_dpo_improvement_engine, DPOImprovementResult
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def print_banner():
    """Print script banner"""
    print("=" * 80)
    print("🔧 DPO Model Improvement Script")
    print("=" * 80)
    print("This script will:")
    print("1. 📊 Analyze evaluation results to find wrong answers")
    print("2. 🏗️  Create DPO training data from incorrect responses")
    print("3. 🎯 Perform Direct Preference Optimization fine-tuning")
    print("4. 📈 Re-evaluate the improved model")
    print("5. 📋 Generate improvement summary")
    print("")


def validate_files(evaluation_file: str, training_data_file: str) -> tuple[bool, str]:
    """Validate that input files exist and are in correct format"""
    
    # Check evaluation file
    eval_path = Path(evaluation_file)
    if not eval_path.exists():
        return False, f"Evaluation file not found: {evaluation_file}"
    
    # Check training data file
    training_path = Path(training_data_file)
    if not training_path.exists():
        return False, f"Training data file not found: {training_data_file}"
    
    # Validate evaluation file format
    try:
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        
        # Check required fields
        if "evaluation_results" not in eval_data:
            return False, f"Invalid evaluation file format: missing 'evaluation_results'"
        
        if "file_metadata" not in eval_data:
            return False, f"Invalid evaluation file format: missing 'file_metadata'"
        
        if "model_tested" not in eval_data["file_metadata"]:
            return False, f"Invalid evaluation file format: missing 'model_tested' in metadata"
        
        print(f"✅ Evaluation file validated: {evaluation_file}")
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in evaluation file: {e}"
    except Exception as e:
        return False, f"Error reading evaluation file: {e}"
    
    # Validate training data file format
    try:
        with open(training_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Check required fields
        if "training_data" not in training_data:
            return False, f"Invalid training data file format: missing 'training_data'"
        
        print(f"✅ Training data file validated: {training_data_file}")
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in training data file: {e}"
    except Exception as e:
        return False, f"Error reading training data file: {e}"
    
    return True, "All files validated successfully"


def preview_wrong_answers(evaluation_file: str) -> bool:
    """Preview the wrong answers that will be used for DPO training"""
    
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    print("📋 Preview of Wrong Answers:")
    print("-" * 50)
    
    total_questions = 0
    wrong_answers = 0
    
    topic_results = eval_data.get("evaluation_results", {}).get("topic_results", [])
    
    for topic_result in topic_results:
        results = topic_result.get("results", [])
        total_questions += len(results)
        
        for result in results:
            if not result.get("is_correct", True):
                wrong_answers += 1
                print(f"❌ Question {result.get('question_id', 'unknown')}")
                print(f"   Question: {result.get('question', '')[:100]}...")
                print(f"   Category: {result.get('category', 'unknown')}")
                print(f"   Difficulty: {result.get('difficulty', 'unknown')}")
                print("")
    
    print(f"📊 Summary:")
    print(f"   Total questions: {total_questions}")
    print(f"   Wrong answers: {wrong_answers}")
    print(f"   Accuracy: {((total_questions - wrong_answers) / total_questions * 100) if total_questions > 0 else 0:.1f}%")
    print("")
    
    if wrong_answers == 0:
        print("🎉 No wrong answers found! The model is already perfect.")
        return False
    
    return True


def print_improvement_results(result: DPOImprovementResult) -> None:
    """Print formatted improvement results"""
    
    print("🎉 DPO Improvement Completed!")
    print("=" * 80)
    
    print(f"🤖 Models:")
    print(f"   Original: {result.original_model}")
    print(f"   Improved: {result.dpo_model}")
    print("")
    
    print(f"📈 Performance:")
    print(f"   Original accuracy: {result.original_accuracy:.2%}")
    print(f"   Improved accuracy: {result.improved_accuracy:.2%}")
    
    if result.improvement > 0:
        print(f"   ✅ Improvement: +{result.improvement:.2%}")
    elif result.improvement < 0:
        print(f"   ⚠️  Regression: {result.improvement:.2%}")
    else:
        print(f"   ➡️  No change: {result.improvement:.2%}")
    print("")
    
    print(f"🔧 Training Details:")
    print(f"   DPO examples used: {result.dpo_examples_count}")
    print(f"   Processing time: {result.processing_time:.1f} seconds")
    print("")
    
    print(f"📁 Generated Files:")
    print(f"   DPO training data: {result.dpo_training_file}")
    print(f"   Original evaluation: {result.original_evaluation_file}")
    print(f"   Improved evaluation: {result.improved_evaluation_file}")
    print("")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Improve model using DPO on wrong answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (domain extracted automatically)
    python run_dpo_improvement.py \\
        --evaluation-file evaluation_results_finetuned_test.json \\
        --training-data-file training_data_python_programming_20250630_021709.json
    
    # With custom output directory
    python run_dpo_improvement.py \\
        --evaluation-file evaluation_results_finetuned_test.json \\
        --training-data-file training_data_python_programming_20250630_021709.json \\
        --output-dir data/dpo_results
    
    # Preview mode only
    python run_dpo_improvement.py \\
        --evaluation-file evaluation_results_finetuned_test.json \\
        --training-data-file training_data_python_programming_20250630_021709.json \\
        --preview-only
        """
    )
    
    parser.add_argument(
        "--evaluation-file", 
        required=True, 
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--training-data-file", 
        required=True, 
        help="Path to original training data JSON file"
    )
    parser.add_argument(
        "--domain", 
        required=False, 
        help="Domain name (optional, will be extracted from evaluation results)"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/evaluations", 
        help="Output directory for results (default: data/evaluations)"
    )
    parser.add_argument(
        "--preview-only", 
        action="store_true", 
        help="Only preview wrong answers, don't run DPO training"
    )
    parser.add_argument(
        "--no-summary", 
        action="store_true", 
        help="Don't save improvement summary file"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    try:
        # Validate input files
        print("🔍 Validating input files...")
        is_valid, message = validate_files(args.evaluation_file, args.training_data_file)
        if not is_valid:
            print(f"❌ Validation failed: {message}")
            return 1
        
        print(f"✅ {message}")
        print("")
        
        # Preview wrong answers
        has_wrong_answers = preview_wrong_answers(args.evaluation_file)
        
        if not has_wrong_answers:
            print("🎯 No improvements needed!")
            return 0
        
        if args.preview_only:
            print("👀 Preview mode - exiting without running DPO training")
            return 0
        
        # Confirm with user
        response = input("🚀 Proceed with DPO improvement? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ DPO improvement cancelled by user")
            return 0
        
        print("")
        print("🔧 Starting DPO improvement workflow...")
        print("-" * 50)
        
        # Create DPO improvement engine
        engine = create_dpo_improvement_engine()
        
        # Run improvement workflow
        result = await engine.improve_model_with_dpo(
            evaluation_results_file=args.evaluation_file,
            training_data_file=args.training_data_file,
            output_dir=args.output_dir
        )
        
        # Print results
        print_improvement_results(result)
        
        # Save improvement summary
        if not args.no_summary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = f"dpo_improvement_summary_{timestamp}.json"
            await engine.save_improvement_summary(result, summary_file)
            print(f"📋 Improvement summary saved to: {summary_file}")
        
        print("🏁 DPO improvement workflow completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ DPO improvement interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"DPO improvement failed: {e}")
        print(f"❌ DPO improvement failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result) 