#!/usr/bin/env python3
"""
Fine-tuning script for OpenAI GPT models using training data
"""

import asyncio
import argparse
import sys
import json
from typing import Optional
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.fine_tuner import create_fine_tuner, FineTuningHyperparameters, FineTuningError
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def auto_run_evaluation(fine_tuned_model: str, training_file: str) -> bool:
    """Automatically run evaluation on the fine-tuned model"""
    try:
        # Convert JSONL training file to JSON evaluation file
        training_path = Path(training_file)
        json_file = training_path.with_suffix('.json')
        
        # Check if corresponding JSON file exists
        if not json_file.exists():
            # Look for any JSON file with similar name
            pattern = training_path.stem.replace('_openai', '')
            json_files = list(training_path.parent.glob(f"{pattern}*.json"))
            
            if json_files:
                json_file = json_files[0]
            else:
                print(f"\n⚠️  No evaluation data found for {training_file}")
                print("To run evaluation manually:")
                print(f"python run_evaluation.py <evaluation_data.json> --model {fine_tuned_model}")
                return False
        
        print(f"\n🧪 Running automatic evaluation...")
        print(f"📁 Evaluation data: {json_file}")
        print(f"🤖 Model: {fine_tuned_model}")
        
        # Import and run evaluation
        import subprocess
        import os
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_finetuned_{timestamp}.json"
        
        # Run evaluation as subprocess
        cmd = [
            sys.executable, "run_evaluation.py",
            str(json_file),
            "--model", fine_tuned_model,
            "--output", output_file,
            "--concurrent", "2"  # Conservative concurrency
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Evaluation completed successfully!")
            print(f"📊 Results saved to: {output_file}")
            
            # Try to show quick summary
            try:
                with open(output_file, 'r') as f:
                    eval_results = json.load(f)
                    
                if 'evaluation_summary' in eval_results:
                    summary = eval_results['evaluation_summary']
                    total = summary.get('total_questions', 0)
                    correct = summary.get('total_correct', 0)
                    accuracy = (correct / total * 100) if total > 0 else 0
                    
                    print(f"🎯 Quick Results: {correct}/{total} correct ({accuracy:.1f}%)")
                    
            except Exception as e:
                logger.debug(f"Could not parse evaluation results: {e}")
            
            print(f"\n💡 Compare with base model:")
            print(f"python compare_model_performance.py <base_results.json> {output_file}")
            
            return True
        else:
            print(f"❌ Evaluation failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Auto-evaluation error: {e}")
        print(f"💡 Run evaluation manually:")
        print(f"python run_evaluation.py <evaluation_data.json> --model {fine_tuned_model}")
        return False


def validate_training_file(file_path: str) -> bool:
    """Validate training file format"""
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Training file not found: {file_path}")
            return False
        
        if path.suffix.lower() != '.jsonl':
            logger.error(f"Training file must be JSONL format: {file_path}")
            return False
        
        # Quick validation of first few lines
        with open(path, 'r') as f:
            lines_checked = 0
            for line in f:
                if lines_checked >= 3:  # Check first 3 lines
                    break
                try:
                    data = json.loads(line.strip())
                    if "messages" not in data:
                        logger.error(f"Invalid format: line {lines_checked + 1} missing 'messages' field")
                        return False
                    lines_checked += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {lines_checked + 1}: {e}")
                    return False
        
        logger.info(f"Training file validation passed: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating training file: {e}")
        return False


async def run_fine_tuning(
    training_file: str,
    model: str = "gpt-4.1-2025-04-14",
    validation_file: Optional[str] = None,
    hyperparameters: Optional[FineTuningHyperparameters] = None,
    suffix: Optional[str] = None,
    wait_for_completion: bool = True,
    poll_interval: int = 60
) -> bool:
    """
    Run fine-tuning process
    
    Args:
        training_file: Path to training JSONL file
        model: Base model to fine-tune
        validation_file: Optional validation file
        hyperparameters: Training hyperparameters
        suffix: Optional model suffix
        wait_for_completion: Whether to wait for completion
        poll_interval: Polling interval in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate input files
        if not validate_training_file(training_file):
            return False
        
        if validation_file and not validate_training_file(validation_file):
            return False
        
        # Create fine-tuner
        fine_tuner = create_fine_tuner()
        
        logger.info("🚀 Starting fine-tuning process...")
        logger.info(f"📁 Training file: {training_file}")
        logger.info(f"🤖 Base model: {model}")
        if validation_file:
            logger.info(f"📋 Validation file: {validation_file}")
        if suffix:
            logger.info(f"🏷️  Model suffix: {suffix}")
        
        # Run fine-tuning
        result = await fine_tuner.fine_tune_from_file(
            training_file_path=training_file,
            model=model,
            validation_file_path=validation_file,
            hyperparameters=hyperparameters,
            suffix=suffix,
            wait_for_completion=wait_for_completion,
            poll_interval=poll_interval
        )
        
        logger.info("✅ Fine-tuning process completed!")
        logger.info(f"📊 Job ID: {result.job_id}")
        logger.info(f"📈 Status: {result.status}")
        
        if result.fine_tuned_model:
            logger.info(f"🎯 Fine-tuned model: {result.fine_tuned_model}")
            print(f"\n🎉 Fine-tuning successful!")
            print(f"Fine-tuned model ID: {result.fine_tuned_model}")
            print(f"You can now use this model in your applications.")
            
            # Auto-run evaluation if training data exists
            await auto_run_evaluation(result.fine_tuned_model, training_file)
        else:
            logger.info(f"⏳ Job still in progress. Job ID: {result.job_id}")
            print(f"\n⏳ Fine-tuning job created: {result.job_id}")
            print(f"Status: {result.status}")
            print(f"Use 'python run_fine_tuning.py --status {result.job_id}' to check progress")
        
        if result.trained_tokens:
            logger.info(f"📊 Tokens trained: {result.trained_tokens:,}")
        
        return True
        
    except FineTuningError as e:
        logger.error(f"Fine-tuning error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


async def check_job_status(job_id: str) -> bool:
    """Check the status of a fine-tuning job"""
    try:
        fine_tuner = create_fine_tuner()
        
        logger.info(f"🔍 Checking status for job: {job_id}")
        
        result = fine_tuner.get_job_status(job_id)
        
        print(f"\n📊 Fine-tuning Job Status")
        print(f"Job ID: {result.job_id}")
        print(f"Model: {result.model}")
        print(f"Status: {result.status}")
        print(f"Created: {result.created_at}")
        
        if result.fine_tuned_model:
            print(f"Fine-tuned model: {result.fine_tuned_model}")
            
        if result.trained_tokens:
            print(f"Tokens trained: {result.trained_tokens:,}")
            
        if result.error:
            print(f"Error: {result.error}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking job status: {e}")
        return False


async def list_jobs(limit: int = 10) -> bool:
    """List recent fine-tuning jobs"""
    try:
        fine_tuner = create_fine_tuner()
        
        logger.info(f"📋 Listing {limit} recent fine-tuning jobs...")
        
        jobs = fine_tuner.list_fine_tuning_jobs(limit=limit)
        
        if not jobs:
            print("No fine-tuning jobs found.")
            return True
        
        print(f"\n📋 Recent Fine-tuning Jobs ({len(jobs)} found)")
        print("-" * 80)
        
        for job in jobs:
            print(f"Job ID: {job.job_id}")
            print(f"Model: {job.model}")
            print(f"Status: {job.status}")
            if job.fine_tuned_model:
                print(f"Fine-tuned model: {job.fine_tuned_model}")
            if job.trained_tokens:
                print(f"Tokens: {job.trained_tokens:,}")
            print("-" * 40)
        
        return True
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune OpenAI GPT models",
        epilog="""
Examples:
  python run_fine_tuning.py training_data.jsonl
  python run_fine_tuning.py training_data.jsonl --model gpt-4.1-mini-2025-04-14
  python run_fine_tuning.py training_data.jsonl --epochs 3 --suffix "python-expert"
  python run_fine_tuning.py --status ftjob-abc123  # Check job status
  python run_fine_tuning.py --list  # List recent jobs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main arguments
    parser.add_argument(
        "training_file", nargs="?",
        help="Path to training data JSONL file"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1-2025-04-14",
        choices=["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"],
        help="Base model to fine-tune (default: gpt-4.1-2025-04-14)"
    )
    
    parser.add_argument(
        "--validation-file", "-v",
        help="Path to validation data JSONL file"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of training epochs (1-50)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        help="Batch size (integer or 'auto')"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        help="Learning rate multiplier"
    )
    
    parser.add_argument(
        "--suffix", "-s",
        help="Suffix for the fine-tuned model name"
    )
    
    # Job management
    parser.add_argument(
        "--no-wait", "-n",
        action="store_true",
        help="Don't wait for job completion"
    )
    
    parser.add_argument(
        "--poll-interval", "-p",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--status",
        help="Check status of fine-tuning job by ID"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List recent fine-tuning jobs"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of jobs to list (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.status:
        success = asyncio.run(check_job_status(args.status))
        sys.exit(0 if success else 1)
    
    if args.list:
        success = asyncio.run(list_jobs(args.limit))
        sys.exit(0 if success else 1)
    
    # Main fine-tuning mode
    if not args.training_file:
        parser.error("Training file is required for fine-tuning")
    
    # Build hyperparameters
    hyperparameters = None
    if args.epochs or args.batch_size or args.learning_rate:
        hyperparameters = FineTuningHyperparameters()
        if args.epochs:
            hyperparameters.n_epochs = args.epochs
        if args.batch_size:
            hyperparameters.batch_size = args.batch_size
        if args.learning_rate:
            hyperparameters.learning_rate_multiplier = args.learning_rate
    
    # Run fine-tuning
    success = asyncio.run(run_fine_tuning(
        training_file=args.training_file,
        model=args.model,
        validation_file=args.validation_file,
        hyperparameters=hyperparameters,
        suffix=args.suffix,
        wait_for_completion=not args.no_wait,
        poll_interval=args.poll_interval
    ))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 