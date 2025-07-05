#!/usr/bin/env python3
"""
Test script for DPO improvement functionality

This script demonstrates how to use the DPO improvement engine to improve
a model based on evaluation results.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.dpo_improvement import create_dpo_improvement_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_dpo_improvement():
    """Test DPO improvement functionality"""
    
    print("🧪 Testing DPO Improvement Functionality")
    print("=" * 60)
    
    # Input files (using existing evaluation results)
    evaluation_file = "evaluation_results_finetuned_test.json"
    training_data_file = "training_data_python_programming_20250630_021709.json"
    
    print(f"📁 Input files:")
    print(f"   Evaluation: {evaluation_file}")
    print(f"   Training data: {training_data_file}")
    print("")
    
    # Check if files exist
    if not Path(evaluation_file).exists():
        print(f"❌ Evaluation file not found: {evaluation_file}")
        return False
    
    if not Path(training_data_file).exists():
        print(f"❌ Training data file not found: {training_data_file}")
        return False
    
    try:
        # Load evaluation results to check wrong answers
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        
        print("📊 Current Evaluation Results:")
        original_accuracy = eval_data["file_metadata"]["overall_accuracy"]
        model_tested = eval_data["file_metadata"]["model_tested"]
        total_questions = eval_data["file_metadata"]["total_questions"]
        
        print(f"   Model: {model_tested}")
        print(f"   Total questions: {total_questions}")
        print(f"   Current accuracy: {original_accuracy:.2%}")
        
        # Count wrong answers
        wrong_count = 0
        topic_results = eval_data.get("evaluation_results", {}).get("topic_results", [])
        
        for topic_result in topic_results:
            results = topic_result.get("results", [])
            for result in results:
                if not result.get("is_correct", True):
                    wrong_count += 1
        
        print(f"   Wrong answers: {wrong_count}")
        print("")
        
        if wrong_count == 0:
            print("🎉 No wrong answers found! Model is already perfect.")
            return True
        
        print(f"🎯 Found {wrong_count} wrong answers that can be improved with DPO")
        print("")
        
        # Preview some wrong answers
        print("❌ Sample Wrong Answers:")
        print("-" * 40)
        count = 0
        for topic_result in topic_results:
            results = topic_result.get("results", [])
            for result in results:
                if not result.get("is_correct", True) and count < 2:  # Show first 2
                    count += 1
                    print(f"{count}. Question: {result.get('question', '')[:80]}...")
                    print(f"   Model answer: {result.get('model_answer', '')[:80]}...")
                    print(f"   Correct answer: {result.get('ideal_answer', '')[:80]}...")
                    print("")
        
        print("✅ DPO improvement functionality is ready to use!")
        print("")
        print("🚀 To run DPO improvement, use:")
        print(f"   python run_dpo_improvement.py \\")
        print(f"       --evaluation-file {evaluation_file} \\")
        print(f"       --training-data-file {training_data_file}")
        print("")
        print("👀 To preview only (without running DPO):")
        print(f"   python run_dpo_improvement.py \\")
        print(f"       --evaluation-file {evaluation_file} \\")
        print(f"       --training-data-file {training_data_file} \\")
        print(f"       --preview-only")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing DPO improvement: {e}")
        print(f"❌ Error: {e}")
        return False


async def main():
    """Main function"""
    success = await test_dpo_improvement()
    return 0 if success else 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result) 