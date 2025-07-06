"""
DPO Improvement Module

This module processes evaluation results, creates DPO training data for incorrect answers,
performs Direct Preference Optimization fine-tuning, and re-evaluates the improved model.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import uuid

from openai import OpenAI
from pydantic import BaseModel, Field
from langsmith import traceable

try:
    from ..config.settings import settings
    from ..utils.logger import get_logger, log_api_call, log_cost, log_error
    from ..utils.async_file_utils import async_read_json, async_write_json, async_write_text, async_mkdir
    from .fine_tuner import OpenAIFineTuner, FineTuningHyperparameters
    from .evaluator import ModelEvaluator
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import settings
    from utils.logger import get_logger, log_api_call, log_cost, log_error
    from utils.async_file_utils import async_read_json, async_write_json, async_write_text, async_mkdir
    from core.fine_tuner import OpenAIFineTuner, FineTuningHyperparameters
    from core.evaluator import ModelEvaluator

logger = get_logger(__name__)


class DPOTrainingExample(BaseModel):
    """A single DPO training example"""
    prompt: str = Field(..., description="The question/prompt")
    chosen: str = Field(..., description="The preferred/correct answer")
    rejected: str = Field(..., description="The rejected/incorrect answer")
    domain: str = Field(..., description="The domain/subject area for the question")
    
    def to_dpo_format(self) -> Dict[str, Any]:
        """Convert to OpenAI DPO format"""
        return {
            "input": {
                "messages": [
                    {"role": "user", "content": self.prompt}
                ]
            },
            "preferred_output": [
                {"role": "assistant", "content": self.chosen}
            ],
            "non_preferred_output": [
                {"role": "assistant", "content": self.rejected}
            ]
        }


class DPOImprovementResult(BaseModel):
    """Result of DPO improvement process"""
    original_model: str = Field(..., description="Original model ID")
    dpo_model: str = Field(..., description="DPO-improved model ID")
    original_accuracy: float = Field(..., description="Original model accuracy")
    improved_accuracy: float = Field(..., description="DPO model accuracy")
    improvement: float = Field(..., description="Accuracy improvement")
    dpo_examples_count: int = Field(..., description="Number of DPO examples used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    original_evaluation_file: str = Field(..., description="Original evaluation results file")
    improved_evaluation_file: str = Field(..., description="Improved evaluation results file")
    dpo_training_file: str = Field(..., description="DPO training data file")


class DPOImprovementEngine:
    """Engine for improving models using DPO on incorrect answers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai.api_key
        self.client = OpenAI(api_key=self.api_key)
        self.fine_tuner = OpenAIFineTuner(api_key=self.api_key)
        self.evaluator = ModelEvaluator()
        
    def extract_wrong_answers(self, evaluation_results: Dict[str, Any]) -> List[DPOTrainingExample]:
        """Extract wrong answers from evaluation results and create DPO training examples"""
        dpo_examples = []
        
        logger.info("Extracting wrong answers from evaluation results...")
        
        # Extract domain from evaluation results
        domain = evaluation_results.get("evaluation_results", {}).get("domain", "Unknown Domain")
        logger.info(f"Detected domain from evaluation results: {domain}")
        
        # Navigate the evaluation results structure
        topic_results = evaluation_results.get("evaluation_results", {}).get("topic_results", [])
        
        for topic_result in topic_results:
            results = topic_result.get("results", [])
            
            for result in results:
                if not result.get("is_correct", True):  # If answer is wrong
                    # Create DPO example
                    dpo_example = DPOTrainingExample(
                        prompt=result.get("question", ""),
                        chosen=result.get("ideal_answer", ""),
                        rejected=result.get("model_answer", ""),
                        domain=domain
                    )
                    dpo_examples.append(dpo_example)
                    
                    logger.info(f"Added DPO example for question: {result.get('question_id', 'unknown')}")
        
        logger.info(f"Extracted {len(dpo_examples)} DPO training examples")
        return dpo_examples
    
    async def create_dpo_training_file(self, dpo_examples: List[DPOTrainingExample], 
                               output_dir: str = "data/training_data",
                               min_examples: int = 10) -> str:
        """Create DPO training file in JSONL format"""
        await async_mkdir(Path(output_dir))
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dpo_training_{timestamp}.jsonl"
        file_path = Path(output_dir) / filename
        
        logger.info(f"Creating DPO training file: {file_path}")
        
        # Duplicate examples if we don't have enough
        if len(dpo_examples) < min_examples:
            original_count = len(dpo_examples)
            # Make copies of the original examples
            examples_to_write = dpo_examples[:]
            
            while len(examples_to_write) < min_examples:
                # Add copies of existing examples
                for example in dpo_examples:
                    if len(examples_to_write) >= min_examples:
                        break
                    examples_to_write.append(example)
            
            logger.info(f"Duplicated examples from {original_count} to {len(examples_to_write)} to meet minimum requirement")
        else:
            examples_to_write = dpo_examples
        
        # Build all content first, then write in one operation
        lines = []
        for example in examples_to_write:
            lines.append(json.dumps(example.to_dpo_format()))
        
        content = '\n'.join(lines) + '\n'
        await async_write_text(file_path, content)
        
        logger.info(f"Created DPO training file with {len(examples_to_write)} examples")
        return str(file_path)
    
    @traceable
    async def perform_dpo_fine_tuning(self, 
                              training_file_path: str,
                              base_model: str,
                              hyperparameters: Optional[FineTuningHyperparameters] = None) -> str:
        """Perform DPO fine-tuning and return the new model ID"""
        logger.info(f"Starting DPO fine-tuning on base model: {base_model}")
        
        # Note: DPO method may not support custom hyperparameters
        # Using None to let OpenAI use default DPO settings
        hyperparameters = None
        
        # Upload training file
        logger.info("Uploading DPO training file...")
        upload_result = self.fine_tuner.upload_training_file(training_file_path)
        
        # Create DPO fine-tuning job
        logger.info("Creating DPO fine-tuning job...")
        
        # Create DPO fine-tuning job
        from .fine_tuner import FineTuningMethod
        job_result = self.fine_tuner.create_fine_tuning_job(
            training_file_id=upload_result.file_id,
            model=base_model,
            hyperparameters=hyperparameters,
            suffix="dpo",
            method=FineTuningMethod.DPO
        )
        
        # Wait for completion
        logger.info(f"Waiting for DPO fine-tuning job completion: {job_result.job_id}")
        final_result = await self.fine_tuner.async_wait_for_job_completion(job_result.job_id)
        
        if final_result.status.value == "succeeded":
            if final_result.fine_tuned_model:
                logger.info(f"DPO fine-tuning completed successfully: {final_result.fine_tuned_model}")
                return final_result.fine_tuned_model
            else:
                raise Exception("DPO fine-tuning succeeded but no model ID returned")
        else:
            raise Exception(f"DPO fine-tuning failed: {final_result.error}")
    
    @traceable
    async def improve_model_with_dpo(self, 
                                   evaluation_results_file: str,
                                   training_data_file: str,
                                   output_dir: str = "data/evaluations") -> DPOImprovementResult:
        """
        Complete DPO improvement workflow:
        1. Extract wrong answers from evaluation results
        2. Create DPO training data
        3. Perform DPO fine-tuning
        4. Re-evaluate the improved model
        5. Return improvement results
        """
        start_time = datetime.now()
        
        logger.info(f"Starting DPO improvement workflow for evaluation: {evaluation_results_file}")
        
        # Step 1: Load evaluation results
        evaluation_results = await async_read_json(evaluation_results_file)
        
        original_model = evaluation_results["file_metadata"]["model_tested"]
        original_accuracy = evaluation_results["file_metadata"]["overall_accuracy"]
        
        logger.info(f"Original model: {original_model}")
        logger.info(f"Original accuracy: {original_accuracy:.2%}")
        
        # Step 2: Extract wrong answers
        dpo_examples = self.extract_wrong_answers(evaluation_results)
        
        if not dpo_examples:
            logger.warning("No wrong answers found - model is already perfect!")
            # Return the same model with no improvement
            return DPOImprovementResult(
                original_model=original_model,
                dpo_model=original_model,
                original_accuracy=original_accuracy,
                improved_accuracy=original_accuracy,
                improvement=0.0,
                dpo_examples_count=0,
                processing_time=0.0,
                original_evaluation_file=evaluation_results_file,
                improved_evaluation_file=evaluation_results_file,
                dpo_training_file=""
            )
        
        # Step 3: Create DPO training file
        dpo_training_file = await self.create_dpo_training_file(dpo_examples)
        
        # Step 4: Perform DPO fine-tuning
        dpo_model = await self.perform_dpo_fine_tuning(dpo_training_file, original_model)
        
        # Step 5: Re-evaluate the DPO model
        logger.info(f"Re-evaluating DPO model: {dpo_model}")
        
        # Load training data for evaluation
        training_data_json = await async_read_json(training_data_file)
        
        # Create a new evaluator with the DPO model
        from .training_data_generator import CurriculumTrainingData
        
        # Convert JSON to CurriculumTrainingData object
        training_data = CurriculumTrainingData.model_validate(training_data_json["training_data"])
        
        # Create evaluator for DPO model
        dpo_evaluator = ModelEvaluator(model_to_test=dpo_model)
        
        # Run evaluation on DPO model
        evaluation_summary = await dpo_evaluator.evaluate_training_data(training_data)
        
        # Save evaluation results to get JSON format compatible with our expected structure
        eval_filename = dpo_evaluator.save_evaluation_results(evaluation_summary)
        
        # Load the saved results to get the expected JSON structure
        improved_results = await async_read_json(eval_filename)
        
        # Step 6: Save improved evaluation results
        await async_mkdir(Path(output_dir))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        improved_eval_file = Path(output_dir) / f"evaluation_results_dpo_{timestamp}.json"
        
        await async_write_json(improved_eval_file, improved_results)
        
        # Step 7: Calculate improvement
        improved_accuracy = improved_results["file_metadata"]["overall_accuracy"]
        improvement = improved_accuracy - original_accuracy
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = DPOImprovementResult(
            original_model=original_model,
            dpo_model=dpo_model,
            original_accuracy=original_accuracy,
            improved_accuracy=improved_accuracy,
            improvement=improvement,
            dpo_examples_count=len(dpo_examples),
            processing_time=processing_time,
            original_evaluation_file=evaluation_results_file,
            improved_evaluation_file=str(improved_eval_file),
            dpo_training_file=dpo_training_file
        )
        
        logger.info("DPO improvement workflow completed!")
        logger.info(f"Original accuracy: {original_accuracy:.2%}")
        logger.info(f"Improved accuracy: {improved_accuracy:.2%}")
        logger.info(f"Improvement: {improvement:.2%}")
        logger.info(f"DPO examples used: {len(dpo_examples)}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        
        return result
    
    async def save_improvement_summary(self, result: DPOImprovementResult, 
                               output_file: str = "dpo_improvement_summary.json") -> None:
        """Save improvement summary to JSON file"""
        summary = {
            "dpo_improvement_summary": {
                "timestamp": datetime.now().isoformat(),
                "original_model": result.original_model,
                "dpo_model": result.dpo_model,
                "accuracy_improvement": {
                    "original_accuracy": result.original_accuracy,
                    "improved_accuracy": result.improved_accuracy,
                    "improvement": result.improvement,
                    "improvement_percentage": f"{result.improvement:.2%}"
                },
                "training_details": {
                    "dpo_examples_count": result.dpo_examples_count,
                    "processing_time_seconds": result.processing_time
                },
                "files": {
                    "original_evaluation": result.original_evaluation_file,
                    "improved_evaluation": result.improved_evaluation_file,
                    "dpo_training_data": result.dpo_training_file
                }
            }
        }
        
        await async_write_json(output_file, summary)
        
        logger.info(f"Improvement summary saved to: {output_file}")


# Factory function
def create_dpo_improvement_engine(api_key: Optional[str] = None) -> DPOImprovementEngine:
    """Create a configured DPO improvement engine"""
    return DPOImprovementEngine(api_key=api_key)


# Main function for CLI usage
async def main():
    """Main function for running DPO improvement from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improve model using DPO on wrong answers")
    parser.add_argument("--evaluation-file", required=True, help="Path to evaluation results JSON file")
    parser.add_argument("--training-data-file", required=True, help="Path to original training data file")
    parser.add_argument("--domain", required=False, help="Domain name (optional, extracted from evaluation results)")
    parser.add_argument("--output-dir", default="data/evaluations", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create DPO improvement engine
    engine = create_dpo_improvement_engine()
    
    # Run improvement workflow
    result = await engine.improve_model_with_dpo(
        evaluation_results_file=args.evaluation_file,
        training_data_file=args.training_data_file,
        output_dir=args.output_dir
    )
    
    # Save improvement summary
    await engine.save_improvement_summary(result)
    
    print(f"DPO improvement completed!")
    print(f"Original accuracy: {result.original_accuracy:.2%}")
    print(f"Improved accuracy: {result.improved_accuracy:.2%}")
    print(f"Improvement: {result.improvement:.2%}")


if __name__ == "__main__":
    asyncio.run(main()) 