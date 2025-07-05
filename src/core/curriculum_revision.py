"""
Curriculum revision utilities based on evaluation results
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    from .deep_research_client import create_deep_research_client, Curriculum, CurriculumRevisionResult
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.deep_research_client import create_deep_research_client, Curriculum, CurriculumRevisionResult
    from utils.logger import get_logger

logger = get_logger(__name__)


class CurriculumRevisionEngine:
    """Engine for revising curriculum based on evaluation performance"""
    
    def __init__(self, accuracy_threshold: float = 0.9):
        """
        Initialize the curriculum revision engine
        
        Args:
            accuracy_threshold: Threshold for determining topic mastery (default 0.9 for 90%)
        """
        self.accuracy_threshold = accuracy_threshold
        self.client = create_deep_research_client()
    
    async def revise_curriculum_from_evaluation_file(self, 
                                                   evaluation_file_path: str,
                                                   current_curriculum_file: Optional[str] = None,
                                                   save_results: bool = True) -> Optional[CurriculumRevisionResult]:
        """
        Revise curriculum from evaluation results file
        
        Args:
            evaluation_file_path: Path to evaluation results JSON file
            current_curriculum_file: Optional path to current curriculum JSON file
            save_results: Whether to save results to file
        
        Returns:
            CurriculumRevisionResult or None if failed
        """
        
        try:
            # Load evaluation results
            with open(evaluation_file_path, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            
            # Load current curriculum if provided
            current_curriculum = None
            if current_curriculum_file:
                try:
                    with open(current_curriculum_file, 'r', encoding='utf-8') as f:
                        curriculum_data = json.load(f)
                        current_curriculum = curriculum_data.get("curriculum")
                        logger.info(f"Loaded current curriculum from {current_curriculum_file}")
                except Exception as e:
                    logger.warning(f"Could not load current curriculum: {e}")
            
            # Generate revision
            result = await self.revise_curriculum_from_evaluation(
                evaluation_results=evaluation_results,
                current_curriculum=current_curriculum
            )
            
            # Save results if requested
            if save_results and result:
                self.save_revision_results(result, evaluation_results)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to revise curriculum from file: {e}")
            return None
    
    async def revise_curriculum_from_evaluation(self, 
                                              evaluation_results: Dict[str, Any],
                                              current_curriculum: Optional[Dict[str, Any]] = None) -> Optional[CurriculumRevisionResult]:
        """
        Revise curriculum from evaluation results
        
        Args:
            evaluation_results: Evaluation results dictionary
            current_curriculum: Optional current curriculum dictionary
        
        Returns:
            CurriculumRevisionResult or None if failed
        """
        
        # Convert dictionary to Curriculum object if needed
        curriculum_obj = None
        if current_curriculum:
            try:
                from pydantic import ValidationError
                curriculum_obj = Curriculum.model_validate(current_curriculum)
            except (ValidationError, Exception) as e:
                logger.warning(f"Could not convert curriculum dict to Curriculum object: {e}")
                curriculum_obj = None
        
        return await self.client.generate_revised_curriculum_from_evaluation(
            evaluation_results=evaluation_results,
            current_curriculum=curriculum_obj,
            accuracy_threshold=self.accuracy_threshold
        )
    
    def analyze_evaluation_performance(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze evaluation performance and extract key metrics
        
        Args:
            evaluation_results: Evaluation results dictionary
        
        Returns:
            Dictionary with performance analysis
        """
        
        eval_data = evaluation_results.get("evaluation_results", {})
        topic_results = eval_data.get("topic_results", [])
        
        analysis = {
            "domain": eval_data.get("domain", "Unknown"),
            "overall_accuracy": eval_data.get("overall_accuracy", 0.0),
            "total_questions": eval_data.get("total_questions", 0),
            "total_topics": eval_data.get("total_topics", 0),
            "mastered_topics": [],
            "failed_topics": [],
            "topic_performance": [],
            "failed_questions_count": 0
        }
        
        for topic_result in topic_results:
            topic_name = topic_result.get("topic_name", "Unknown Topic")
            topic_accuracy = topic_result.get("accuracy", 0.0)
            
            topic_info = {
                "name": topic_name,
                "accuracy": topic_accuracy,
                "total_questions": topic_result.get("total_questions", 0),
                "correct_answers": topic_result.get("correct_answers", 0),
                "incorrect_answers": topic_result.get("incorrect_answers", 0),
                "status": "mastered" if topic_accuracy >= self.accuracy_threshold else "needs_work"
            }
            
            analysis["topic_performance"].append(topic_info)
            
            if topic_accuracy >= self.accuracy_threshold:
                analysis["mastered_topics"].append(topic_name)
            else:
                analysis["failed_topics"].append(topic_name)
                
                # Count failed questions in this topic
                results = topic_result.get("results", [])
                failed_count = sum(1 for result in results if not result.get("is_correct", True))
                analysis["failed_questions_count"] += failed_count
        
        return analysis
    
    def save_revision_results(self, 
                            revision_result: CurriculumRevisionResult, 
                            evaluation_results: Dict[str, Any],
                            filename: Optional[str] = None) -> str:
        """
        Save revision results to JSON file
        
        Args:
            revision_result: The curriculum revision result
            evaluation_results: Original evaluation results
            filename: Optional custom filename
        
        Returns:
            Filename where results were saved
        """
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = evaluation_results.get("evaluation_results", {}).get("domain", "unknown")
            domain_clean = domain.lower().replace(" ", "_").replace("-", "_")
            filename = f"curriculum_revision_{domain_clean}_{timestamp}.json"
        
        try:
            # Convert Pydantic models to dicts for JSON serialization
            results_data = {
                "revision_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "source_evaluation": evaluation_results.get("file_metadata", {}),
                    "accuracy_threshold": self.accuracy_threshold,
                    "revision_summary": revision_result.revision_summary
                },
                "performance_analysis": {
                    "mastered_topics": revision_result.mastered_topics,
                    "failed_topics": revision_result.failed_topics,
                    "failed_questions_count": revision_result.failed_questions_count
                },
                "original_curriculum": revision_result.original_curriculum.model_dump() if revision_result.original_curriculum else None,
                "revised_curriculum": revision_result.revised_curriculum.model_dump() if revision_result.revised_curriculum else None
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Revision results saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save revision results: {e}")
            raise


async def revise_curriculum_from_dpo_results(
    evaluation_results_path: str,
    current_curriculum_path: Optional[str] = None,
    accuracy_threshold: float = 0.9,
    iteration: int = 1  # Add iteration parameter
) -> Optional[CurriculumRevisionResult]:
    """
    Convenience function to revise curriculum from DPO evaluation results
    
    Args:
        evaluation_results_path: Path to DPO evaluation results JSON file
        current_curriculum_path: Optional path to current curriculum JSON file
        accuracy_threshold: Threshold for determining mastery (default 0.9)
        iteration: Current learning iteration number
        
    Returns:
        CurriculumRevisionResult or None if failed
    """
    
    try:
        # Load evaluation results
        with open(evaluation_results_path, 'r', encoding='utf-8') as f:
            evaluation_results = json.load(f)
        
        # Load current curriculum if provided
        current_curriculum = None
        if current_curriculum_path and Path(current_curriculum_path).exists():
            with open(current_curriculum_path, 'r', encoding='utf-8') as f:
                curriculum_data = json.load(f)
                
                # Handle nested curriculum structure (e.g., from test files)
                if "curriculum" in curriculum_data:
                    curriculum_dict = curriculum_data["curriculum"]
                else:
                    curriculum_dict = curriculum_data
                    
                current_curriculum = Curriculum(**curriculum_dict)
        
        # Create deep research client and generate revision
        client = create_deep_research_client()
        
        result = await client.generate_revised_curriculum_from_evaluation(
            evaluation_results=evaluation_results,
            current_curriculum=current_curriculum,
            accuracy_threshold=accuracy_threshold,
            iteration=iteration  # Pass iteration parameter
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to revise curriculum from DPO results: {e}")
        return None


def analyze_dpo_evaluation_performance(evaluation_file_path: str,
                                     accuracy_threshold: float = 0.9) -> Dict[str, Any]:
    """
    Analyze DPO evaluation performance and extract key metrics
    
    Args:
        evaluation_file_path: Path to DPO evaluation results JSON file
        accuracy_threshold: Threshold for determining mastery (default 90%)
    
    Returns:
        Dictionary with performance analysis
    
    Example:
        >>> analysis = analyze_dpo_evaluation_performance("results.json")
        >>> print(f"Domain: {analysis['domain']}")
        >>> print(f"Overall accuracy: {analysis['overall_accuracy']:.1%}")
        >>> print(f"Mastered topics: {len(analysis['mastered_topics'])}")
    """
    
    try:
        with open(evaluation_file_path, 'r', encoding='utf-8') as f:
            evaluation_results = json.load(f)
        
        engine = CurriculumRevisionEngine(accuracy_threshold=accuracy_threshold)
        return engine.analyze_evaluation_performance(evaluation_results)
        
    except Exception as e:
        logger.error(f"Failed to analyze evaluation performance: {e}")
        raise


# Example usage functions for different scenarios

async def handle_high_performance_learner(evaluation_file_path: str) -> Optional[CurriculumRevisionResult]:
    """
    Handle curriculum revision for high-performing learner (>90% accuracy)
    Focus on advanced topics building on mastered knowledge
    """
    
    logger.info("Processing high-performance learner curriculum revision")
    return await revise_curriculum_from_dpo_results(
        evaluation_results_path=evaluation_file_path,
        accuracy_threshold=0.9,
        iteration=1
    )


async def handle_struggling_learner(evaluation_file_path: str) -> Optional[CurriculumRevisionResult]:
    """
    Handle curriculum revision for struggling learner (<70% accuracy)
    Focus on remedial topics addressing knowledge gaps
    """
    
    logger.info("Processing struggling learner curriculum revision")
    return await revise_curriculum_from_dpo_results(
        evaluation_results_path=evaluation_file_path,
        accuracy_threshold=0.7,  # Lower threshold for struggling learners
        iteration=1
    )


async def handle_mixed_performance_learner(evaluation_file_path: str, 
                                         current_curriculum_file: str) -> Optional[CurriculumRevisionResult]:
    """
    Handle curriculum revision for mixed-performance learner (70-90% accuracy)
    Balance remedial and advanced topics based on specific performance
    """
    
    logger.info("Processing mixed-performance learner curriculum revision")
    return await revise_curriculum_from_dpo_results(
        evaluation_results_path=evaluation_file_path,
        current_curriculum_path=current_curriculum_file,
        accuracy_threshold=0.8,  # Moderate threshold
        iteration=1
    ) 