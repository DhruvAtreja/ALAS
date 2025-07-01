"""
Evaluator for testing model performance on training questions using GPT-4.1 and Deep Research API
"""

import asyncio
import json
import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .deep_research_client import create_deep_research_client
from .training_data_generator import TrainingQuestion, TopicTrainingData, CurriculumTrainingData
from ..utils.logger import get_logger, log_api_call, log_cost, log_error
from ..config.settings import settings

logger = get_logger(__name__)


class ModelResponse(BaseModel):
    """Response from a model to a question"""
    question_id: str
    question: str
    model_answer: str
    ideal_answer: str
    topic_id: str
    topic_name: str
    category: str
    difficulty: str


class EvaluationResult(BaseModel):
    """Result of evaluating a model's answer"""
    question_id: str
    question: str
    model_answer: str
    ideal_answer: str
    is_correct: bool
    explanation: str
    topic_id: str
    topic_name: str
    category: str
    difficulty: str


class TopicEvaluationResults(BaseModel):
    """Evaluation results for a specific topic"""
    topic_id: str
    topic_name: str
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    accuracy: float
    results: List[EvaluationResult]
    evaluation_metadata: Dict[str, Any]


class EvaluationSummary(BaseModel):
    """Summary of evaluation across all topics"""
    domain: str
    model_tested: str
    total_topics: int
    total_questions: int
    overall_accuracy: float
    topic_results: List[TopicEvaluationResults]
    category_performance: Dict[str, Dict[str, Any]]
    difficulty_performance: Dict[str, Dict[str, Any]]
    evaluation_metadata: Dict[str, Any]


class ModelEvaluator:
    """Evaluates model performance on training data using Deep Research API"""
    
    def __init__(self, model_to_test: str = "gpt-4.1-2025-04-14", max_concurrent: int = 3):
        self.model_to_test = model_to_test
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize clients
        self.openai_client = AsyncOpenAI(api_key=settings.openai.api_key)
        self.deep_research_client = create_deep_research_client()
        
    async def get_model_answer(self, question: str, topic_name: str) -> str:
        """Get the model's answer to a question"""
        
        async with self.semaphore:
            try:
                # Create a simple prompt for the model
                messages: List[ChatCompletionMessageParam] = [
                    {
                        "role": "system", 
                        "content": f"You are an expert in {topic_name}. Answer the following question accurately and comprehensively."
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ]
                
                response = await self.openai_client.chat.completions.create(
                    model=self.model_to_test,
                    messages=messages,  # type: ignore
                    temperature=0.1,
                    max_tokens=2048
                )
                
                return response.choices[0].message.content or ""
                
            except Exception as e:
                log_error(e, {"question": question[:100], "model": self.model_to_test})
                return f"Error getting model response: {e}"
    
    async def get_all_model_responses(self, training_data: CurriculumTrainingData) -> List[ModelResponse]:
        """Get model responses for all questions in the training data"""
        
        logger.info(f"Getting {self.model_to_test} responses for {training_data.total_questions} questions")
        
        # Create tasks for all questions
        tasks = []
        for topic in training_data.topics:
            for question in topic.questions:
                task = self._get_single_response(question, topic.topic_name)
                tasks.append(task)
        
        # Execute in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        model_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to get response for question {i}: {response}")
            elif isinstance(response, ModelResponse):
                model_responses.append(response)
        
        logger.info(f"✅ Got {len(model_responses)} model responses")
        return model_responses
    
    async def _get_single_response(self, question: TrainingQuestion, topic_name: str) -> ModelResponse:
        """Get a single model response"""
        
        try:
            model_answer = await self.get_model_answer(question.question, topic_name)
            
            return ModelResponse(
                question_id=question.id,
                question=question.question,
                model_answer=model_answer,
                ideal_answer=question.answer,
                topic_id=question.topic_id,
                topic_name=topic_name,
                category=question.category,
                difficulty=question.difficulty
            )
            
        except Exception as e:
            logger.error(f"Error processing question {question.id}: {e}")
            raise
    
    def group_responses_by_topic(self, responses: List[ModelResponse]) -> Dict[str, List[ModelResponse]]:
        """Group model responses by topic"""
        
        grouped = {}
        for response in responses:
            topic_id = response.topic_id
            if topic_id not in grouped:
                grouped[topic_id] = []
            grouped[topic_id].append(response)
        
        return grouped
    
    async def evaluate_topic_responses(self, topic_id: str, responses: List[ModelResponse]) -> TopicEvaluationResults:
        """Evaluate model responses for a specific topic using Deep Research API with retry logic"""
        
        if not responses:
            raise ValueError(f"No responses provided for topic {topic_id}")
        
        topic_name = responses[0].topic_name
        logger.info(f"Evaluating {len(responses)} responses for topic: {topic_name}")
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(topic_name, responses)
        
        # Retry logic for API call and parsing
        max_retries = 2
        last_error: Optional[Exception] = None
        
        for attempt in range(max_retries + 1):  # 0, 1, 2 (3 total attempts)
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for topic: {topic_name}")
                
                # Call Deep Research API
                start_time = datetime.now()
                evaluation_response = await self.deep_research_client.research_topic(
                    query=prompt,
                    domain=f"Evaluation of {topic_name}",
                    depth="fast"
                )
                duration = (datetime.now() - start_time).total_seconds()

                if attempt == 0:  # Only print on first attempt to avoid spam
                    print(evaluation_response)
                
                # Parse evaluation results
                evaluation_results = self._parse_evaluation_response(evaluation_response, responses)
                
                # Check if parsing was successful (not all fallback evaluations)
                successful_evaluations = [r for r in evaluation_results if r.explanation != "Evaluation failed - unable to parse Deep Research response"]
                
                if len(successful_evaluations) > 0:
                    # Successful parsing - calculate topic statistics
                    correct_count = sum(1 for result in evaluation_results if result.is_correct)
                    total_count = len(evaluation_results)
                    accuracy = correct_count / total_count if total_count > 0 else 0.0
                    
                    topic_results = TopicEvaluationResults(
                        topic_id=topic_id,
                        topic_name=topic_name,
                        total_questions=total_count,
                        correct_answers=correct_count,
                        incorrect_answers=total_count - correct_count,
                        accuracy=accuracy,
                        results=evaluation_results,
                        evaluation_metadata={
                            "evaluated_at": datetime.now().isoformat(),
                            "evaluation_duration": duration,
                            "model_tested": self.model_to_test,
                            "evaluator": "Deep Research API",
                            "retry_attempts": attempt
                        }
                    )
                    
                    if attempt > 0:
                        logger.info(f"✅ Successful evaluation on attempt {attempt + 1}")
                    logger.info(f"✅ Topic {topic_name}: {correct_count}/{total_count} correct ({accuracy:.1%})")
                    return topic_results
                else:
                    # All evaluations were fallbacks - this means parsing failed
                    raise ValueError("Failed to parse any evaluations from response")
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    if "XML parsing error" in str(e) or "Failed to parse" in str(e):
                        logger.warning(f"Parsing error on attempt {attempt + 1}: {e}")
                        logger.info(f"Will retry... ({max_retries - attempt} retries remaining)")
                        continue
                    else:
                        # Non-parsing error, don't retry
                        logger.error(f"Non-retryable error: {e}")
                        break
                else:
                    # Final attempt failed
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
        
        # All retries exhausted - log final error and raise
        if last_error is None:
            last_error = Exception(f"Evaluation failed for topic {topic_name} after {max_retries + 1} attempts")
            
        log_error(last_error, {
            "topic_id": topic_id, 
            "topic_name": topic_name, 
            "max_retries": max_retries,
            "final_attempt": True
        })
        raise last_error
    
    def _build_evaluation_prompt(self, topic_name: str, responses: List[ModelResponse]) -> str:
        """Build evaluation prompt for Deep Research API"""
        
        prompt_parts = [
            f"Evaluate the following {len(responses)} question-answer pairs for the topic '{topic_name}'.",
            "",
            "For each question, you have:",
            "1. The original question",
            "2. The model's answer",
            "3. The ideal/correct answer",
            "",
            "Your task is to determine if the model's answer is correct",
            "Use the following criteria:",
            "- The model's answer must convey the core information from the ideal answer",
            "- Differences in wording or style are acceptable",
            "- The model's answer should not contradict the ideal answer",
            "- Additional information is fine as long as it's accurate and relevant",
            "- All we need to evaluate is if the model understands the concept, not if it matches exactly the ideal answer",
            "Format your response as XML:",
            "",
            "<evaluations>",
        ]
        
        # Add template for each question
        for i, response in enumerate(responses, 1):
            prompt_parts.extend([
                f"<evaluation-{i}>",
                f"<question_id>{response.question_id}</question_id>",
                f"<is_correct>yes/no</is_correct>",
                f"<explanation>Brief explanation of why the answer is correct or incorrect</explanation>",
                f"</evaluation-{i}>",
            ])
        
        prompt_parts.extend([
            "</evaluations>",
            "",
            "Here are the question-answer pairs to evaluate:",
            ""
        ])
        
        # Add actual questions and answers
        for i, response in enumerate(responses, 1):
            prompt_parts.extend([
                f"Question {i}:",
                f"ID: {response.question_id}",
                f"Question: {response.question}",
                f"Model Answer: {response.model_answer}",
                f"Ideal Answer: {response.ideal_answer}",
                f"Category: {response.category}",
                f"Difficulty: {response.difficulty}",
                ""
            ])
        
        return "\n".join(prompt_parts)
    
    def _parse_evaluation_response(self, response: str, original_responses: List[ModelResponse]) -> List[EvaluationResult]:
        """Parse evaluation results from XML response"""
        
        try:
            # Extract XML content
            xml_match = re.search(r'<evaluations>(.*?)</evaluations>', response, re.DOTALL)
            if not xml_match:
                logger.warning("No <evaluations> tags found in response")
                return self._create_fallback_evaluations(original_responses)
            
            xml_content = f"<evaluations>{xml_match.group(1)}</evaluations>"
            
            # Clean XML content
            xml_content = xml_content.replace('&', '&amp;')
            xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Create mapping of question ID to original response
            response_map = {resp.question_id: resp for resp in original_responses}
            
            evaluations = []
            evaluation_elements = [elem for elem in root if elem.tag.startswith('evaluation')]
            
            for eval_elem in evaluation_elements:
                try:
                    # Extract evaluation data
                    question_id_elem = eval_elem.find('question_id')
                    is_correct_elem = eval_elem.find('is_correct')
                    explanation_elem = eval_elem.find('explanation')
                    
                    if question_id_elem is None or is_correct_elem is None:
                        logger.warning(f"Missing required fields in evaluation element")
                        continue
                    
                    question_id = question_id_elem.text.strip() if question_id_elem.text else ""
                    is_correct_text = is_correct_elem.text.strip().lower() if is_correct_elem.text else "no"
                    is_correct = is_correct_text in ["yes", "true", "correct"]
                    explanation = explanation_elem.text.strip() if explanation_elem is not None and explanation_elem.text else "No explanation provided"
                    
                    # Get original response
                    original_response = response_map.get(question_id)
                    if not original_response:
                        logger.warning(f"No original response found for question ID: {question_id}")
                        continue
                    
                    evaluation = EvaluationResult(
                        question_id=question_id,
                        question=original_response.question,
                        model_answer=original_response.model_answer,
                        ideal_answer=original_response.ideal_answer,
                        is_correct=is_correct,
                        explanation=explanation,
                        topic_id=original_response.topic_id,
                        topic_name=original_response.topic_name,
                        category=original_response.category,
                        difficulty=original_response.difficulty
                    )
                    
                    evaluations.append(evaluation)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse evaluation element: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(evaluations)} evaluations")
            return evaluations
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return self._create_fallback_evaluations(original_responses)
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return self._create_fallback_evaluations(original_responses)
    
    def _create_fallback_evaluations(self, responses: List[ModelResponse]) -> List[EvaluationResult]:
        """Create fallback evaluations when parsing fails"""
        
        fallback_evaluations = []
        for response in responses:
            # Simple fallback: consider it incorrect with explanation
            evaluation = EvaluationResult(
                question_id=response.question_id,
                question=response.question,
                model_answer=response.model_answer,
                ideal_answer=response.ideal_answer,
                is_correct=False,
                explanation="Evaluation failed - unable to parse Deep Research response",
                topic_id=response.topic_id,
                topic_name=response.topic_name,
                category=response.category,
                difficulty=response.difficulty
            )
            fallback_evaluations.append(evaluation)
        
        return fallback_evaluations
    
    async def evaluate_training_data(self, training_data: CurriculumTrainingData) -> EvaluationSummary:
        """Evaluate model performance on complete training data"""
        
        logger.info(f"Starting evaluation of {self.model_to_test} on {training_data.total_questions} questions")
        start_time = datetime.now()
        
        # Get all model responses
        model_responses = await self.get_all_model_responses(training_data)
        
        # Group by topic
        topic_groups = self.group_responses_by_topic(model_responses)
        
        # Evaluate each topic
        topic_evaluations = []
        for topic_id, responses in topic_groups.items():
            try:
                topic_eval = await self.evaluate_topic_responses(topic_id, responses)
                topic_evaluations.append(topic_eval)
            except Exception as e:
                logger.error(f"Failed to evaluate topic {topic_id}: {e}")
                continue
        
        # Calculate overall statistics
        total_questions = sum(topic.total_questions for topic in topic_evaluations)
        total_correct = sum(topic.correct_answers for topic in topic_evaluations)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # Calculate category and difficulty performance
        category_performance = self._calculate_category_performance(topic_evaluations)
        difficulty_performance = self._calculate_difficulty_performance(topic_evaluations)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        evaluation_summary = EvaluationSummary(
            domain=training_data.domain,
            model_tested=self.model_to_test,
            total_topics=len(topic_evaluations),
            total_questions=total_questions,
            overall_accuracy=overall_accuracy,
            topic_results=topic_evaluations,
            category_performance=category_performance,
            difficulty_performance=difficulty_performance,
            evaluation_metadata={
                "evaluated_at": datetime.now().isoformat(),
                "total_duration_seconds": total_duration,
                "evaluator_version": "1.0.0",
                "questions_per_minute": (total_questions / total_duration) * 60 if total_duration > 0 else 0
            }
        )
        
        logger.info(f"🎉 Evaluation completed!")
        logger.info(f"  Overall accuracy: {overall_accuracy:.1%} ({total_correct}/{total_questions})")
        logger.info(f"  Topics evaluated: {len(topic_evaluations)}")
        logger.info(f"  Total time: {total_duration:.2f}s")
        
        return evaluation_summary
    
    def _calculate_category_performance(self, topic_evaluations: List[TopicEvaluationResults]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance statistics by category"""
        
        category_stats = {}
        
        for topic in topic_evaluations:
            for result in topic.results:
                category = result.category
                if category not in category_stats:
                    category_stats[category] = {"correct": 0, "total": 0}
                
                category_stats[category]["total"] += 1
                if result.is_correct:
                    category_stats[category]["correct"] += 1
        
        # Calculate percentages
        for category in category_stats:
            stats = category_stats[category]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return category_stats
    
    def _calculate_difficulty_performance(self, topic_evaluations: List[TopicEvaluationResults]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance statistics by difficulty"""
        
        difficulty_stats = {}
        
        for topic in topic_evaluations:
            for result in topic.results:
                difficulty = result.difficulty
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {"correct": 0, "total": 0}
                
                difficulty_stats[difficulty]["total"] += 1
                if result.is_correct:
                    difficulty_stats[difficulty]["correct"] += 1
        
        # Calculate percentages
        for difficulty in difficulty_stats:
            stats = difficulty_stats[difficulty]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return difficulty_stats
    
    def save_evaluation_results(self, evaluation_summary: EvaluationSummary, filename: Optional[str] = None) -> str:
        """Save evaluation results to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_domain = evaluation_summary.domain.lower().replace(' ', '_')
            filename = f"evaluation_results_{safe_domain}_{self.model_to_test}_{timestamp}.json"
        
        try:
            # Convert to dictionary for JSON serialization
            data_dict = evaluation_summary.model_dump()
            
            # Add file metadata
            file_metadata = {
                "file_generated_at": datetime.now().isoformat(),
                "evaluator_version": "1.0.0",
                "format_version": "1.0",
                "model_tested": self.model_to_test,
                "total_questions": evaluation_summary.total_questions,
                "overall_accuracy": evaluation_summary.overall_accuracy
            }
            
            final_data = {
                "file_metadata": file_metadata,
                "evaluation_results": data_dict
            }
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Evaluation results saved to {filename}")
            logger.info(f"  - File size: {Path(filename).stat().st_size / 1024:.1f} KB")
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            raise


# Factory function
def create_model_evaluator(model_to_test: str = "gpt-4.1-2025-04-14", max_concurrent: int = 3) -> ModelEvaluator:
    """Create a configured model evaluator"""
    return ModelEvaluator(model_to_test=model_to_test, max_concurrent=max_concurrent) 