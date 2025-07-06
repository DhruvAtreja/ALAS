"""
Training Data Generator for ALAS - generates Q&A pairs from curriculum topics
"""

import asyncio
import json
import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from langsmith import traceable

from .deep_research_client import DeepResearchClient, create_deep_research_client, Curriculum, CurriculumTopic
from ..utils.logger import get_logger, log_cost, log_error
from ..utils.async_file_utils import async_write_json, async_write_text, async_append_text
from ..config.settings import settings

logger = get_logger(__name__)


class TrainingQuestion(BaseModel):
    """Represents a single training question"""
    id: str
    topic_id: str
    question: str
    answer: str
    category: str  # Factual, Conceptual, Application, Analysis, Synthesis
    difficulty: str  # easy, medium, hard
    explanation: Optional[str] = None
    source_topic: str


class TopicTrainingData(BaseModel):
    """Training data for a specific topic"""
    topic_id: str
    topic_name: str
    questions: List[TrainingQuestion]
    generation_metadata: Dict[str, Any]


class CurriculumTrainingData(BaseModel):
    """Complete training data for a curriculum"""
    domain: str
    curriculum_metadata: Dict[str, Any]
    topics: List[TopicTrainingData]
    total_questions: int
    generation_summary: Dict[str, Any]


class TrainingDataGenerator:
    """Generates training data from curriculum topics using Deep Research API"""
    
    def __init__(self, max_concurrent: int = 50, questions_per_topic: int = 10):
        self.client = create_deep_research_client()
        self.max_concurrent = max_concurrent
        self.questions_per_topic = questions_per_topic
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    @traceable
    async def generate_topic_questions(self, topic: CurriculumTopic, domain: str) -> TopicTrainingData:
        """Generate training questions for a single topic"""
        
        async with self.semaphore:
            try:
                logger.info(f"Generating {self.questions_per_topic} questions for topic: {topic.name}")
                
                # Create detailed prompt for question generation
                prompt = self._build_question_generation_prompt(topic, domain)
                
                # Make API call
                start_time = datetime.now()
                response = await self.client.research_topic(
                    query=prompt,
                    domain=topic.name,
                    depth="comprehensive"
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                
                # Parse questions from response
                questions = self._parse_questions_from_response(response, topic)
                
                # Create training data object
                training_data = TopicTrainingData(
                    topic_id=topic.id,
                    topic_name=topic.name,
                    questions=questions,
                    generation_metadata={
                        "generated_at": datetime.now().isoformat(),
                        "generation_duration": duration,
                        "questions_requested": self.questions_per_topic,
                        "questions_generated": len(questions),
                        "topic_difficulty": topic.difficulty.value,
                        "topic_depth": topic.depth
                    }
                )
                
                logger.info(f"✅ Generated {len(questions)} questions for {topic.name} in {duration:.2f}s")
                log_cost(topic.name, "question_generation", 0.1)  # Estimated cost
                
                return training_data
                
            except Exception as e:
                log_error(e, {"topic_id": topic.id, "topic_name": topic.name})
                
                # Return empty training data on error
                return TopicTrainingData(
                    topic_id=topic.id,
                    topic_name=topic.name,
                    questions=[],
                    generation_metadata={
                        "generated_at": datetime.now().isoformat(),
                        "error": str(e),
                        "questions_requested": self.questions_per_topic,
                        "questions_generated": 0
                    }
                )
    
    def _build_question_generation_prompt(self, topic: CurriculumTopic, domain: str) -> str:
        """Build a detailed prompt for generating training questions"""
        
        prompt = f"""Generate {self.questions_per_topic} diverse training questions and answers for the topic: "{topic.name}" in the domain of "{domain}. Use your web search to make sure you have up to date information about the topic".

Topic Details:
- Domain: {domain}
- Description: {topic.description}
- Learning Objectives: {topic.learning_objectives}
- Difficulty Level: {topic.difficulty.value}
- Prerequisites: {', '.join(topic.prerequisites) if topic.prerequisites else 'None'}

Generate questions across these categories (distribute evenly):
1. Factual Recall (definitions, terminology, basic facts)
2. Conceptual Understanding (explanations, relationships, principles)  
3. Application (problem-solving, implementation, real-world use)
4. Analysis (comparison, evaluation, critical thinking)
5. Synthesis (combining concepts, creative solutions, design)

Format your response as XML:

<questions>
<question-1>
<text>What is the definition of...?</text>
<answer>A detailed, accurate answer explaining...</answer>
<category>Factual Recall</category>
<difficulty>easy</difficulty>
<explanation>This tests basic knowledge of...</explanation>
</question-1>
<question-2>
<text>How would you implement...?</text>
<answer>To implement this, you would...</answer>
<category>Application</category>
<difficulty>medium</difficulty>
<explanation>This requires applying concepts to...</explanation>
</question-2>
...continue for {self.questions_per_topic} questions...
</questions>

Requirements:
- Questions should be specific and unambiguous
- Answers should be comprehensive and educational
- Vary difficulty levels within the topic's overall difficulty. Try to keep the difficulty level on the easy side.
- Focus on the learning objectives specified
- Make questions/answers suitable for fine-tuning a language model"""

        return prompt
    
    def _parse_questions_from_response(self, response: str, topic: CurriculumTopic) -> List[TrainingQuestion]:
        """Parse training questions from XML response"""
        
        try:
            # Extract XML content
            xml_match = re.search(r'<questions>(.*?)</questions>', response, re.DOTALL)
            if not xml_match:
                logger.warning(f"No <questions> tags found in response for {topic.name}")
                return []
            
            xml_content = f"<questions>{xml_match.group(1)}</questions>"
            
            # Clean XML content
            xml_content = xml_content.replace('&', '&amp;')
            xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            questions = []
            question_elements = [elem for elem in root if elem.tag.startswith('question')]
            
            for i, question_elem in enumerate(question_elements, 1):
                try:
                    # Extract question data
                    text_elem = question_elem.find('text')
                    answer_elem = question_elem.find('answer')
                    category_elem = question_elem.find('category')
                    difficulty_elem = question_elem.find('difficulty')
                    explanation_elem = question_elem.find('explanation')
                    
                    if text_elem is None or answer_elem is None:
                        logger.warning(f"Missing required fields in question {i} for {topic.name}")
                        continue
                    
                    question = TrainingQuestion(
                        id=f"{topic.id}_q{i:03d}",
                        topic_id=topic.id,
                        question=text_elem.text.strip() if text_elem.text else "",
                        answer=answer_elem.text.strip() if answer_elem.text else "",
                        category=category_elem.text.strip() if category_elem is not None and category_elem.text else "Conceptual Understanding",
                        difficulty=difficulty_elem.text.strip() if difficulty_elem is not None and difficulty_elem.text else topic.difficulty.value,
                        explanation=explanation_elem.text.strip() if explanation_elem is not None and explanation_elem.text else None,
                        source_topic=topic.name
                    )
                    
                    questions.append(question)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse question {i} for {topic.name}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(questions)} questions for {topic.name}")
            return questions
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error for {topic.name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing questions for {topic.name}: {e}")
            return []
    
    @traceable
    async def generate_curriculum_training_data(self, curriculum: Curriculum) -> CurriculumTrainingData:
        """Generate training data for all topics in a curriculum"""
        
        logger.info(f"Starting training data generation for {len(curriculum.topics)} topics")
        start_time = datetime.now()
        
        # Create tasks for parallel processing
        tasks = [
            self.generate_topic_questions(topic, curriculum.domain) 
            for topic in curriculum.topics
        ]
        
        # Execute in parallel with rate limiting
        logger.info(f"Processing topics with max {self.max_concurrent} concurrent requests")
        topic_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_topics = []
        failed_topics = []
        total_questions = 0
        
        for i, result in enumerate(topic_results):
            if isinstance(result, Exception):
                failed_topics.append({
                    "topic_name": curriculum.topics[i].name,
                    "error": str(result)
                })
                logger.error(f"Failed to generate questions for {curriculum.topics[i].name}: {result}")
            elif isinstance(result, TopicTrainingData):
                successful_topics.append(result)
                total_questions += len(result.questions)
        
        # Calculate generation summary
        total_duration = (datetime.now() - start_time).total_seconds()
        
        generation_summary = {
            "total_topics": len(curriculum.topics),
            "successful_topics": len(successful_topics),
            "failed_topics": len(failed_topics),
            "total_questions_generated": total_questions,
            "total_duration_seconds": total_duration,
            "average_questions_per_topic": total_questions / len(successful_topics) if successful_topics else 0,
            "questions_per_minute": (total_questions / total_duration) * 60 if total_duration > 0 else 0,
            "failed_topic_details": failed_topics
        }
        
        # Create final training data
        training_data = CurriculumTrainingData(
            domain=curriculum.domain,
            curriculum_metadata=curriculum.metadata.model_dump(),
            topics=successful_topics,
            total_questions=total_questions,
            generation_summary=generation_summary
        )
        
        logger.info(f"✅ Training data generation completed!")
        logger.info(f"  - Generated {total_questions} questions across {len(successful_topics)} topics")
        logger.info(f"  - Total time: {total_duration:.2f} seconds")
        logger.info(f"  - Rate: {generation_summary['questions_per_minute']:.1f} questions/minute")
        
        return training_data
    
    async def save_training_data(self, training_data: CurriculumTrainingData, filename: Optional[str] = None) -> str:
        """Save training data to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{training_data.domain.lower().replace(' ', '_')}_{timestamp}.json"
        
        try:
            # Convert to dictionary for JSON serialization
            data_dict = training_data.model_dump()
            
            # Add file metadata
            file_metadata = {
                "file_generated_at": datetime.now().isoformat(),
                "generator_version": "1.0.0",
                "format_version": "1.0",
                "total_topics": len(training_data.topics),
                "total_questions": training_data.total_questions
            }
            
            final_data = {
                "file_metadata": file_metadata,
                "training_data": data_dict
            }
            
            # Save to file
            await async_write_json(filename, final_data)
            
            logger.info(f"✅ Training data saved to {filename}")
            logger.info(f"  - File size: {Path(filename).stat().st_size / 1024:.1f} KB")
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            raise
    
    async def export_for_openai_finetuning(self, training_data: CurriculumTrainingData, filename: Optional[str] = None) -> str:
        """Export training data in OpenAI fine-tuning format (JSONL)"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openai_training_{training_data.domain.lower().replace(' ', '_')}_{timestamp}.jsonl"
        
        try:
            # Build all content first, then write in one operation
            lines = []
            for topic in training_data.topics:
                for question in topic.questions:
                    # Format for OpenAI fine-tuning
                    openai_example = {
                        "messages": [
                            {
                                "role": "system", 
                                "content": f"Answer the question"
                            },
                            {
                                "role": "user", 
                                "content": question.question
                            },
                            {
                                "role": "assistant", 
                                "content": question.answer
                            }
                        ]
                    }
                    
                    lines.append(json.dumps(openai_example, ensure_ascii=False))
            
            # Write all lines at once
            content = '\n'.join(lines) + '\n'
            await async_write_text(filename, content)
            
            logger.info(f"✅ OpenAI fine-tuning data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export OpenAI fine-tuning data: {e}")
            raise


# Factory function
def create_training_data_generator(max_concurrent: int = 50, questions_per_topic: int = 10) -> TrainingDataGenerator:
    """Create a configured training data generator"""
    return TrainingDataGenerator(max_concurrent=max_concurrent, questions_per_topic=questions_per_topic) 