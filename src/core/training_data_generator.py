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
    
    def __init__(self, max_concurrent: int = 50, questions_per_topic: int = 20):
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
                
                # Make API call with retry logic for XML parsing failures
                start_time = datetime.now()
                questions = []
                response = None
                retry_count = 0
                max_retries = 1
                
                while retry_count <= max_retries:
                    try:
                        response = await self.client.research_topic(
                            query=prompt,
                            domain=topic.name,
                            depth="comprehensive"
                        )
                        
                        # Parse questions from response
                        questions = self._parse_questions_from_response(response, topic)
                        
                        # If we got questions, break out of retry loop
                        if questions:
                            break
                        
                        # If no questions and this is our first attempt, retry once
                        if retry_count == 0:
                            logger.warning(f"XML parsing failed for {topic.name}, retrying once...")
                            retry_count += 1
                            continue
                        else:
                            logger.error(f"XML parsing failed for {topic.name} after retry, giving up")
                            break
                            
                    except Exception as parse_error:
                        logger.error(f"API call failed for {topic.name} on attempt {retry_count + 1}: {parse_error}")
                        if retry_count == 0:
                            retry_count += 1
                            continue
                        else:
                            raise parse_error
                
                duration = (datetime.now() - start_time).total_seconds()
                
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
                        "topic_depth": topic.depth,
                        "retry_count": retry_count
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

Generate questions across these categories:
1. Factual Recall (definitions, terminology, basic facts)
2. Conceptual Understanding (explanations, relationships, principles)  
3. Application (problem-solving, implementation, real-world use)
4. Analysis (comparison, evaluation, critical thinking)
5. Synthesis (combining concepts, creative solutions, design)

Note: Include code examples wherever you can in a <code> tag.

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
- Make questions/answers suitable for fine-tuning a language model
- You must follow the format of the XML tags and the structure of the questions.
"""

        return prompt
    
    def _parse_questions_from_response(self, response: str, topic: CurriculumTopic) -> List[TrainingQuestion]:
        """Parse training questions from XML response with improved error handling"""
        
        try:
            # Extract XML content
            xml_match = re.search(r'<questions>(.*?)</questions>', response, re.DOTALL)
            if not xml_match:
                logger.warning(f"No <questions> tags found in response for {topic.name}")
                return []
            
            xml_content = f"<questions>{xml_match.group(1)}</questions>"
            
            # Improved XML cleaning - only escape unescaped ampersands
            xml_content = self._clean_xml_content(xml_content)
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            questions = []
            question_elements = [elem for elem in root if elem.tag.startswith('question')]
            
            for i, question_elem in enumerate(question_elements, 1):
                try:
                    # Extract question data with better text handling
                    text_elem = question_elem.find('text')
                    answer_elem = question_elem.find('answer')
                    category_elem = question_elem.find('category')
                    difficulty_elem = question_elem.find('difficulty')
                    explanation_elem = question_elem.find('explanation')
                    
                    if text_elem is None or answer_elem is None:
                        logger.warning(f"Missing required fields in question {i} for {topic.name}")
                        continue
                    
                    # Extract text content including nested elements
                    question_text = self._extract_element_text(text_elem)
                    answer_text = self._extract_element_text(answer_elem)
                    
                    # Skip if critical content is missing
                    if not question_text.strip() or not answer_text.strip():
                        logger.warning(f"Empty question or answer in question {i} for {topic.name}")
                        continue
                    
                    question = TrainingQuestion(
                        id=f"{topic.id}_q{i:03d}",
                        topic_id=topic.id,
                        question=question_text.strip(),
                        answer=answer_text.strip(),
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
            # Try fallback parsing
            return self._fallback_parse_questions(response, topic)
        except Exception as e:
            logger.error(f"Error parsing questions for {topic.name}: {e}")
            return []
    
    def _clean_xml_content(self, xml_content: str) -> str:
        """Clean XML content more intelligently"""
        
        # Remove control characters
        xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)
        
        # Only escape unescaped ampersands (not already part of entities)
        # This regex finds & that are not followed by valid entity patterns
        xml_content = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', xml_content)
        
        # Fix common XML structure issues
        xml_content = self._fix_common_xml_issues(xml_content)
        
        return xml_content
    
    def _fix_common_xml_issues(self, xml_content: str) -> str:
        """Fix common XML structure issues"""
        
        # Fix unclosed <code> tags by ensuring they're properly paired
        # Count opening and closing code tags
        open_code_count = len(re.findall(r'<code[^>]*>', xml_content))
        close_code_count = len(re.findall(r'</code>', xml_content))
        
        # If there are more opening tags than closing, add missing closing tags
        if open_code_count > close_code_count:
            missing_closes = open_code_count - close_code_count
            # Add them at the end before the last </questions> tag
            xml_content = xml_content.replace('</questions>', '</code>' * missing_closes + '</questions>')
        
        # Fix other common issues
        # Remove any stray < or > that aren't part of tags
        xml_content = re.sub(r'(?<![<>])<(?![/!?a-zA-Z])', '&lt;', xml_content)
        xml_content = re.sub(r'(?<![a-zA-Z0-9/\-"\s])>(?![<>])', '&gt;', xml_content)
        
        return xml_content
    
    def _extract_element_text(self, element) -> str:
        """Extract text content from XML element, including nested elements"""
        if element is None:
            return ""
        
        # Get all text content including from nested elements
        text_parts = []
        
        # Add element's direct text
        if element.text:
            text_parts.append(element.text)
        
        # Add text from nested elements
        for child in element:
            if child.tag == 'code':
                # Handle code blocks specially
                code_text = child.text or ""
                text_parts.append(f"`{code_text}`")
            else:
                # For other nested elements, just get their text
                child_text = child.text or ""
                if child_text.strip():
                    text_parts.append(child_text)
            
            # Add tail text after the child element
            if child.tail:
                text_parts.append(child.tail)
        
        return " ".join(text_parts)
    
    def _fallback_parse_questions(self, response: str, topic: CurriculumTopic) -> List[TrainingQuestion]:
        """Fallback parsing method using regex when XML parsing fails"""
        
        logger.info(f"Attempting fallback parsing for {topic.name}")
        
        questions = []
        
        # Use regex to find question blocks
        question_pattern = r'<question-(\d+)>(.*?)</question-\d+>'
        question_matches = re.findall(question_pattern, response, re.DOTALL)
        
        for i, (question_num, question_content) in enumerate(question_matches, 1):
            try:
                # Extract individual fields using regex
                text_match = re.search(r'<text>(.*?)</text>', question_content, re.DOTALL)
                answer_match = re.search(r'<answer>(.*?)</answer>', question_content, re.DOTALL)
                category_match = re.search(r'<category>(.*?)</category>', question_content, re.DOTALL)
                difficulty_match = re.search(r'<difficulty>(.*?)</difficulty>', question_content, re.DOTALL)
                explanation_match = re.search(r'<explanation>(.*?)</explanation>', question_content, re.DOTALL)
                
                if not text_match or not answer_match:
                    logger.warning(f"Missing required fields in fallback question {i} for {topic.name}")
                    continue
                
                # Clean the extracted text
                question_text = self._clean_extracted_text(text_match.group(1))
                answer_text = self._clean_extracted_text(answer_match.group(1))
                
                if not question_text.strip() or not answer_text.strip():
                    logger.warning(f"Empty question or answer in fallback question {i} for {topic.name}")
                    continue
                
                question = TrainingQuestion(
                    id=f"{topic.id}_q{i:03d}",
                    topic_id=topic.id,
                    question=question_text.strip(),
                    answer=answer_text.strip(),
                    category=category_match.group(1).strip() if category_match else "Conceptual Understanding",
                    difficulty=difficulty_match.group(1).strip() if difficulty_match else topic.difficulty.value,
                    explanation=explanation_match.group(1).strip() if explanation_match else None,
                    source_topic=topic.name
                )
                
                questions.append(question)
                
            except Exception as e:
                logger.warning(f"Failed to parse fallback question {i} for {topic.name}: {e}")
                continue
        
        logger.info(f"Fallback parsing extracted {len(questions)} questions for {topic.name}")
        return questions
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean text extracted from XML"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&apos;', "'")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
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
    
    def _create_safe_filename(self, domain: str, timestamp: str, file_type: str = "training_data") -> str:
        """Create a safe filename from domain name"""
        import hashlib
        
        # Clean domain: remove problematic characters and limit length
        clean_domain = re.sub(r'[^\w\s-]', '', domain)  # Remove non-alphanumeric chars except spaces and hyphens
        clean_domain = re.sub(r'\s+', '_', clean_domain)  # Replace spaces with underscores
        clean_domain = clean_domain.lower().strip('_')  # Convert to lowercase and strip edge underscores
        
        # If domain is still too long, truncate and add hash
        if len(clean_domain) > 30:
            # Use first 30 chars + hash of full domain
            domain_hash = hashlib.md5(domain.encode()).hexdigest()[:8]
            clean_domain = clean_domain[:20] + "_" + domain_hash
        
        # Create filename ensuring it's under filesystem limits
        filename = f"{file_type}_{clean_domain}_{timestamp}.json"
        
        # Final safety check - if still too long, use hash only
        if len(filename) > 100:
            domain_hash = hashlib.md5(domain.encode()).hexdigest()[:16]
            filename = f"{file_type}_{domain_hash}_{timestamp}.json"
        
        return filename

    async def save_training_data(self, training_data: CurriculumTrainingData, filename: Optional[str] = None) -> str:
        """Save training data to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self._create_safe_filename(training_data.domain, timestamp, "training_data")
        
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
            filename = self._create_safe_filename(training_data.domain, timestamp, "openai_training").replace('.json', '.jsonl')
        
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
def create_training_data_generator(max_concurrent: int = 50, questions_per_topic: int = 20) -> TrainingDataGenerator:
    """Create a configured training data generator"""
    return TrainingDataGenerator(max_concurrent=max_concurrent, questions_per_topic=questions_per_topic) 