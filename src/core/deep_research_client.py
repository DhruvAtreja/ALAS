"""
Deep Research API client for OpenAI's o3-deep-research model
"""

import asyncio
import json
import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
from pathlib import Path

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from enum import Enum
from langsmith import traceable

try:
    from ..config.settings import settings
    from ..utils.logger import get_logger, log_api_call, log_cost, log_error
    from ..utils.async_file_utils import async_read_json, async_write_json
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import settings
    from utils.logger import get_logger, log_api_call, log_cost, log_error
    from utils.async_file_utils import async_read_json, async_write_json

logger = get_logger(__name__)


class ResearchCitation(BaseModel):
    """Represents a citation from research results"""
    title: str
    url: str
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    snippet: Optional[str] = None


class ResearchStep(BaseModel):
    """Represents a step in the research process"""
    type: str  # "search", "code_execution", "analysis"
    action: Optional[str] = None
    query: Optional[str] = None
    result: Optional[str] = None


class DeepResearchResponse(BaseModel):
    """Structured response from Deep Research API"""
    id: str
    content: str
    citations: List[ResearchCitation]
    research_steps: List[ResearchStep]
    timestamp: datetime
    model: str
    usage: Optional[Dict[str, Any]] = None
    cost_estimate: Optional[float] = None


class DeepResearchError(Exception):
    """Custom exception for Deep Research API errors"""
    pass


class DifficultyLevel(str, Enum):
    """Difficulty levels for learning topics"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TopicPriority(str, Enum):
    """Priority levels for learning topics"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CurriculumTopic(BaseModel):
    """Represents a single topic in a learning curriculum"""
    id: str = Field(..., description="Unique identifier for the topic")
    name: str = Field(..., description="Display name of the topic")
    description: str = Field(..., description="Detailed description/summary of the topic")
    prerequisites: List[str] = Field(default_factory=list, description="List of prerequisite topics")
    learning_objectives: str = Field(..., description="What the learner should achieve")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Difficulty level")
    depth: int = Field(default=1, ge=1, le=5, description="Topic depth level (1-5)")
    priority: TopicPriority = Field(default=TopicPriority.MEDIUM, description="Learning priority")


class CurriculumDifficultyStats(BaseModel):
    """Statistics about curriculum difficulty distribution"""
    easy: int = Field(default=0, ge=0, description="Number of easy topics")
    medium: int = Field(default=0, ge=0, description="Number of medium topics")
    hard: int = Field(default=0, ge=0, description="Number of hard topics")


class CurriculumMetadata(BaseModel):
    """Metadata about the generated curriculum"""
    generated_at: str = Field(..., description="ISO timestamp when curriculum was generated")
    total_topics: int = Field(..., ge=0, description="Total number of topics")
    source: str = Field(..., description="Source of curriculum generation (xml_extraction, fallback)")
    difficulties: CurriculumDifficultyStats = Field(default_factory=CurriculumDifficultyStats)
    note: Optional[str] = Field(default=None, description="Additional notes about generation")


class Curriculum(BaseModel):
    """Complete curriculum structure"""
    domain: str = Field(..., description="The learning domain")
    topics: List[CurriculumTopic] = Field(..., description="List of curriculum topics")
    metadata: CurriculumMetadata = Field(..., description="Curriculum metadata")


class LearnedTopic(BaseModel):
    """Represents a topic that has been learned/mastered"""
    topic_name: str = Field(..., description="Name of the learned topic")
    learned_date: str = Field(..., description="ISO timestamp when topic was mastered")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy achieved on this topic")
    iteration: int = Field(..., ge=1, description="Learning iteration when mastered")


class LearnedTopicsHistory(BaseModel):
    """Historical record of all learned topics"""
    domain: str = Field(..., description="The learning domain")
    learned_topics: List[LearnedTopic] = Field(default_factory=list, description="List of all learned topics")
    last_updated: str = Field(..., description="ISO timestamp of last update")
    total_topics_learned: int = Field(default=0, description="Total count of learned topics")


class CurriculumRevisionRequest(BaseModel):
    """Request for curriculum revision based on evaluation results"""
    domain: str = Field(..., description="The learning domain")
    mastered_topics: List[str] = Field(default_factory=list, description="Topics with >90% accuracy")
    failed_topics: List[str] = Field(default_factory=list, description="Topics with <90% accuracy")
    failed_questions: List[Dict[str, Any]] = Field(default_factory=list, description="Failed questions with explanations")
    total_questions_analyzed: int = Field(default=0, description="Total questions analyzed")
    overall_accuracy: float = Field(default=0.0, description="Overall accuracy across all topics")


class CurriculumRevisionResult(BaseModel):
    """Result of curriculum revision"""
    original_curriculum: Optional[Curriculum] = Field(default=None, description="Original curriculum if provided")
    revised_curriculum: Optional[Curriculum] = Field(default=None, description="New/revised curriculum")
    revision_summary: str = Field(default="", description="Summary of revision changes")
    mastered_topics: List[str] = Field(default_factory=list, description="Topics mastered")
    failed_topics: List[str] = Field(default_factory=list, description="Topics that need improvement")
    failed_questions_count: int = Field(default=0, description="Number of failed questions analyzed")


class DeepResearchClient:
    """Client for OpenAI's Deep Research API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai.api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Deep Research")
            
        self.client = OpenAI(api_key=self.api_key, timeout=settings.openai.timeout)
        self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=settings.openai.timeout)
        
        # Use correct Deep Research models from OpenAI documentation
        self.comprehensive_model = "o3"          # For deep, comprehensive research  
        self.fast_model = "o4-mini"              # For quicker research tasks
        
        # Fallback to GPT-4 models if Deep Research models aren't available
        self.fallback_comprehensive = "gpt-4.1-2025-04-14"
        self.fallback_fast = "gpt-4.1-2025-04-14"
        
        # Path for learned topics history
        self.learned_topics_file = Path("learned_topics.json")

    def _estimate_cost(self, response: Any, model: str) -> float:
        """Estimate cost based on token usage"""
        try:
            if hasattr(response, 'usage') and response.usage:
                # These are estimated rates - actual rates may vary
                if model == "o3":
                    input_rate = 0.015  # per 1K tokens (estimated for o3)
                    output_rate = 0.060  # per 1K tokens (estimated for o3)
                elif model == "o4-mini":
                    input_rate = 0.003  # per 1K tokens (estimated for o4-mini)
                    output_rate = 0.012  # per 1K tokens (estimated for o4-mini)
                elif model == "gpt-4o":
                    input_rate = 0.0025  # per 1K tokens
                    output_rate = 0.01   # per 1K tokens
                else:  # gpt-4o-mini fallback
                    input_rate = 0.00015  # per 1K tokens
                    output_rate = 0.0006   # per 1K tokens
                
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                
                cost = (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)
                return round(cost, 4)
                
        except Exception as e:
            logger.debug(f"Could not calculate cost: {e}")
        
        # Fallback estimate for deep research
        return 0.50 if model == "o3" else 0.10

    async def save_learned_topics(self, domain: str, mastered_topics: List[str], accuracy_scores: Dict[str, float], iteration: int) -> None:
        """Save newly mastered topics to the learned topics history file"""
        try:
            current_time = datetime.now().isoformat()
            
            # Load existing history or create new
            history = await self.load_learned_topics_history(domain)
            
            # Add new learned topics
            for topic_name in mastered_topics:
                accuracy = accuracy_scores.get(topic_name, 1.0)  # Default to 1.0 if not found
                
                # Check if topic is already in history (avoid duplicates)
                existing_topic = next(
                    (lt for lt in history.learned_topics if lt.topic_name == topic_name), 
                    None
                )
                
                if not existing_topic:
                    learned_topic = LearnedTopic(
                        topic_name=topic_name,
                        learned_date=current_time,
                        accuracy=accuracy,
                        iteration=iteration
                    )
                    history.learned_topics.append(learned_topic)
                    logger.info(f"Added learned topic: {topic_name} (accuracy: {accuracy:.1%})")
                else:
                    # Update existing topic if accuracy improved
                    if accuracy > existing_topic.accuracy:
                        existing_topic.accuracy = accuracy
                        existing_topic.learned_date = current_time
                        existing_topic.iteration = iteration
                        logger.info(f"Updated learned topic: {topic_name} (new accuracy: {accuracy:.1%})")
            
            # Update metadata
            history.last_updated = current_time
            history.total_topics_learned = len(history.learned_topics)
            
            # Save to file
            await async_write_json(self.learned_topics_file, history.model_dump())
            
            logger.info(f"Saved {len(mastered_topics)} learned topics to {self.learned_topics_file}")
            logger.info(f"Total topics learned so far: {history.total_topics_learned}")
            
        except Exception as e:
            logger.error(f"Failed to save learned topics: {e}")

    async def load_learned_topics_history(self, domain: str) -> LearnedTopicsHistory:
        """Load the learned topics history from file"""
        try:
            if self.learned_topics_file.exists():
                data = await async_read_json(self.learned_topics_file)
                
                # Validate domain matches
                if data.get("domain") == domain:
                    return LearnedTopicsHistory(**data)
                else:
                    logger.warning(f"Domain mismatch in learned topics file. Expected: {domain}, Found: {data.get('domain')}")
                    # Create new history for different domain
                    return self._create_new_learned_topics_history(domain)
            else:
                logger.info(f"No learned topics history found, creating new file")
                return self._create_new_learned_topics_history(domain)
                
        except Exception as e:
            logger.error(f"Failed to load learned topics history: {e}")
            return self._create_new_learned_topics_history(domain)
    
    def _create_new_learned_topics_history(self, domain: str) -> LearnedTopicsHistory:
        """Create a new learned topics history"""
        return LearnedTopicsHistory(
            domain=domain,
            learned_topics=[],
            last_updated=datetime.now().isoformat(),
            total_topics_learned=0
        )
    
    async def get_all_learned_topic_names(self, domain: str) -> List[str]:
        """Get a list of all historically learned topic names"""
        history = await self.load_learned_topics_history(domain)
        return [topic.topic_name for topic in history.learned_topics]
    
    @traceable
    async def research_topic(self, 
                           query: str, 
                           context: Optional[str] = None,
                           depth: str = "comprehensive",
                           domain: Optional[str] = None) -> str:
        """
        Conduct deep research on a specific topic
        
        Args:
            query: The research question or topic
            context: Optional context to provide background
            depth: "comprehensive" for o3, "fast" for o4-mini
            domain: The domain context for the research
        """
        start_time = datetime.now()
        
        try:
            # Choose model based on depth
            primary_model = self.comprehensive_model if depth == "comprehensive" else self.fast_model
            
            # Construct research prompt            
            logger.info(f"Starting deep research: {query}")
            
            # Try primary model first, then fallback
            model_used = primary_model
            try:
                # Prepare request parameters
                request_params = {
                    "model": primary_model,
                    "messages": [{"role": "user", "content": query}]
                }
                
                # Only add temperature for models that support it (not o3/o4-mini)
                if primary_model not in ["o3", "o4-mini"]:
                    request_params["temperature"] = 0.1
                
                # Add max_completion_tokens for o3/o4-mini models if needed
                # (These models have different parameter requirements)
                
                response = await self.async_client.responses.create(
                model="o3-deep-research",
                input=query,
                tools=[
                    {"type": "web_search_preview"},
                ],
                max_output_tokens=100000,
                )
                
            except Exception as primary_error:
                logger.warning(f"Primary model {primary_model} failed, trying fallback: {primary_error}")
                         
                response = await self.async_client.responses.create(
                    model="o4-mini-deep-research",
                    input=query,
                    tools=[
                        {"type": "web_search_preview"},
                    ],
                    max_output_tokens=100000,
                )
            
            # Parse response
            research_response = response.output_text
            
            # Log metrics
            duration = (datetime.now() - start_time).total_seconds()
            log_api_call("OpenAI Deep Research", model_used, {"query": query[:100]}, duration)
            
            logger.info(f"Research completed in {duration:.2f}s using {model_used}")
            return research_response
            
        except Exception as e:
            log_error(e, {"query": query, "model": "unknown"})
            raise DeepResearchError(f"Research failed: {e}")
    
    def _build_research_prompt(self, 
                              query: str, 
                              context: Optional[str] = None,
                              domain: Optional[str] = None) -> str:
        """Build a comprehensive research prompt"""
        
        prompt_parts = [
            "Conduct comprehensive research on the following topic and provide a detailed analysis."
        ]
        
        if domain:
            prompt_parts.append(f"Domain context: {domain}")
        
        if context:
            prompt_parts.append(f"Background context: {context}")
        
        prompt_parts.extend([
            f"{query}",
        ])
        
        return "\n".join(prompt_parts)
    
    @traceable
    async def generate_curriculum(self, 
                                domain: str, 
                                current_topics: Optional[List[str]] = None,
                                learning_goals: Optional[List[str]] = None) -> Optional[Curriculum]:
        """
        Generate a learning curriculum for a domain using deep research
        
        Args:
            domain: The domain to create curriculum for
            current_topics: Already covered topics to build upon
            learning_goals: Specific learning objectives
        """
        
        prompt_parts = [
            f"Create a comprehensive learning curriculum for the domain: <domain>{domain}</domain>",
            "",
            "Generate a structured curriculum as a list of topics and a quick summary of the topic with the following characteristics:",
            "1. Coverage of fundamental topics",
            "2. Clear learning objectives for each topic",
            "3. Estimated difficulty levels (easy/medium/hard)",
            "4. You must make at least 10 topics.",
            "5. You must make sure the user has a good understanding of the domain. The topics must not be too broad. You can create as many topics as needed. You can also add topics which are inter-related, but in this case, you must include in their summaries how these topics are related.",
        ]
        if current_topics:
            prompt_parts.extend([
                "Already covered topics by the user:",
                *[f"- <topic>{topic}</topic>" for topic in current_topics],
                "You can create new topics which add depth to these topics.",
                ""
            ])
        
        if learning_goals:
            prompt_parts.extend([
                "Specific learning goals by the user:",
                *[f"- <goal>{goal}</goal>" for goal in learning_goals],
                ""
            ])

        prompt_parts.extend([            "Your answer must be in the following format.",
            '''<curriculum>
            <topic-1>
            <name>Topic 1</name>
            <summary>Summary of the topic</summary>
            <prerequisites>Prerequisites of the topic</prerequisites>
            <learning_objectives>Learning objectives of the topic</learning_objectives>
            <difficulty>Difficulty of the topic (easy/medium/hard)</difficulty>
            </topic-1>
            <topic-2>
            <name>Topic 2</name>
            <summary>Summary of the topic</summary>
            <prerequisites>Prerequisites of the topic</prerequisites>
            <learning_objectives>Learning objectives of the topic</learning_objectives>
            <difficulty>Difficulty of the topic (easy/medium/hard)</difficulty>
            </topic-2>
            ...
            </curriculum>
            ''',
            "For example, if the user is learning about Web Development, you can create a curriculum like this:",
            '''<curriculum>
            <topic-1>
            <name>Web Development</name>
            <summary>Web development is the process of creating websites and web applications.</summary>
            <prerequisites>None</prerequisites>
            <learning_objectives>Learn the basic concepts of web development. Learn about the different types of websites and web applications. Learn about the different technologies used in web development. Learn about the history of web development. Get an understanding of the different roles in web development. Learn about why web development is important.</learning_objectives>
            <difficulty>Easy</difficulty>
            </topic-1>
            <topic-2>
            <name>HTML</name>
            <summary>HTML is the standard markup language for creating web pages.</summary>
            <prerequisites>None</prerequisites>
            <learning_objectives>Learn what is HTML and why HTML is used. Learn the basic structure of HTML documents. Learn about the different tags in HTML. Learn about the different attributes in HTML. Learn about the different elements in HTML. See concrete examples of HTML code. Understand best practices for writing HTML code. What is the current state of usage of HTML?</learning_objectives>
            <difficulty>Easy</difficulty>
            </topic-2>
            ...
            ... 30 more topics
            <topic-31>
            <name>Server side programming</name>
            <summary>Server side programming is the process of creating the server side of a website or web application.</summary>
            <prerequisites>None</prerequisites>
            <learning_objectives>Learn what is server side programming and why server side programming is used. Learn the basic structure of server side programming. Learn about the different languages used in server side programming. Learn about the different frameworks used in server side programming. Learn about the different databases used in server side programming. See concrete examples of server side programming code. Understand best practices for writing server side programming code. What is the current state of usage of server side programming?</learning_objectives>
            <difficulty>Medium</difficulty>
            </topic-31>
            ...
            </curriculum>
            ''',
        ])
        
        prompt = "\n".join(prompt_parts)
        
        response = await self.research_topic(
            query=prompt,
            domain=domain,
            depth="comprehensive"
        )
        
        try:
            # Extract curriculum from XML response
            extracted_curriculum = self._extract_curriculum_from_text(response, domain)
            return extracted_curriculum
        except Exception as e:
            logger.error(f"Failed to parse curriculum response: {e}")
            logger.debug(f"Response content: {response[:500]}...")
            return None
    
    def _extract_curriculum_from_text(self, content: str, domain: str) -> Curriculum:
        """Extract curriculum structure from XML response"""
        try:
            # Extract XML content from the response
            xml_match = re.search(r'<curriculum>(.*?)</curriculum>', content, re.DOTALL)
            if not xml_match:
                raise ValueError("No <curriculum> tags found in response")
            
            xml_content = f"<curriculum>{xml_match.group(1)}</curriculum>"
            
            # Clean up the XML content - remove any unwanted characters that might cause parsing issues
            xml_content = xml_content.replace('&', '&amp;')  # Escape ampersands
            xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)  # Remove control chars
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            topics = []
            
            # Find all topic elements (they might be named topic-1, topic-2, etc.)
            topic_elements = root.findall('.//topic-*') or root.findall('.//topic')
            
            # If standard findall doesn't work, try a different approach
            if not topic_elements:
                # Find all child elements that start with 'topic'
                topic_elements = [elem for elem in root if elem.tag.startswith('topic')]
            
            for i, topic_elem in enumerate(topic_elements, 1):
                # Extract topic information
                name_elem = topic_elem.find('name')
                summary_elem = topic_elem.find('summary')
                prerequisites_elem = topic_elem.find('prerequisites')
                objectives_elem = topic_elem.find('learning_objectives')
                difficulty_elem = topic_elem.find('difficulty')
                
                # Handle missing elements gracefully
                name = name_elem.text.strip() if name_elem is not None and name_elem.text else f"Topic {i}"
                summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else "No summary provided"
                prerequisites_text = prerequisites_elem.text.strip() if prerequisites_elem is not None and prerequisites_elem.text else "None"
                objectives = objectives_elem.text.strip() if objectives_elem is not None and objectives_elem.text else "No objectives provided"
                difficulty = difficulty_elem.text.strip().lower() if difficulty_elem is not None and difficulty_elem.text else "medium"
                
                # Parse prerequisites
                prerequisites = []
                if prerequisites_text.lower() not in ["none", "no prerequisites", ""]:
                    # Split by common delimiters and clean up
                    prereq_items = re.split(r'[,;]\s*', prerequisites_text)
                    prerequisites = [p.strip() for p in prereq_items if p.strip()]
                
                # Create topic object
                topic = CurriculumTopic(
                    id=f"{domain.lower().replace(' ', '_')}_{i:03d}",
                    name=name,
                    description=summary,
                    prerequisites=prerequisites,
                    learning_objectives=objectives,
                    difficulty=DifficultyLevel(difficulty) if difficulty in ["easy", "medium", "hard"] else DifficultyLevel.MEDIUM,
                    depth=self._calculate_topic_depth(prerequisites),
                    priority=TopicPriority.HIGH if i <= 5 else TopicPriority.MEDIUM
                )
                
                topics.append(topic)
            
            if not topics:
                raise ValueError("No topics found in curriculum XML")
            
            # Create difficulty stats
            difficulty_stats = CurriculumDifficultyStats(
                easy=len([t for t in topics if t.difficulty == DifficultyLevel.EASY]),
                medium=len([t for t in topics if t.difficulty == DifficultyLevel.MEDIUM]),
                hard=len([t for t in topics if t.difficulty == DifficultyLevel.HARD])
            )
            
            # Create metadata
            metadata = CurriculumMetadata(
                generated_at=datetime.now().isoformat(),
                total_topics=len(topics),
                source="xml_extraction",
                difficulties=difficulty_stats
            )
            
            return Curriculum(
                domain=domain,
                topics=topics,
                metadata=metadata
            )
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise ValueError(f"Invalid XML structure: {e}")
        except Exception as e:
            logger.error(f"Error extracting curriculum: {e}")
            # Return a basic fallback curriculum
            raise e
    
    def _calculate_topic_depth(self, prerequisites: List[str]) -> int:
        """Calculate topic depth based on prerequisites"""
        if not prerequisites:
            return 1
        return min(len(prerequisites) + 1, 5)  # Cap at depth 5
   
    async def generate_training_questions(self, 
                                        topic_content: str,
                                        topic_name: str,
                                        num_questions: int = 20) -> List[Dict[str, Any]]:
        """
        Generate training questions and answers from topic content
        
        Args:
            topic_content: The researched content about the topic
            topic_name: Name of the topic
            num_questions: Number of questions to generate
        """
        
        prompt = f"""
        Based on the following content about "{topic_name}", generate {num_questions} diverse training questions and answers. Use your web search to make sure you have up to date information.
        
        Content:
        {topic_content}
        
        Generate questions across these categories:
        1. Factual Recall (definitions, terminology, facts)
        2. Conceptual Understanding (explanations, relationships)
        3. Application (problem-solving, implementation)
        4. Analysis (comparison, evaluation)
        5. Synthesis (combining concepts, design)
        
        For each question, provide:
        - question: Clear, specific question
        - answer: Comprehensive, accurate answer
        - category: One of the categories above
        - difficulty: easy/medium/hard
        - explanation: Why this answer is correct
        
        Format as JSON array of question objects.
        Ensure questions test deep understanding and are suitable for fine-tuning a language model.
        """
        
        response = await self.research_topic(
            query=prompt,
            domain=topic_name,
            depth="fast"  # Use faster model for question generation
        )
        
        try:
            questions = json.loads(response)
            return questions if isinstance(questions, list) else questions.get('questions', [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse questions as JSON, using fallback")
            return self._extract_questions_from_text(response, topic_name)
    
    def _extract_questions_from_text(self, content: str, topic_name: str) -> List[Dict[str, Any]]:
        """Extract questions from text response as fallback"""
        # Simplified fallback - in practice, you'd parse the text more thoroughly
        return [
            {
                "question": f"What are the key concepts in {topic_name}?",
                "answer": "This is a placeholder answer that would be extracted from the content.",
                "category": "Conceptual Understanding",
                "difficulty": "medium",
                "explanation": "Tests understanding of fundamental concepts"
            }
        ]
    
    async def parallel_research(self, 
                              queries: List[str],
                              domain: str,
                              max_concurrent: int = 50) -> List[DeepResearchResponse]:
        """
        Research multiple topics concurrently with rate limiting
        
        Args:
            queries: List of research queries
            domain: Domain context
            max_concurrent: Maximum concurrent requests
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def research_with_limit(query: str):
            async with semaphore:
                return await self.research_topic(query, domain=domain)
        
        logger.info(f"Starting parallel research for {len(queries)} topics")
        
        tasks = [research_with_limit(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Research failed for query {i}: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"Completed {len(successful_results)}/{len(queries)} research tasks")
        return successful_results


    @traceable
    async def generate_revised_curriculum_from_evaluation(self, 
                                                        evaluation_results: Dict[str, Any],
                                                        current_curriculum: Optional[Curriculum] = None,
                                                        accuracy_threshold: float = 0.9,
                                                        iteration: int = 1) -> Optional[CurriculumRevisionResult]:
        """
        Generate revised curriculum based on DPO evaluation results
        
        Args:
            evaluation_results: DPO evaluation results in expected format
            current_curriculum: Optional current curriculum for context
            accuracy_threshold: Threshold for determining mastery (default 0.9 for 90%)
            iteration: Current learning iteration number
        
        Returns:
            CurriculumRevisionResult containing revised curriculum and analysis
        """
        
        try:
            # Extract domain and basic info
            domain = evaluation_results.get("evaluation_results", {}).get("domain", "Unknown Domain")
            overall_accuracy = evaluation_results.get("evaluation_results", {}).get("overall_accuracy", 0.0)
            topic_results = evaluation_results.get("evaluation_results", {}).get("topic_results", [])
            
            # Analyze topic performance
            mastered_topics = []
            failed_topics = []
            failed_questions = []
            accuracy_scores = {}
            
            for topic_result in topic_results:
                topic_name = topic_result.get("topic_name", "Unknown Topic")
                topic_accuracy = topic_result.get("accuracy", 0.0)
                accuracy_scores[topic_name] = topic_accuracy
                
                if topic_accuracy >= accuracy_threshold:
                    mastered_topics.append(topic_name)
                else:
                    failed_topics.append(topic_name)
                    
                    # Extract failed questions from this topic
                    topic_failed_questions = self._extract_failed_questions(topic_result)
                    failed_questions.extend(topic_failed_questions)
            
            # Save newly mastered topics to learned topics history
            if mastered_topics:
                await self.save_learned_topics(domain, mastered_topics, accuracy_scores, iteration)
            
            # Load all historical learned topics
            all_learned_topics = await self.get_all_learned_topic_names(domain)
            
            # Limit failed questions to maximum 100 as requested
            if len(failed_questions) > 100:
                failed_questions = failed_questions[:100]
                logger.warning(f"Truncated failed questions to 100 (originally {len(failed_questions)})")
            
            # Generate curriculum revision prompt (including historical learned topics)
            revision_prompt = self._build_curriculum_revision_prompt(
                domain=domain,
                mastered_topics=mastered_topics,
                all_learned_topics=all_learned_topics,  # Include all historical topics
                failed_topics=failed_topics,
                failed_questions=failed_questions,
                overall_accuracy=overall_accuracy,
                current_curriculum=current_curriculum
            )
            
            # Call deep research API
            logger.info(f"Generating revised curriculum for domain: {domain}")
            logger.info(f"Current mastered topics: {len(mastered_topics)}, Failed topics: {len(failed_topics)}")
            logger.info(f"Total historical learned topics: {len(all_learned_topics)}")
            logger.info(f"Failed questions to analyze: {len(failed_questions)}")
            
            response = await self.research_topic(
                query=revision_prompt,
                domain=domain,
                depth="comprehensive"
            )
            
            # Parse the revised curriculum
            revised_curriculum = self._extract_curriculum_from_text(response, domain)
            
            # Create revision summary
            revision_summary = self._create_revision_summary(
                mastered_topics=mastered_topics,
                all_learned_topics=all_learned_topics,
                failed_topics=failed_topics,
                failed_questions_count=len(failed_questions),
                overall_accuracy=overall_accuracy
            )
            
            return CurriculumRevisionResult(
                original_curriculum=current_curriculum,
                revised_curriculum=revised_curriculum,
                revision_summary=revision_summary,
                mastered_topics=mastered_topics,
                failed_topics=failed_topics,
                failed_questions_count=len(failed_questions)
            )
            
        except Exception as e:
            logger.error(f"Failed to generate revised curriculum: {e}")
            return None
    
    def _extract_failed_questions(self, topic_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract failed questions from a topic result"""
        failed_questions = []
        
        results = topic_result.get("results", [])
        for result in results:
            if not result.get("is_correct", True):  # Default to True if not specified
                failed_question = {
                    "question_id": result.get("question_id", "unknown"),
                    "question": result.get("question", "Question not available"),
                    "model_answer": result.get("model_answer", "Model answer not available"),
                    "ideal_answer": result.get("ideal_answer", "Ideal answer not available"),
                    "explanation": result.get("explanation", "No explanation provided"),
                    "topic_name": topic_result.get("topic_name", "Unknown Topic"),
                    "category": result.get("category", "Unknown Category"),
                    "difficulty": result.get("difficulty", "Unknown Difficulty")
                }
                failed_questions.append(failed_question)
        
        return failed_questions
    
    def _build_curriculum_revision_prompt(self,
                                        domain: str,
                                        mastered_topics: List[str],
                                        all_learned_topics: List[str],
                                        failed_topics: List[str],
                                        failed_questions: List[Dict[str, Any]],
                                        overall_accuracy: float,
                                        current_curriculum: Optional[Curriculum] = None) -> str:
        """Build comprehensive prompt for curriculum revision"""
        
        prompt_parts = [
            f"Create a revised learning curriculum for the domain: <domain>{domain}</domain>",
            "",
            "## Current Learning Status:",
            f"- Overall accuracy: {overall_accuracy:.1%}",
            f"- Topics mastered in this iteration: {len(mastered_topics)}",
            f"- Topics needing improvement: {len(failed_topics)}",
            f"- Failed questions analyzed: {len(failed_questions)}",
            ""
        ]
        
        # Add ALL historical learned topics
        if all_learned_topics:
            prompt_parts.extend([
                "## ALL Topics Learned Throughout History:",
                "The learner has demonstrated proficiency in these topics across all learning iterations:",
                *[f"- <learned_topic>{topic}</learned_topic>" for topic in all_learned_topics],
                "",
                "**Important**: Do not include these topics in the new curriculum as they have already been mastered.",
                ""
            ])
        
        # Add current iteration mastered topics (subset of historical)
        if mastered_topics:
            prompt_parts.extend([
                "## Topics Mastered in Current Iteration:",
                "These topics were just mastered in the most recent evaluation:",
                *[f"- <current_mastered_topic>{topic}</current_mastered_topic>" for topic in mastered_topics],
                ""
            ])
        
        # Add failed topics and questions
        if failed_topics or failed_questions:
            prompt_parts.extend([
                "## Topics Needing Improvement:",
                "The learner struggled with these topics and questions:",
                ""
            ])
            
            if failed_topics:
                prompt_parts.extend([
                    "### Failed Topics:",
                    *[f"- <failed_topic>{topic}</failed_topic>" for topic in failed_topics],
                    ""
                ])
            
            if failed_questions:
                prompt_parts.extend([
                    "### Specific Questions That Were Answered Incorrectly:",
                    ""
                ])
                
                for i, question in enumerate(failed_questions, 1):
                    prompt_parts.extend([
                        f"**Question {i}:** {question['question']}",
                        f"**Topic:** {question['topic_name']}",
                        f"**Category:** {question['category']} | **Difficulty:** {question['difficulty']}",
                        f"**Model's Incorrect Answer:** {question['model_answer'][:200]}{'...' if len(question['model_answer']) > 200 else ''}",
                        f"**Correct Answer:** {question['ideal_answer'][:200]}{'...' if len(question['ideal_answer']) > 200 else ''}",
                        f"**Why the answer was wrong:** {question['explanation']}",
                        ""
                    ])
        
        # Add current curriculum context if available
        if current_curriculum:
            prompt_parts.extend([
                "## Current Curriculum Context:",
                f"The current curriculum has {len(current_curriculum.topics)} topics.",
                "Consider this context when creating the revised curriculum.",
                ""
            ])
        
        # Add revision instructions
        prompt_parts.extend([
            "## Curriculum Revision Instructions:",
            "",
            "Based on the learning performance analysis above, create a comprehensive revised curriculum that:",
            "",
            "1. **Avoids All Previously Learned Topics**: Do NOT include any topics from the historical learned topics list above.",
            "",
            "2. **Reinforces Failed Topics**: For topics where the learner struggled, create focused sub-topics that address the specific knowledge gaps revealed by the failed questions.",
            "",
            "3. **Addresses Specific Mistakes**: Design topics that directly address the misunderstandings shown in the failed questions above.",
            "",
            "4. **Builds on Historical Knowledge**: Since the learner has mastered many topics over time, create more advanced topics that build upon that accumulated foundation.",
            "",
            "5. **Expands Domain Knowledge**: Include new topics that represent the next logical progression in learning this domain.",
            "",
            "6. **Maintains Appropriate Difficulty**: Balance remedial topics (easy-medium) for failed areas with advanced topics (medium-hard) for progression.",
            "",
            "## Output Format Requirements:",
            "",
            "Your response must be in the following XML format:",
            "",
            '''<curriculum>
            <topic-1>
            <name>Topic Name</name>
            <summary>Detailed summary addressing specific knowledge gaps or building on mastered concepts</summary>
            <prerequisites>Prerequisites (reference learned topics where applicable)</prerequisites>
            <learning_objectives>Clear learning objectives that address identified weaknesses or advance knowledge</learning_objectives>
            <difficulty>easy/medium/hard</difficulty>
            </topic-1>
            <topic-2>
            <name>Topic Name</name>
            <summary>Detailed summary addressing specific knowledge gaps or building on mastered concepts</summary>
            <prerequisites>Prerequisites (reference learned topics where applicable)</prerequisites>
            <learning_objectives>Clear learning objectives that address identified weaknesses or advance knowledge</learning_objectives>
            <difficulty>easy/medium/hard</difficulty>
            </topic-2>
            ...
            </curriculum>''',
            "",
            "## Important Guidelines:",
            "",
            "- Create at least 10 topics for a comprehensive curriculum",
            "- **NEVER repeat any topic from the historical learned topics list**",
            "- For failed topics, create remedial topics that address the specific misunderstandings",
            "- For areas where the learner is strong, create advanced topics that build upon that knowledge", 
            "- Include a mix of difficulty levels appropriate for the learner's current level",
            "- Ensure topics are logically sequenced and build upon each other",
            "- Make summaries detailed enough to understand what knowledge gaps are being addressed",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_revision_summary(self,
                               mastered_topics: List[str],
                               all_learned_topics: List[str],
                               failed_topics: List[str],
                               failed_questions_count: int,
                               overall_accuracy: float) -> str:
        """Create a summary of the curriculum revision"""
        
        summary_parts = [
            f"Curriculum revision based on {overall_accuracy:.1%} overall accuracy:",
            f"- {len(mastered_topics)} topics mastered in current iteration",
            f"- {len(all_learned_topics)} total topics mastered historically",
            f"- {len(failed_topics)} topics need improvement",
            f"- {failed_questions_count} failed questions analyzed for knowledge gaps",
            "",
            "Revision approach:",
            "- Avoided all previously mastered topics from learning history",
            "- Reinforcement topics created for failed areas",
            "- Advanced topics created building on accumulated knowledge",
            "- Specific knowledge gaps addressed based on failed questions",
            "- Balanced difficulty progression maintained"
        ]
        
        return "\n".join(summary_parts)


# Factory function
def create_deep_research_client(api_key: Optional[str] = None) -> DeepResearchClient:
    """Create a configured deep research client"""
    return DeepResearchClient(api_key=api_key) 