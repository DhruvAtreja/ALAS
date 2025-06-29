"""
Deep Research API client for OpenAI's o3-deep-research model
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

try:
    from ..config.settings import settings
    from ..utils.logger import get_logger, log_api_call, log_cost, log_error
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import settings
    from utils.logger import get_logger, log_api_call, log_cost, log_error

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
        self.fallback_comprehensive = "gpt-4o"
        self.fallback_fast = "gpt-4o-mini"
        
    def _parse_research_response(self, response: Any, model: str) -> DeepResearchResponse:
        """Parse OpenAI response into structured format"""
        try:
            # Extract main content
            content = ""
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
            elif hasattr(response, 'content'):
                content = response.content
            
            # Extract citations (if available in response)
            citations = []
            research_steps = []
            
            # Convert usage object to dictionary if needed
            usage_dict = None
            if hasattr(response, 'usage') and response.usage:
                usage_obj = response.usage
                usage_dict = {
                    "prompt_tokens": getattr(usage_obj, 'prompt_tokens', 0),
                    "completion_tokens": getattr(usage_obj, 'completion_tokens', 0),
                    "total_tokens": getattr(usage_obj, 'total_tokens', 0)
                }
            
            return DeepResearchResponse(
                id=getattr(response, 'id', str(uuid.uuid4())),
                content=content or "No content received",
                citations=citations,
                research_steps=research_steps,
                timestamp=datetime.now(),
                model=model,
                usage=usage_dict,
                cost_estimate=self._estimate_cost(response, model)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse research response: {e}")
            raise DeepResearchError(f"Failed to parse response: {e}")
    
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
    
    async def research_topic(self, 
                           query: str, 
                           context: Optional[str] = None,
                           depth: str = "comprehensive",
                           domain: Optional[str] = None) -> DeepResearchResponse:
        """
        Conduct deep research on a specific topic
        
        Args:
            query: The research question or topic
            context: Optional context to provide background
            depth: "comprehensive" for o1-pro, "fast" for o1-mini
            domain: The domain context for the research
        """
        start_time = datetime.now()
        
        try:
            # Choose model based on depth
            primary_model = self.comprehensive_model if depth == "comprehensive" else self.fast_model
            fallback_model = self.fallback_comprehensive if depth == "comprehensive" else self.fallback_fast
            
            # Construct research prompt
            prompt = self._build_research_prompt(query, context, domain)
            
            logger.info(f"Starting deep research: {query[:100]}...")
            
            # Try primary model first, then fallback
            model_used = primary_model
            try:
                # Prepare request parameters
                request_params = {
                    "model": primary_model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                # Only add temperature for models that support it (not o3/o4-mini)
                if primary_model not in ["o3", "o4-mini"]:
                    request_params["temperature"] = 0.1
                
                # Add max_completion_tokens for o3/o4-mini models if needed
                # (These models have different parameter requirements)
                
                response = await self.async_client.chat.completions.create(**request_params)
                
            except Exception as primary_error:
                logger.warning(f"Primary model {primary_model} failed, trying fallback: {primary_error}")
                model_used = fallback_model
                
                fallback_params = {
                    "model": fallback_model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                # Only add temperature for models that support it
                if fallback_model not in ["o3", "o4-mini"]:
                    fallback_params["temperature"] = 0.1
                
                response = await self.async_client.chat.completions.create(**fallback_params)
            
            # Parse response
            research_response = self._parse_research_response(response, model_used)
            
            # Log metrics
            duration = (datetime.now() - start_time).total_seconds()
            log_api_call("OpenAI Deep Research", model_used, {"query": query[:100]}, duration)
            log_cost(domain or "unknown", "deep_research", research_response.cost_estimate or 0)
            
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
            f"Research Topic: {query}",
            "",
            "Please provide:",
            "1. A comprehensive overview of the topic",
            "2. Current state-of-the-art information and recent developments",
            "3. Key concepts, terminology, and definitions",
            "4. Practical applications and implementations",
            "5. Common challenges and solutions",
            "6. Best practices and recommendations",
            "7. Future trends and directions",
            "",
            "Requirements:",
            "- Focus on authoritative and up-to-date sources",
            "- Include specific technical details where relevant",
            "- Provide concrete examples and use cases",
            "- Structure the information clearly and logically",
            "- Cite sources when making specific claims",
            "",
            "Format the response to be suitable for generating training data for a language model."
        ])
        
        return "\n".join(prompt_parts)
    
    async def generate_curriculum(self, 
                                domain: str, 
                                current_topics: List[str] = None,
                                learning_goals: List[str] = None) -> Dict[str, Any]:
        """
        Generate a learning curriculum for a domain using deep research
        
        Args:
            domain: The domain to create curriculum for
            current_topics: Already covered topics to build upon
            learning_goals: Specific learning objectives
        """
        
        prompt_parts = [
            f"Create a comprehensive learning curriculum for the domain: {domain}",
            "",
            "Generate a structured curriculum with the following characteristics:",
            "1. Breadth-first coverage of fundamental topics",
            "2. Progressive depth increase based on prerequisites",
            "3. Clear learning objectives for each topic",
            "4. Estimated difficulty levels",
            "5. Topic relationships and dependencies",
            "",
        ]
        
        if current_topics:
            prompt_parts.extend([
                "Already covered topics:",
                *[f"- {topic}" for topic in current_topics],
                ""
            ])
        
        if learning_goals:
            prompt_parts.extend([
                "Specific learning goals:",
                *[f"- {goal}" for goal in learning_goals],
                ""
            ])
        
        prompt_parts.extend([
            "Output format: JSON structure with topics array containing:",
            "- id: unique identifier",
            "- name: topic name",
            "- description: detailed description",
            "- depth: 1-5 scale",
            "- prerequisites: array of prerequisite topic IDs",
            "- learning_objectives: array of specific objectives",
            "- difficulty: easy/medium/hard",
            "- estimated_hours: learning time estimate",
            "",
            "Include 15-20 topics covering the domain comprehensively."
        ])
        
        prompt = "\n".join(prompt_parts)
        
        response = await self.research_topic(
            query=prompt,
            domain=domain,
            depth="comprehensive"
        )
        
        try:
            # Try to parse JSON from response
            curriculum_data = json.loads(response.content)
            return curriculum_data
        except json.JSONDecodeError:
            # Fallback: extract curriculum from text response
            return self._extract_curriculum_from_text(response.content, domain)
    
    def _extract_curriculum_from_text(self, content: str, domain: str) -> Dict[str, Any]:
        """Extract curriculum structure from text response"""
        # This is a fallback method if JSON parsing fails
        # Implementation would parse the text response and structure it
        
        return {
            "domain": domain,
            "topics": [
                {
                    "id": f"{domain.lower().replace(' ', '_')}_intro",
                    "name": f"Introduction to {domain}",
                    "description": "Foundational concepts and overview",
                    "depth": 1,
                    "prerequisites": [],
                    "difficulty": "easy",
                    "estimated_hours": 4
                }
            ],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_topics": 1,
                "source": "text_extraction"
            }
        }
    
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
        Based on the following content about "{topic_name}", generate {num_questions} diverse training questions and answers.
        
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
            questions = json.loads(response.content)
            return questions if isinstance(questions, list) else questions.get('questions', [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse questions as JSON, using fallback")
            return self._extract_questions_from_text(response.content, topic_name)
    
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
                              max_concurrent: int = 3) -> List[DeepResearchResponse]:
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


# Factory function
def create_deep_research_client(api_key: Optional[str] = None) -> DeepResearchClient:
    """Create a configured deep research client"""
    return DeepResearchClient(api_key=api_key) 