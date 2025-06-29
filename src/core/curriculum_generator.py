"""
Curriculum Generator using Deep Research API
"""

import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from .deep_research_client import create_deep_research_client, DeepResearchError
    from ..workflows.state_management import Topic, TopicStatus
    from ..utils.logger import get_logger
    from ..config.settings import settings
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.deep_research_client import create_deep_research_client, DeepResearchError
    from workflows.state_management import Topic, TopicStatus
    from utils.logger import get_logger
    from config.settings import settings

logger = get_logger(__name__)


class CurriculumGenerator:
    """Generates learning curricula using OpenAI's Deep Research API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.research_client = create_deep_research_client(api_key)
        
    async def generate_initial_curriculum(self, 
                                        domain: str,
                                        learning_goals: Optional[List[str]] = None,
                                        breadth_topics: Optional[int] = None) -> List[Topic]:
        """
        Generate initial breadth-first curriculum for a domain
        
        Args:
            domain: The domain to create curriculum for
            learning_goals: Specific learning objectives  
            breadth_topics: Number of initial breadth topics
        """
        
        breadth_topics = breadth_topics or settings.learning.initial_topics_breadth
        
        logger.info(f"Generating initial curriculum for domain: {domain}")
        
        try:
            # Use deep research to generate comprehensive curriculum
            curriculum_data = await self.research_client.generate_curriculum(
                domain=domain,
                learning_goals=learning_goals
            )
            
            # Convert to Topic objects
            topics = self._convert_to_topics(curriculum_data, domain)
            
            # Limit to breadth_topics for initial learning
            initial_topics = topics[:breadth_topics]
            
            logger.info(f"Generated {len(initial_topics)} initial topics for {domain}")
            return initial_topics
            
        except DeepResearchError as e:
            logger.error(f"Failed to generate curriculum: {e}")
            # Fallback to basic curriculum
            return self._generate_fallback_curriculum(domain)
    
    async def expand_topic_depth(self, 
                               parent_topic: Topic,
                               domain: str,
                               max_subtopics: int = 5) -> List[Topic]:
        """
        Generate deeper subtopics for a mastered topic
        
        Args:
            parent_topic: The topic to expand
            domain: Domain context
            max_subtopics: Maximum number of subtopics to generate
        """
        
        logger.info(f"Expanding topic depth: {parent_topic.name}")
        
        expansion_query = f"""
        Create advanced subtopics for "{parent_topic.name}" in the domain of {domain}.
        
        Parent topic description: {parent_topic.description}
        
        Generate {max_subtopics} advanced subtopics that:
        1. Build upon the parent topic knowledge
        2. Explore specialized aspects in greater depth
        3. Include practical applications and implementations
        4. Cover emerging trends and advanced concepts
        5. Maintain clear learning progression
        
        For each subtopic, provide:
        - name: Specific, descriptive name
        - description: Detailed description of what will be learned
        - learning_objectives: 3-5 specific learning goals
        - prerequisites: Knowledge required before studying this topic
        - difficulty: advanced (since these are depth topics)
        - practical_applications: Real-world uses
        
        Format as JSON array.
        """
        
        try:
            response = await self.research_client.research_topic(
                query=expansion_query,
                domain=domain,
                depth="comprehensive"
            )
            
            # Parse subtopics from response
            subtopics_data = self._extract_subtopics_from_response(response.content)
            
            # Convert to Topic objects
            subtopics = []
            for i, subtopic_data in enumerate(subtopics_data[:max_subtopics]):
                topic = Topic(
                    id=f"{parent_topic.id}_depth_{i+1}",
                    name=subtopic_data.get("name", f"{parent_topic.name} - Advanced Topic {i+1}"),
                    description=subtopic_data.get("description", "Advanced topic details"),
                    depth=parent_topic.depth + 1,
                    parent_id=parent_topic.id,
                    status=TopicStatus.PENDING,
                    priority="medium"
                )
                subtopics.append(topic)
            
            logger.info(f"Generated {len(subtopics)} subtopics for {parent_topic.name}")
            return subtopics
            
        except Exception as e:
            logger.error(f"Failed to expand topic {parent_topic.name}: {e}")
            return []
    
    async def revise_curriculum_based_on_performance(self,
                                                   domain: str,
                                                   performance_data: Dict[str, Any],
                                                   current_topics: List[Topic]) -> List[Topic]:
        """
        Revise curriculum based on evaluation performance
        
        Args:
            domain: The learning domain
            performance_data: Results from evaluation
            current_topics: Currently active topics
        """
        
        logger.info("Revising curriculum based on performance data")
        
        # Analyze performance to determine revision strategy
        weak_areas = self._identify_weak_areas(performance_data)
        strong_areas = self._identify_strong_areas(performance_data)
        
        revision_query = f"""
        Revise the learning curriculum for {domain} based on performance analysis.
        
        Current topics and performance:
        {json.dumps([{"name": t.name, "status": t.status.value} for t in current_topics], indent=2)}
        
        Weak performance areas requiring remediation:
        {json.dumps(weak_areas, indent=2)}
        
        Strong performance areas ready for advancement:
        {json.dumps(strong_areas, indent=2)}
        
        Generate revised curriculum that:
        1. Provides remedial topics for weak areas
        2. Advances to deeper topics for strong areas
        3. Maintains learning progression and prerequisites
        4. Includes cross-topic connections
        5. Balances breadth and depth appropriately
        
        Return 5-10 new topics that address the performance gaps and build on strengths.
        Format as JSON array with topic objects.
        """
        
        try:
            response = await self.research_client.research_topic(
                query=revision_query,
                domain=domain,
                depth="comprehensive"
            )
            
            # Parse revised topics
            revised_data = self._extract_topics_from_response(response.content)
            revised_topics = self._convert_to_topics({"topics": revised_data}, domain)
            
            logger.info(f"Generated {len(revised_topics)} revised topics")
            return revised_topics
            
        except Exception as e:
            logger.error(f"Failed to revise curriculum: {e}")
            return []
    
    def _convert_to_topics(self, curriculum_data: Dict[str, Any], domain: str) -> List[Topic]:
        """Convert curriculum data to Topic objects"""
        topics = []
        
        topics_data = curriculum_data.get("topics", [])
        
        for topic_data in topics_data:
            topic = Topic(
                id=topic_data.get("id", str(uuid.uuid4())),
                name=topic_data.get("name", "Unnamed Topic"),
                description=topic_data.get("description", "Topic description"),
                depth=topic_data.get("depth", 1),
                parent_id=topic_data.get("parent_id"),
                status=TopicStatus.PENDING,
                priority=self._determine_priority(topic_data),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            topics.append(topic)
        
        return topics
    
    def _determine_priority(self, topic_data: Dict[str, Any]) -> str:
        """Determine topic priority based on topic data"""
        difficulty = topic_data.get("difficulty", "medium").lower()
        depth = topic_data.get("depth", 1)
        
        if depth == 1 or difficulty == "easy":
            return "high"  # Foundational topics
        elif difficulty == "hard" or depth > 3:
            return "low"   # Advanced topics
        else:
            return "medium"  # Intermediate topics
    
    def _extract_subtopics_from_response(self, content: str) -> List[Dict[str, Any]]:
        """Extract subtopic data from research response"""
        try:
            # Try to parse as JSON first
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "subtopics" in data:
                return data["subtopics"]
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract from text
        return self._parse_topics_from_text(content)
    
    def _extract_topics_from_response(self, content: str) -> List[Dict[str, Any]]:
        """Extract topic data from research response"""
        try:
            # Try to parse as JSON
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "topics" in data:
                return data["topics"]
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse from text
        return self._parse_topics_from_text(content)
    
    def _parse_topics_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Parse topics from text content as fallback"""
        # This is a simplified fallback parser
        # In practice, you'd implement more sophisticated text parsing
        
        topics = []
        lines = content.split('\n')
        
        current_topic = {}
        for line in lines:
            line = line.strip()
            if line.startswith('Name:') or line.startswith('Topic:'):
                if current_topic:
                    topics.append(current_topic)
                current_topic = {"name": line.split(':', 1)[1].strip()}
            elif line.startswith('Description:'):
                current_topic["description"] = line.split(':', 1)[1].strip()
            elif line.startswith('Difficulty:'):
                current_topic["difficulty"] = line.split(':', 1)[1].strip()
        
        if current_topic:
            topics.append(current_topic)
        
        return topics
    
    def _identify_weak_areas(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas with poor performance"""
        weak_areas = []
        
        # Extract topics with low scores
        topic_scores = performance_data.get("topic_scores", {})
        threshold = settings.learning.evaluation_threshold
        
        for topic_id, score in topic_scores.items():
            if score < threshold:
                weak_areas.append({
                    "topic_id": topic_id,
                    "score": score,
                    "needs": "remediation"
                })
        
        return weak_areas
    
    def _identify_strong_areas(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas with strong performance"""
        strong_areas = []
        
        topic_scores = performance_data.get("topic_scores", {})
        threshold = settings.learning.mastery_threshold
        
        for topic_id, score in topic_scores.items():
            if score >= threshold:
                strong_areas.append({
                    "topic_id": topic_id,
                    "score": score,
                    "ready_for": "advancement"
                })
        
        return strong_areas
    
    def _generate_fallback_curriculum(self, domain: str) -> List[Topic]:
        """Generate basic fallback curriculum if deep research fails"""
        logger.warning(f"Using fallback curriculum for domain: {domain}")
        
        fallback_topics = [
            f"Introduction to {domain}",
            f"Fundamental Concepts in {domain}",
            f"Basic Principles of {domain}",
            f"Common Applications in {domain}",
            f"Tools and Technologies for {domain}"
        ]
        
        topics = []
        for i, topic_name in enumerate(fallback_topics):
            topic = Topic(
                id=f"{domain.lower().replace(' ', '_')}_topic_{i+1}",
                name=topic_name,
                description=f"Learn about {topic_name.lower()}",
                depth=1,
                status=TopicStatus.PENDING,
                priority="medium" if i > 0 else "high"
            )
            topics.append(topic)
        
        return topics


# Factory function
def create_curriculum_generator(api_key: Optional[str] = None) -> CurriculumGenerator:
    """Create a configured curriculum generator"""
    return CurriculumGenerator(api_key=api_key) 