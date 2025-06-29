"""
State management for the ALAS workflow using LangGraph
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.graph import add_messages


# Enums for status tracking
class TopicStatus(str, Enum):
    """Status of a learning topic"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    MASTERED = "mastered"


class IterationStatus(str, Enum):
    """Status of a learning iteration"""
    CURRICULUM_GENERATION = "curriculum_generation"
    KNOWLEDGE_GATHERING = "knowledge_gathering"
    TRAINING_GENERATION = "training_generation"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    COMPLETED = "completed"


# Data models
class Topic(BaseModel):
    """Represents a learning topic"""
    id: str
    name: str
    description: str
    depth: int = 1
    parent_id: Optional[str] = None
    status: TopicStatus = TopicStatus.PENDING
    performance_score: Optional[float] = None
    priority: str = "medium"
    subtopics: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TrainingExample(BaseModel):
    """Represents a training example for fine-tuning"""
    id: str
    topic_id: str
    messages: List[Dict[str, str]]  # OpenAI format
    category: str  # Factual, Conceptual, Application, etc.
    difficulty: str  # easy, medium, hard
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Represents an evaluation result"""
    id: str
    topic_id: str
    question: str
    model_answer: str
    correct_answer: Optional[str] = None
    is_correct: bool
    score: float
    category: str
    feedback: str
    timestamp: datetime = Field(default_factory=datetime.now)


class PerformanceMetric(BaseModel):
    """Tracks performance metrics over time"""
    iteration: int
    overall_accuracy: float
    topic_scores: Dict[str, float]
    category_scores: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str
    training_examples_used: int
    evaluation_questions_count: int


class KnowledgeBase(BaseModel):
    """Represents gathered knowledge for a topic"""
    topic_id: str
    content: str
    sources: List[Dict[str, str]]  # List of citations
    research_depth: str  # "shallow", "medium", "deep"
    timestamp: datetime = Field(default_factory=datetime.now)


class ResearchReport(BaseModel):
    """Deep Research API response structure"""
    topic_id: str
    topic_name: str
    content: str
    citations: List[Dict[str, Any]]
    research_steps: List[Dict[str, Any]]
    timestamp: str


# LangGraph State Definition
class LearningAgentState(TypedDict):
    """Main state for the self-learning agent workflow"""
    
    # Core identifiers
    domain: str
    session_id: str
    
    # Learning curriculum
    curriculum: List[Topic]
    current_topic: Optional[Topic]
    topics_completed: List[str]
    
    # Knowledge management
    knowledge_base: Dict[str, KnowledgeBase]  # topic_id -> knowledge
    research_reports: List[ResearchReport]
    
    # Training data
    training_data: List[TrainingExample]
    dpo_pairs: List[Dict[str, Any]]  # For preference optimization
    
    # Evaluation
    evaluation_results: List[EvalResult]
    current_eval_results: Optional[Dict[str, Any]]  # Latest evaluation
    
    # Model management
    model_version: str
    base_model: str
    fine_tuned_models: List[str]  # History of fine-tuned model IDs
    
    # Progress tracking
    iteration: int
    iteration_status: IterationStatus
    performance_history: List[PerformanceMetric]
    
    # Cost tracking
    total_cost: float
    iteration_costs: Dict[int, float]
    
    # Messages for LangGraph
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # Error handling
    errors: List[Dict[str, Any]]
    retry_count: int
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    config: Dict[str, Any]  # Runtime configuration


# Helper classes for state updates
class StateUpdate(BaseModel):
    """Represents a state update operation"""
    field: str
    operation: str  # "set", "append", "update"
    value: Any
    

class CheckpointMetadata(BaseModel):
    """Metadata for state checkpoints"""
    checkpoint_id: str
    iteration: int
    timestamp: datetime
    performance_score: float
    topics_covered: int
    status: str


# State validation functions
def validate_state(state: LearningAgentState) -> bool:
    """Validate state consistency"""
    # Check required fields
    if not state.get("domain") or not state.get("session_id"):
        return False
    
    # Check curriculum consistency
    if state.get("curriculum"):
        topic_ids = {topic.id for topic in state["curriculum"]}
        for topic in state["curriculum"]:
            if topic.parent_id and topic.parent_id not in topic_ids:
                return False
    
    # Check iteration bounds
    if state.get("iteration", 0) < 0:
        return False
    
    return True


def get_active_topics(state: LearningAgentState, max_topics: int = 5) -> List[Topic]:
    """Get active topics for current iteration"""
    pending_topics = [
        topic for topic in state.get("curriculum", [])
        if topic.status in [TopicStatus.PENDING, TopicStatus.IN_PROGRESS]
    ]
    
    # Sort by priority and depth
    priority_order = {"high": 0, "medium": 1, "low": 2}
    pending_topics.sort(
        key=lambda t: (priority_order.get(t.priority, 1), t.depth)
    )
    
    return pending_topics[:max_topics]


def calculate_domain_progress(state: LearningAgentState) -> float:
    """Calculate overall progress percentage"""
    if not state.get("curriculum"):
        return 0.0
    
    total_topics = len(state["curriculum"])
    completed_topics = len([
        t for t in state["curriculum"] 
        if t.status in [TopicStatus.COMPLETED, TopicStatus.MASTERED]
    ])
    
    return (completed_topics / total_topics) * 100 if total_topics > 0 else 0.0


def should_continue_learning(state: LearningAgentState) -> bool:
    """Determine if learning should continue"""
    # Check iteration limit
    max_iterations = state.get("config", {}).get("max_iterations", 20)
    if state.get("iteration", 0) >= max_iterations:
        return False
    
    # Check if all topics are mastered
    if state.get("curriculum"):
        pending_topics = [
            t for t in state["curriculum"]
            if t.status not in [TopicStatus.COMPLETED, TopicStatus.MASTERED]
        ]
        if not pending_topics:
            return False
    
    # Check performance plateau
    if len(state.get("performance_history", [])) >= 3:
        recent_scores = [p.overall_accuracy for p in state["performance_history"][-3:]]
        if all(s >= 0.95 for s in recent_scores):
            return False
    
    return True 