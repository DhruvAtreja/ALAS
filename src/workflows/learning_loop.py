"""
Main learning loop workflow using LangGraph
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from ..workflows.state_management import (
        LearningAgentState, 
        IterationStatus,
        TopicStatus,
        get_active_topics,
        should_continue_learning,
        Topic,
        PerformanceMetric
    )
    from ..config.settings import settings
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from workflows.state_management import (
        LearningAgentState, 
        IterationStatus,
        TopicStatus,
        get_active_topics,
        should_continue_learning,
        Topic,
        PerformanceMetric
    )
    from config.settings import settings
    from utils.logger import get_logger

logger = get_logger(__name__)


class SelfLearningWorkflow:
    """Main workflow orchestrator for the self-learning agent"""
    
    def __init__(self):
        self.builder = StateGraph(LearningAgentState)
        self.checkpointer = MemorySaver()  # Will be replaced with Redis in production
        self.setup_nodes()
        self.setup_edges()
        self.graph = None
        
    def setup_nodes(self):
        """Register all workflow nodes"""
        self.builder.add_node("initialization", self.initialization_node)
        self.builder.add_node("curriculum_generation", self.curriculum_generation_node)
        self.builder.add_node("knowledge_gathering", self.knowledge_gathering_node)
        self.builder.add_node("training_generation", self.training_generation_node)
        self.builder.add_node("fine_tuning", self.fine_tuning_node)
        self.builder.add_node("evaluation", self.evaluation_node)
        self.builder.add_node("analysis", self.analysis_node)
        
    def setup_edges(self):
        """Define workflow transitions"""
        # Set entry point
        self.builder.set_entry_point("initialization")
        
        # Linear flow for main loop
        self.builder.add_edge("initialization", "curriculum_generation")
        self.builder.add_edge("curriculum_generation", "knowledge_gathering")
        self.builder.add_edge("knowledge_gathering", "training_generation")
        self.builder.add_edge("training_generation", "fine_tuning")
        self.builder.add_edge("fine_tuning", "evaluation")
        self.builder.add_edge("evaluation", "analysis")
        
        # Conditional edge from analysis
        self.builder.add_conditional_edges(
            "analysis",
            self.should_continue,
            {
                "continue": "curriculum_generation",
                "complete": END
            }
        )
        
    def compile(self):
        """Compile the graph"""
        self.graph = self.builder.compile(checkpointer=self.checkpointer)
        return self.graph
        
    # Node implementations
    async def initialization_node(self, state: LearningAgentState) -> Dict[str, Any]:
        """Initialize the learning session"""
        logger.info(f"Initializing learning session for domain: {state['domain']}")
        
        # Generate session ID if not present
        session_id = state.get("session_id", str(uuid.uuid4()))
        
        # Initialize state fields
        updates = {
            "session_id": session_id,
            "iteration": 0,
            "iteration_status": IterationStatus.CURRICULUM_GENERATION,
            "curriculum": [],
            "topics_completed": [],
            "knowledge_base": {},
            "research_reports": [],
            "training_data": [],
            "dpo_pairs": [],
            "evaluation_results": [],
            "performance_history": [],
            "model_version": settings.openai.fine_tuning_model,
            "base_model": settings.openai.fine_tuning_model,
            "fine_tuned_models": [],
            "total_cost": 0.0,
            "iteration_costs": {},
            "errors": [],
            "retry_count": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "messages": [
                {
                    "role": "system",
                    "content": f"Initializing self-learning agent for domain: {state['domain']}"
                }
            ],
            "config": {
                "max_iterations": settings.learning.max_iterations,
                "evaluation_threshold": settings.learning.evaluation_threshold,
                "mastery_threshold": settings.learning.mastery_threshold,
                "topics_per_iteration": settings.learning.topics_per_iteration
            }
        }
        
        logger.info(f"Session initialized: {session_id}")
        return updates
        
    async def curriculum_generation_node(self, state: LearningAgentState) -> Dict[str, Any]:
        """Generate or revise curriculum based on performance"""
        logger.info(f"Generating curriculum for iteration {state['iteration']}")
        
        # Update status
        updates = {
            "iteration_status": IterationStatus.CURRICULUM_GENERATION,
            "updated_at": datetime.now()
        }
        
        # This is a placeholder - actual implementation will use curriculum generator
        if state["iteration"] == 0:
            # Initial curriculum generation
            message = f"Generating initial curriculum for domain: {state['domain']}"
            updates["messages"] = [
                {
                    "role": "assistant",
                    "content": message
                }
            ]
        else:
            # Revise curriculum based on evaluation results
            message = f"Revising curriculum based on evaluation results from iteration {state['iteration'] - 1}"
            updates["messages"] = [
                {
                    "role": "assistant", 
                    "content": message
                }
            ]
            
        logger.info("Curriculum generation complete")
        return updates
        
    async def knowledge_gathering_node(self, state: LearningAgentState) -> Dict[str, Any]:
        """Gather knowledge for active topics"""
        logger.info("Starting knowledge gathering phase")
        
        # Update status
        updates = {
            "iteration_status": IterationStatus.KNOWLEDGE_GATHERING,
            "updated_at": datetime.now()
        }
        
        # Get active topics for this iteration
        active_topics = get_active_topics(state, max_topics=settings.learning.topics_per_iteration)
        
        if active_topics:
            message = f"Gathering knowledge for {len(active_topics)} topics"
            updates["messages"] = [
                {
                    "role": "assistant",
                    "content": message
                }
            ]
            
            # Update topic statuses
            for topic in active_topics:
                topic.status = TopicStatus.IN_PROGRESS
        else:
            updates["messages"] = [
                {
                    "role": "assistant",
                    "content": "No active topics to research"
                }
            ]
            
        logger.info(f"Knowledge gathering initiated for {len(active_topics)} topics")
        return updates
        
    async def training_generation_node(self, state: LearningAgentState) -> Dict[str, Any]:
        """Generate training data from gathered knowledge"""
        logger.info("Generating training data")
        
        updates = {
            "iteration_status": IterationStatus.TRAINING_GENERATION,
            "updated_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Generating training examples for iteration {state['iteration']}"
                }
            ]
        }
        
        return updates
        
    async def fine_tuning_node(self, state: LearningAgentState) -> Dict[str, Any]:
        """Execute fine-tuning job"""
        logger.info("Starting fine-tuning process")
        
        updates = {
            "iteration_status": IterationStatus.FINE_TUNING,
            "updated_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": "Initiating fine-tuning job with OpenAI API"
                }
            ]
        }
        
        return updates
        
    async def evaluation_node(self, state: LearningAgentState) -> Dict[str, Any]:
        """Evaluate the fine-tuned model"""
        logger.info("Evaluating model performance")
        
        updates = {
            "iteration_status": IterationStatus.EVALUATION,
            "updated_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": "Running comprehensive evaluation on fine-tuned model"
                }
            ]
        }
        
        return updates
        
    async def analysis_node(self, state: LearningAgentState) -> Dict[str, Any]:
        """Analyze results and prepare for next iteration"""
        logger.info("Analyzing iteration results")
        
        # Increment iteration counter
        new_iteration = state["iteration"] + 1
        
        updates = {
            "iteration": new_iteration,
            "iteration_status": IterationStatus.COMPLETED,
            "updated_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Completed iteration {state['iteration']}. Preparing for iteration {new_iteration}"
                }
            ]
        }
        
        return updates
        
    def should_continue(self, state: LearningAgentState) -> str:
        """Decide whether to continue learning or complete"""
        if should_continue_learning(state):
            logger.info(f"Continuing to iteration {state['iteration']}")
            return "continue"
        else:
            logger.info("Learning objectives achieved or limits reached")
            return "complete"
            
    async def run(self, domain: str, config: Optional[Dict[str, Any]] = None) -> LearningAgentState:
        """Execute the learning workflow for a domain"""
        if not self.graph:
            self.compile()
            
        # Initialize state
        initial_state = {
            "domain": domain,
            "messages": [],
            "config": config or {}
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": domain}}
        
        async for event in self.graph.astream(initial_state, config):
            logger.debug(f"Event: {event}")
            
        # Get final state
        final_state = await self.graph.aget_state(config)
        return final_state.values


# Factory function
def create_learning_workflow() -> SelfLearningWorkflow:
    """Create and return a configured learning workflow"""
    workflow = SelfLearningWorkflow()
    workflow.compile()
    return workflow 