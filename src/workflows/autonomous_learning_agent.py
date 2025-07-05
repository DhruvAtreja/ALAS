"""
Autonomous Learning Agent using LangGraph
Orchestrates the complete self-learning loop: curriculum → training → evaluation → revision
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, TypedDict, Literal
from datetime import datetime
from pathlib import Path
import uuid

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

try:
    from ..core.deep_research_client import create_deep_research_client, Curriculum
    from ..core.training_data_generator import create_training_data_generator, CurriculumTrainingData
    from ..core.fine_tuner import create_fine_tuner, FineTuningHyperparameters
    from ..core.evaluator import create_model_evaluator
    from ..core.curriculum_revision import revise_curriculum_from_dpo_results
    from ..core.dpo_improvement import DPOImprovementEngine
    from ..config.settings import settings
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.deep_research_client import create_deep_research_client, Curriculum
    from core.training_data_generator import create_training_data_generator, CurriculumTrainingData
    from core.fine_tuner import create_fine_tuner, FineTuningHyperparameters
    from core.evaluator import create_model_evaluator
    from core.curriculum_revision import revise_curriculum_from_dpo_results
    from core.dpo_improvement import DPOImprovementEngine
    from config.settings import settings
    from utils.logger import get_logger

logger = get_logger(__name__)


class LearningIteration(BaseModel):
    """Represents a single learning iteration"""
    iteration: int = Field(..., description="Iteration number (1-based)")
    curriculum_file: Optional[str] = Field(default=None, description="Curriculum file for this iteration")
    training_data_file: Optional[str] = Field(default=None, description="Training data file")
    sft_model_id: Optional[str] = Field(default=None, description="SFT fine-tuned model ID")
    sft_eval_file: Optional[str] = Field(default=None, description="SFT evaluation results file")
    dpo_data_file: Optional[str] = Field(default=None, description="DPO training data file")
    dpo_model_id: Optional[str] = Field(default=None, description="DPO fine-tuned model ID")
    dpo_eval_file: Optional[str] = Field(default=None, description="DPO evaluation results file")
    revised_curriculum_file: Optional[str] = Field(default=None, description="Revised curriculum file")
    started_at: Optional[str] = Field(default=None, description="Iteration start time")
    completed_at: Optional[str] = Field(default=None, description="Iteration completion time")
    status: str = Field(default="pending", description="Iteration status")
    errors: List[str] = Field(default_factory=list, description="Any errors during iteration")


class AutonomousLearningStateRequired(TypedDict):
    """Required state fields"""
    domain: str
    session_id: str
    max_iterations: int
    current_iteration: int
    base_model: str
    current_step: str
    overall_status: str
    total_cost: float
    errors: List[str]
    iterations: List[Dict[str, Any]]
    started_at: str
    last_updated: str

class AutonomousLearningStateOptional(TypedDict, total=False):
    """Optional state fields"""
    current_curriculum_file: Optional[str]
    current_training_data_file: Optional[str] 
    current_sft_model_id: Optional[str]
    current_sft_eval_file: Optional[str]
    current_dpo_data_file: Optional[str]
    current_dpo_model_id: Optional[str]
    current_dpo_eval_file: Optional[str]
    current_revised_curriculum_file: Optional[str]
    completed_at: Optional[str]

class AutonomousLearningState(AutonomousLearningStateRequired, AutonomousLearningStateOptional):
    """Complete state for the autonomous learning agent"""
    pass


class AutonomousLearningAgent:
    """Main autonomous learning agent using LangGraph"""
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.builder = StateGraph(AutonomousLearningState)
        self.checkpointer = MemorySaver()
        self.setup_nodes()
        self.setup_edges()
        self.graph = None
        
        # Initialize components
        self.deep_research_client = create_deep_research_client()
        self.training_data_generator = create_training_data_generator()
        self.fine_tuner = create_fine_tuner()
        self.evaluator = create_model_evaluator()
        self.dpo_engine = DPOImprovementEngine()
        
    def setup_nodes(self):
        """Setup all workflow nodes"""
        self.builder.add_node("initialize", self.initialize_node)
        self.builder.add_node("generate_curriculum", self.generate_curriculum_node)
        self.builder.add_node("generate_training_data", self.generate_training_data_node)
        self.builder.add_node("sft_training", self.sft_training_node)
        self.builder.add_node("sft_evaluation", self.sft_evaluation_node)
        self.builder.add_node("dpo_training", self.dpo_training_node)
        self.builder.add_node("dpo_evaluation", self.dpo_evaluation_node)
        self.builder.add_node("revise_curriculum", self.revise_curriculum_node)
        self.builder.add_node("finalize", self.finalize_node)
        
    def setup_edges(self):
        """Setup workflow transitions"""
        # Entry point
        self.builder.set_entry_point("initialize")
        
        # Main flow
        self.builder.add_edge("initialize", "generate_curriculum")
        self.builder.add_edge("generate_curriculum", "generate_training_data")
        self.builder.add_edge("generate_training_data", "sft_training")
        self.builder.add_edge("sft_training", "sft_evaluation")
        self.builder.add_edge("sft_evaluation", "dpo_training")
        self.builder.add_edge("dpo_training", "dpo_evaluation")
        self.builder.add_edge("dpo_evaluation", "revise_curriculum")
        
        # Conditional edge for iteration control
        self.builder.add_conditional_edges(
            "revise_curriculum",
            self.should_continue_learning,
            {
                "continue": "generate_training_data",  # Go to next iteration
                "complete": "finalize"
            }
        )
        
        self.builder.add_edge("finalize", END)
    
    def compile(self):
        """Compile the workflow graph"""
        self.graph = self.builder.compile(checkpointer=self.checkpointer)
        return self.graph
    
    # Node implementations
    
    async def initialize_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Initialize the learning session"""
        domain = "Self-Adapting Language Models, research paper"
        logger.info(f"🚀 Initializing autonomous learning for domain: {domain}")
        
        current_time = datetime.now().isoformat()
        session_id = state.get('session_id', str(uuid.uuid4()))
        
        return {
            "domain": domain,
            "session_id": session_id,
            "current_iteration": 1,
            "max_iterations": self.max_iterations,
            "iterations": [],
            "base_model": settings.openai.fine_tuning_model,
            "current_step": "initialization",
            "overall_status": "running",
            "total_cost": 0.0,
            "errors": [],
            "started_at": current_time,
            "last_updated": current_time,
            "completed_at": None
        }
    
    async def generate_curriculum_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Generate or revise curriculum"""
        iteration = state["current_iteration"]
        logger.info(f"📚 Generating curriculum for iteration {iteration}")
        
        try:
            current_time = datetime.now().isoformat()
            
            if iteration == 1:
                # Initial curriculum generation
                curriculum = await self.deep_research_client.generate_curriculum(
                    domain=state["domain"],
                    current_topics=None,
                    learning_goals=[f"Master {state['domain']} through iterative learning"]
                )
            else:
                # Use revised curriculum from previous iteration
                curriculum_file = state.get("current_revised_curriculum_file")
                if curriculum_file and Path(curriculum_file).exists():
                    with open(curriculum_file, 'r') as f:
                        curriculum_data = json.load(f)
                        if "revised_curriculum" in curriculum_data:
                            curriculum = Curriculum(**curriculum_data["revised_curriculum"])
                        else:
                            curriculum = Curriculum(**curriculum_data)
                else:
                    raise ValueError("No revised curriculum found for subsequent iteration")
            
            if curriculum is None:
                raise ValueError("Failed to generate curriculum")
            
            # Save curriculum
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            curriculum_file = f"data/curricula/curriculum_iter{iteration}_{timestamp}.json"
            
            # Ensure directory exists
            Path(curriculum_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(curriculum_file, 'w') as f:
                json.dump(curriculum.model_dump(), f, indent=2)
            
            logger.info(f"✅ Curriculum generated: {curriculum.metadata.total_topics} topics")
            
            return {
                "current_curriculum_file": curriculum_file,
                "current_step": "curriculum_generation",
                "last_updated": current_time
            }
            
        except Exception as e:
            error_msg = f"Failed to generate curriculum: {e}"
            logger.error(error_msg)
            return {
                "current_step": "curriculum_generation_failed",
                "errors": state.get("errors", []) + [error_msg],
                "last_updated": datetime.now().isoformat()
            }
    
    async def generate_training_data_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Generate training data from curriculum"""
        iteration = state["current_iteration"]
        logger.info(f"📝 Generating training data for iteration {iteration}")
        
        try:
            current_time = datetime.now().isoformat()
            curriculum_file = state.get("current_curriculum_file")
            
            if not curriculum_file:
                raise ValueError("No curriculum file available")
            
            # Load curriculum from file
            with open(curriculum_file, 'r') as f:
                curriculum_data = json.load(f)
                curriculum = Curriculum(**curriculum_data)
            
            # Generate training data using the existing method
            training_data = await self.training_data_generator.generate_curriculum_training_data(curriculum)
            
            if not training_data:
                raise ValueError("Training data generation failed")
            
            # Save training data
            output_file = self.training_data_generator.save_training_data(training_data)
            
            logger.info(f"✅ Training data generated: {training_data.total_questions} questions")
            
            return {
                "current_training_data_file": output_file,
                "current_step": "training_data_generation",
                "last_updated": current_time
            }
            
        except Exception as e:
            error_msg = f"Failed to generate training data: {e}"
            logger.error(error_msg)
            return {
                "current_step": "training_data_generation_failed",
                "errors": state.get("errors", []) + [error_msg],
                "last_updated": datetime.now().isoformat()
            }
    
    async def sft_training_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Supervised fine-tuning"""
        iteration = state["current_iteration"]
        logger.info(f"🔧 Starting SFT training for iteration {iteration}")
        
        try:
            current_time = datetime.now().isoformat()
            training_file = state.get("current_training_data_file")
            
            if not training_file:
                raise ValueError("No training data file available")
            
            # Convert to OpenAI format
            openai_file = training_file.replace('.json', '_openai.jsonl')
            if not Path(openai_file).exists():
                # Load training data and export for OpenAI
                with open(training_file, 'r') as f:
                    data = json.load(f)
                    training_data = CurriculumTrainingData(**data["training_data"])
                
                openai_file = self.training_data_generator.export_for_openai_finetuning(training_data)
            
            # Get base model (first iteration uses base model, subsequent use previous SFT model)
            if iteration == 1:
                base_model = state["base_model"]
            else:
                previous_model = state.get("current_sft_model_id")
                base_model = previous_model if previous_model else state["base_model"]
            
            # Start fine-tuning
            result = self.fine_tuner.fine_tune_from_file(
                training_file_path=openai_file,
                model=base_model,
                hyperparameters=FineTuningHyperparameters(n_epochs=3),
                wait_for_completion=True
            )
            
            if result.status.value != "succeeded":
                raise ValueError(f"Fine-tuning failed with status: {result.status}")
            
            logger.info(f"✅ SFT completed: {result.fine_tuned_model}")
            
            return {
                "current_sft_model_id": result.fine_tuned_model,
                "current_step": "sft_training",
                "last_updated": current_time,
                "total_cost": state.get("total_cost", 0.0) + 50.0  # Estimate
            }
            
        except Exception as e:
            error_msg = f"SFT training failed: {e}"
            logger.error(error_msg)
            return {
                "current_step": "sft_training_failed",
                "errors": state.get("errors", []) + [error_msg],
                "last_updated": datetime.now().isoformat()
            }
    
    async def sft_evaluation_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Evaluate SFT model"""
        iteration = state["current_iteration"]
        logger.info(f"📊 Evaluating SFT model for iteration {iteration}")
        
        try:
            current_time = datetime.now().isoformat()
            model_id = state.get("current_sft_model_id")
            training_file = state.get("current_training_data_file")
            
            if not model_id or not training_file:
                raise ValueError("Missing model ID or training file for evaluation")
            
            # Load training data for evaluation
            with open(training_file, 'r') as f:
                data = json.load(f)
                training_data = CurriculumTrainingData(**data["training_data"])
            
            # Create evaluator for this specific model
            evaluator = create_model_evaluator(model_to_test=model_id)
            
            # Evaluate the model
            eval_result = await evaluator.evaluate_training_data(training_data)
            
            # Save evaluation results
            eval_file = evaluator.save_evaluation_results(eval_result)
            
            logger.info(f"✅ SFT evaluation completed: {eval_result.overall_accuracy:.1%} accuracy")
            
            return {
                "current_sft_eval_file": eval_file,
                "current_step": "sft_evaluation",
                "last_updated": current_time
            }
            
        except Exception as e:
            error_msg = f"SFT evaluation failed: {e}"
            logger.error(error_msg)
            return {
                "current_step": "sft_evaluation_failed",
                "errors": state.get("errors", []) + [error_msg],
                "last_updated": datetime.now().isoformat()
            }
    
    async def dpo_training_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """DPO training on incorrect answers"""
        iteration = state["current_iteration"]
        logger.info(f"🎯 Starting DPO training for iteration {iteration}")
        
        try:
            current_time = datetime.now().isoformat()
            sft_eval_file = state.get("current_sft_eval_file")
            sft_model_id = state.get("current_sft_model_id")
            
            if not sft_eval_file or not sft_model_id:
                raise ValueError("Missing evaluation file or SFT model for DPO training")
            
            # Load evaluation results
            with open(sft_eval_file, 'r') as f:
                eval_data = json.load(f)
            
            # Generate DPO data from evaluation results using correct method
            dpo_examples = self.dpo_engine.extract_wrong_answers(eval_data)
            
            if not dpo_examples:
                logger.warning("No DPO examples generated - all answers were correct")
                # If no wrong answers, just use the SFT model as DPO model
                return {
                    "current_dpo_model_id": sft_model_id,
                    "current_dpo_data_file": None,
                    "current_step": "dpo_training_skipped",
                    "last_updated": current_time
                }
            
            # Create DPO training file
            dpo_file = self.dpo_engine.create_dpo_training_file(dpo_examples)
            
            # Start DPO fine-tuning using the correct method
            dpo_model = self.dpo_engine.perform_dpo_fine_tuning(
                training_file_path=dpo_file,
                base_model=sft_model_id
            )
            
            logger.info(f"✅ DPO training completed: {dpo_model}")
            
            return {
                "current_dpo_model_id": dpo_model,
                "current_dpo_data_file": dpo_file,
                "current_step": "dpo_training",
                "last_updated": current_time,
                "total_cost": state.get("total_cost", 0.0) + 30.0  # Estimate
            }
            
        except Exception as e:
            error_msg = f"DPO training failed: {e}"
            logger.error(error_msg)
            return {
                "current_step": "dpo_training_failed",
                "errors": state.get("errors", []) + [error_msg],
                "last_updated": datetime.now().isoformat()
            }
    
    async def dpo_evaluation_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Evaluate DPO model"""
        iteration = state["current_iteration"]
        logger.info(f"📈 Evaluating DPO model for iteration {iteration}")
        
        try:
            current_time = datetime.now().isoformat()
            model_id = state.get("current_dpo_model_id")
            training_file = state.get("current_training_data_file")
            
            if not model_id or not training_file:
                raise ValueError("Missing model ID or training file for DPO evaluation")
            
            # Load training data for evaluation
            with open(training_file, 'r') as f:
                data = json.load(f)
                training_data = CurriculumTrainingData(**data["training_data"])
            
            # Create evaluator for this specific model
            evaluator = create_model_evaluator(model_to_test=model_id)
            
            # Evaluate the model
            eval_result = await evaluator.evaluate_training_data(training_data)
            
            # Save evaluation results with DPO suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dpo_eval_file = f"data/evaluations/evaluation_results_dpo_{timestamp}.json"
            
            # Ensure directory exists
            Path(dpo_eval_file).parent.mkdir(parents=True, exist_ok=True)
            
            eval_file = evaluator.save_evaluation_results(eval_result, dpo_eval_file)
            
            logger.info(f"✅ DPO evaluation completed: {eval_result.overall_accuracy:.1%} accuracy")
            
            return {
                "current_dpo_eval_file": eval_file,
                "current_step": "dpo_evaluation",
                "last_updated": current_time
            }
            
        except Exception as e:
            error_msg = f"DPO evaluation failed: {e}"
            logger.error(error_msg)
            return {
                "current_step": "dpo_evaluation_failed",
                "errors": state.get("errors", []) + [error_msg],
                "last_updated": datetime.now().isoformat()
            }
    
    async def revise_curriculum_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Revise curriculum based on DPO evaluation results"""
        iteration = state["current_iteration"]
        logger.info(f"🔄 Revising curriculum for iteration {iteration}")
        
        try:
            current_time = datetime.now().isoformat()
            dpo_eval_file = state["current_dpo_eval_file"]  # type: ignore
            curriculum_file = state["current_curriculum_file"]  # type: ignore
            
            if not dpo_eval_file:
                raise ValueError("No DPO evaluation file available for curriculum revision")
            
            # Generate revised curriculum
            revision_result = await revise_curriculum_from_dpo_results(
                evaluation_results_path=dpo_eval_file,
                current_curriculum_path=curriculum_file,
                accuracy_threshold=0.9,
                iteration=iteration
            )
            
            if not revision_result or not revision_result.revised_curriculum:
                raise ValueError("Curriculum revision failed")
            
            # Save revised curriculum
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            revised_file = f"data/curricula/revised_curriculum_iter{iteration}_{timestamp}.json"
            
            Path(revised_file).parent.mkdir(parents=True, exist_ok=True)
            
            revision_data = {
                "iteration": iteration,
                "revision_metadata": {
                    "generated_at": current_time,
                    "source_evaluation": dpo_eval_file,
                    "mastered_topics": revision_result.mastered_topics,
                    "failed_topics": revision_result.failed_topics
                },
                "revised_curriculum": revision_result.revised_curriculum.model_dump()
            }
            
            with open(revised_file, 'w') as f:
                json.dump(revision_data, f, indent=2)
            
            # Save iteration summary
            current_iteration_data = {
                "iteration": iteration,
                "curriculum_file": curriculum_file,
                "training_data_file": state.get("current_training_data_file"),
                "sft_model_id": state.get("current_sft_model_id"),
                "sft_eval_file": state.get("current_sft_eval_file"),
                "dpo_data_file": state.get("current_dpo_data_file"),
                "dpo_model_id": state.get("current_dpo_model_id"),
                "dpo_eval_file": dpo_eval_file,
                "revised_curriculum_file": revised_file,
                "started_at": state.get("started_at"),
                "completed_at": current_time,
                "status": "completed",
                "mastered_topics": revision_result.mastered_topics,
                "failed_topics": revision_result.failed_topics
            }
            
            iterations = state.get("iterations", [])
            iterations.append(current_iteration_data)
            
            logger.info(f"✅ Curriculum revision completed for iteration {iteration}")
            logger.info(f"📊 {len(revision_result.mastered_topics)} topics mastered, {len(revision_result.failed_topics)} need work")
            
            return {
                "current_revised_curriculum_file": revised_file,
                "current_step": "curriculum_revision",
                "last_updated": current_time,
                "iterations": iterations,
                "current_iteration": iteration + 1  # Prepare for next iteration
            }
            
        except Exception as e:
            error_msg = f"Curriculum revision failed: {e}"
            logger.error(error_msg)
            return {
                "current_step": "curriculum_revision_failed",
                "errors": state.get("errors", []) + [error_msg],
                "last_updated": datetime.now().isoformat()
            }
    
    def should_continue_learning(self, state: AutonomousLearningState) -> Literal["continue", "complete"]:
        """Decide whether to continue learning or complete"""
        current_iteration = state["current_iteration"]
        max_iterations = state["max_iterations"]
        
        if current_iteration > max_iterations:
            logger.info(f"🏁 Reached maximum iterations ({max_iterations})")
            return "complete"
        
        # Could add additional stopping criteria here
        # e.g., if accuracy is consistently high, if no failed topics, etc.
        
        logger.info(f"🔄 Continuing to iteration {current_iteration}")
        return "continue"
    
    async def finalize_node(self, state: AutonomousLearningState) -> Dict[str, Any]:
        """Finalize the learning session"""
        logger.info("🏆 Finalizing autonomous learning session")
        
        current_time = datetime.now().isoformat()
        
        # Generate session summary
        summary = {
            "session_id": state["session_id"],
            "domain": state["domain"],
            "completed_iterations": len(state.get("iterations", [])),
            "total_cost": state.get("total_cost", 0.0),
            "started_at": state["started_at"],
            "completed_at": current_time,
            "final_model": state.get("current_dpo_model_id"),
            "iterations_summary": state.get("iterations", [])
        }
        
        # Save session summary
        summary_file = f"data/sessions/session_summary_{state['session_id']}.json"
        Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Session completed! Summary saved to: {summary_file}")
        logger.info(f"📊 Total iterations: {summary['completed_iterations']}")
        logger.info(f"💰 Total cost: ${summary['total_cost']:.2f}")
        
        return {
            "overall_status": "completed",
            "completed_at": current_time,
            "current_step": "finalized",
            "last_updated": current_time
        }
    
    async def run(self, domain: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the autonomous learning workflow"""
        if not self.graph:
            self.compile()
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        initial_state = {
            "domain": domain,
            "session_id": session_id,
            "max_iterations": self.max_iterations
        }
        
        config = {"configurable": {"thread_id": session_id}}
        
        logger.info(f"🚀 Starting autonomous learning for: {domain}")
        logger.info(f"📋 Session ID: {session_id}")
        logger.info(f"🔄 Max iterations: {self.max_iterations}")
        
        try:
            # Ensure graph is compiled
            if self.graph is None:
                raise ValueError("Graph not compiled - call compile() first")
            
            # Stream the workflow execution
            async for event in self.graph.astream(initial_state, config):  # type: ignore
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                
                current_step = node_output.get("current_step", node_name)
                logger.info(f"🔄 Completed step: {current_step}")
                
                # Log any errors
                if "errors" in node_output and node_output["errors"]:
                    for error in node_output["errors"]:
                        logger.error(f"❌ Error: {error}")
            
            # Get final state
            final_state = await self.graph.aget_state(config)  # type: ignore
            return final_state.values
            
        except Exception as e:
            logger.error(f"❌ Autonomous learning failed: {e}")
            raise


# Factory function
def create_autonomous_learning_agent(max_iterations: int = 5) -> AutonomousLearningAgent:
    """Create and compile an autonomous learning agent"""
    try:
        agent = AutonomousLearningAgent(max_iterations=max_iterations)
        agent.compile()
        return agent
    except Exception as e:
        logger.error(f"Failed to create autonomous learning agent: {e}")
        import traceback
        traceback.print_exc()
        raise

# Factory function for LangGraph Studio
def create_autonomous_learning_graph(max_iterations: int = 5):
    """Create and compile an autonomous learning agent graph for LangGraph Studio"""
    try:
        agent = AutonomousLearningAgent(max_iterations=max_iterations)
        graph = agent.compile()
        if graph is None:
            raise ValueError("Failed to compile graph - returned None")
        return graph
    except Exception as e:
        logger.error(f"Failed to create autonomous learning agent: {e}")
        import traceback
        traceback.print_exc()
        raise 