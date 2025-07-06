"""
Demo script for the Autonomous Learning Agent
Tests the complete self-learning loop with LangGraph orchestration
"""

import asyncio
import json
from pathlib import Path
from langsmith import traceable

from src.workflows.autonomous_learning_agent import create_autonomous_learning_agent


@traceable
async def main():
    """Demo the autonomous learning agent"""
    
    print("🚀 Autonomous Learning Agent Demo")
    print("=" * 60)
    
    try:
        # Configuration
        domain = "Self-Adapting Language Models, research paper"
        max_iterations = 3  # Reduced for demo
        session_id = "demo_session_001"
        
        print(f"📚 Domain: {domain}")
        print(f"🔄 Max iterations: {max_iterations}")
        print(f"📋 Session ID: {session_id}")
        print()
        
        # Create autonomous learning agent
        print("🛠️  Creating autonomous learning agent...")
        agent = create_autonomous_learning_agent(max_iterations=max_iterations)
        
        # Run the complete learning workflow
        print("🎯 Starting autonomous learning workflow...")
        print("This will take a while as it includes:")
        print("  1. Curriculum generation")
        print("  2. Training data creation")
        print("  3. Supervised fine-tuning")
        print("  4. Model evaluation")
        print("  5. DPO training on wrong answers")
        print("  6. DPO model evaluation")
        print("  7. Curriculum revision")
        print("  8. Repeat for multiple iterations")
        print()
        
        # Run the workflow
        final_state = await agent.run(domain=domain, session_id=session_id)
        
        # Display results
        print("\n🎉 Autonomous Learning Completed!")
        print("=" * 50)
        
        print(f"📋 Session ID: {final_state.get('session_id')}")
        print(f"🔄 Completed iterations: {len(final_state.get('iterations', []))}")
        print(f"💰 Total estimated cost: ${final_state.get('total_cost', 0):.2f}")
        print(f"🏆 Final model: {final_state.get('current_dpo_model_id', 'None')}")
        print(f"🎯 Status: {final_state.get('overall_status', 'Unknown')}")
        
        # Show iteration summary
        iterations = final_state.get('iterations', [])
        if iterations:
            print("\n📊 Iteration Summary:")
            print("-" * 40)
            
            for iteration_data in iterations:
                iteration = iteration_data.get('iteration', 'Unknown')
                mastered = len(iteration_data.get('mastered_topics', []))
                failed = len(iteration_data.get('failed_topics', []))
                
                print(f"  Iteration {iteration}:")
                print(f"    ✅ Mastered topics: {mastered}")
                print(f"    ❌ Failed topics: {failed}")
                print(f"    📂 Models: SFT({iteration_data.get('sft_model_id', 'None')[:20]}...) -> DPO({iteration_data.get('dpo_model_id', 'None')[:20]}...)")
                print()
        
        # Show any errors
        errors = final_state.get('errors', [])
        if errors:
            print("⚠️  Errors encountered:")
            for error in errors:
                print(f"  - {error}")
        
        # Show file locations
        print("\n📁 Generated Files:")
        print("-" * 20)
        
        if final_state.get('current_curriculum_file'):
            print(f"📚 Final curriculum: {final_state['current_curriculum_file']}")
        
        if final_state.get('current_revised_curriculum_file'):
            print(f"🔄 Revised curriculum: {final_state['current_revised_curriculum_file']}")
            
        if final_state.get('current_dpo_eval_file'):
            print(f"📊 Final evaluation: {final_state['current_dpo_eval_file']}")
        
        # Check for session summary
        summary_file = f"data/sessions/session_summary_{session_id}.json"
        if Path(summary_file).exists():
            print(f"📋 Session summary: {summary_file}")
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_requirements():
    """Show requirements for running the demo"""
    print("📋 Requirements:")
    print("- OpenAI API key configured")
    print("- Deep Research API access")
    print("- Sufficient OpenAI credits for fine-tuning")
    print("- Python 3.8+ with required packages")
    print()


def estimate_cost():
    """Estimate the cost of running the demo"""
    print("💰 Estimated Cost Breakdown (3 iterations):")
    print("- Curriculum generation: ~$0.10 per iteration")
    print("- Training data generation: ~$0.50 per iteration")
    print("- Supervised fine-tuning: ~$20-50 per iteration")
    print("- Model evaluation: ~$0.20 per iteration")
    print("- DPO fine-tuning: ~$15-30 per iteration")
    print("- DPO evaluation: ~$0.20 per iteration")
    print("- Curriculum revision: ~$0.10 per iteration")
    print()
    print("🔢 Total estimated: $110-240 for 3 iterations")
    print("💡 Tip: Start with 1 iteration for testing")
    print()


if __name__ == "__main__":
    print("🤖 Autonomous Learning Agent Demo")
    print("This demo will run a complete self-learning workflow")
    print()
    
    show_requirements()
    estimate_cost()
    
    # Ask for confirmation
    response = input("Continue with the demo? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting demo...")
        try:
            # Run the demo
            success = asyncio.run(main())
            
            if success:
                print("\n🎉 Demo completed! Check the generated files for results.")
            else:
                print("\n⚠️  Demo ended with issues. Check the logs above.")
                
        except Exception as e:
            print(f"\n❌ Failed to run demo: {e}")
            
    else:
        print("Demo cancelled. Run with 'python demo_autonomous_learning.py' when ready.") 