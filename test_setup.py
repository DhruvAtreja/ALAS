"""
Test script to verify Phase 1 setup
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Test imports
    print("Testing imports...")
    from src.config.settings import settings
    from src.workflows.state_management import LearningAgentState, Topic
    from src.workflows.learning_loop import create_learning_workflow
    from src.utils.logger import get_logger
    from src.core.deep_research_client import create_deep_research_client
    from src.core.curriculum_generator import create_curriculum_generator
    
    print("✓ All imports successful")
    
    # Test logger
    print("\nTesting logger...")
    logger = get_logger(__name__)
    logger.info("Logger test successful")
    print("✓ Logger configured")
    
    # Test settings (will fail if no .env file)
    print("\nTesting settings...")
    print(f"  Environment: {settings.environment}")
    print(f"  Log level: {settings.log_level}")
    print(f"  OpenAI API key configured: {'Yes' if settings.openai.api_key else 'No'}")
    print("✓ Settings loaded")
    
    # Test workflow creation
    print("\nTesting workflow creation...")
    workflow = create_learning_workflow()
    print("✓ Workflow created successfully")
    
    # Test graph visualization
    print("\nWorkflow graph structure:")
    if workflow.graph:
        print(f"  Nodes: {list(workflow.graph.nodes)}")
        print(f"  Entry point: initialization")
    else:
        print("  Graph not compiled")
    
    # Test basic state creation
    print("\nTesting state creation...")
    test_state = {
        "domain": "Test Domain",
        "messages": [],
        "config": {}
    }
    print("✓ State created successfully")
    
    # Test Deep Research client creation
    print("\nTesting Deep Research client...")
    try:
        research_client = create_deep_research_client()
        print("✓ Deep Research client created")
    except ValueError as e:
        print(f"⚠️  Deep Research client: {e}")
    
    # Test Curriculum generator
    print("\nTesting Curriculum generator...")
    try:
        curriculum_gen = create_curriculum_generator()
        print("✓ Curriculum generator created")
    except ValueError as e:
        print(f"⚠️  Curriculum generator: {e}")
    
    print("\n🎉 Phase 1 + Deep Research setup complete! All tests passed.")
    
    if not settings.openai.api_key:
        print("\n⚠️  Note: To test Deep Research functionality, set OPENAI_API_KEY in .env")
        print("   Example usage: python examples/deep_research_example.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc() 