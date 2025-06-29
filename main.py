#!/usr/bin/env python3
"""
Main entry point for ALAS (Autonomous Learning Agent System)
"""

import asyncio
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workflows.learning_loop import create_learning_workflow
from utils.logger import get_logger, log_iteration_start, log_error
from config.settings import settings

logger = get_logger(__name__)


async def run_learning_session(domain: str, config: dict = None):
    """Run a complete learning session for a domain"""
    try:
        logger.info(f"Starting ALAS learning session for domain: {domain}")
        
        # Create workflow
        workflow = create_learning_workflow()
        
        # Run the workflow
        final_state = await workflow.run(domain, config)
        
        logger.info(f"Learning session completed for domain: {domain}")
        logger.info(f"Final iteration: {final_state.get('iteration', 0)}")
        logger.info(f"Topics covered: {len(final_state.get('topics_completed', []))}")
        
        return final_state
        
    except Exception as e:
        log_error(e, {"domain": domain, "operation": "learning_session"})
        raise


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ALAS - Autonomous Learning Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Machine Learning"
  python main.py "AI Agents" --max-iterations 5
  python main.py "LangChain" --topics-per-iteration 3
        """
    )
    
    parser.add_argument(
        "domain",
        help="The domain to learn about (e.g., 'Machine Learning', 'AI Agents')"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of learning iterations (default: from settings)"
    )
    
    parser.add_argument(
        "--topics-per-iteration",
        type=int,
        default=None,
        help="Number of topics to process per iteration (default: from settings)"
    )
    
    parser.add_argument(
        "--evaluation-threshold",
        type=float,
        default=None,
        help="Minimum accuracy threshold for topic mastery (default: from settings)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without making actual API calls"
    )
    
    args = parser.parse_args()
    
    # Build configuration from arguments
    config = {}
    if args.max_iterations:
        config["max_iterations"] = args.max_iterations
    if args.topics_per_iteration:
        config["topics_per_iteration"] = args.topics_per_iteration
    if args.evaluation_threshold:
        config["evaluation_threshold"] = args.evaluation_threshold
    if args.dry_run:
        config["dry_run"] = True
        logger.warning("Running in DRY-RUN mode - no actual API calls will be made")
    
    # Run the learning session
    try:
        asyncio.run(run_learning_session(args.domain, config))
    except KeyboardInterrupt:
        logger.info("Learning session interrupted by user")
    except Exception as e:
        logger.error(f"Failed to complete learning session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 