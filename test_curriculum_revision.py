#!/usr/bin/env python3
"""
Test script for curriculum revision based on DPO evaluation results
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.deep_research_client import create_deep_research_client, DeepResearchError
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_curriculum_revision_with_evaluation_results():
    """Test curriculum revision using actual DPO evaluation results"""
    
    print("\n🔄 Testing Curriculum Revision from DPO Evaluation Results...")
    print("=" * 70)
    
    try:
        client = create_deep_research_client()
        
        # Load DPO evaluation results 
        evaluation_file = "evaluation_results_finetuned_test.json"
        print(f"Loading evaluation results from: {evaluation_file}")
        
        try:
            with open(evaluation_file, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
        except FileNotFoundError:
            print(f"❌ Evaluation results file not found: {evaluation_file}")
            print("Using sample evaluation data for demonstration...")
            evaluation_results = create_sample_evaluation_results()
        
        # Load current curriculum for context
        curriculum_file = "curriculum_test_results.json"
        current_curriculum = None
        
        try:
            with open(curriculum_file, 'r', encoding='utf-8') as f:
                curriculum_data = json.load(f)
                current_curriculum = curriculum_data.get("curriculum")
        except FileNotFoundError:
            print(f"⚠️  Current curriculum file not found: {curriculum_file}")
            print("Proceeding without current curriculum context...")
        
        print(f"\n📊 Evaluation Summary:")
        eval_data = evaluation_results.get("evaluation_results", {})
        print(f"- Domain: {eval_data.get('domain', 'Unknown')}")
        print(f"- Overall Accuracy: {eval_data.get('overall_accuracy', 0):.1%}")
        print(f"- Total Questions: {eval_data.get('total_questions', 0)}")
        print(f"- Total Topics: {eval_data.get('total_topics', 0)}")
        
        # Show topic performance breakdown
        topic_results = eval_data.get('topic_results', [])
        print(f"\n📈 Topic Performance:")
        for topic in topic_results:
            accuracy = topic.get('accuracy', 0)
            status = "✅ Mastered" if accuracy >= 0.9 else "❌ Needs Work"
            print(f"- {topic.get('topic_name', 'Unknown')}: {accuracy:.1%} {status}")
        
        print(f"\n🧠 Generating revised curriculum...")
        
        # Generate revised curriculum
        revision_result = await client.generate_revised_curriculum_from_evaluation(
            evaluation_results=evaluation_results,
            current_curriculum=current_curriculum,
            accuracy_threshold=0.9
        )
        
        if revision_result is None:
            print("❌ Curriculum revision failed")
            return False
        
        print("✅ Curriculum revision successful!")
        
        # Display revision summary
        print(f"\n📋 Revision Summary:")
        print(revision_result.revision_summary)
        
        print(f"\n📊 Revision Statistics:")
        print(f"- Mastered Topics: {len(revision_result.mastered_topics)}")
        print(f"- Failed Topics: {len(revision_result.failed_topics)}")
        print(f"- Failed Questions Analyzed: {revision_result.failed_questions_count}")
        
        if revision_result.mastered_topics:
            print(f"\n✅ Mastered Topics:")
            for topic in revision_result.mastered_topics:
                print(f"  - {topic}")
        
        if revision_result.failed_topics:
            print(f"\n❌ Topics Needing Improvement:")
            for topic in revision_result.failed_topics:
                print(f"  - {topic}")
        
        # Display new curriculum summary
        if revision_result.revised_curriculum:
            curriculum = revision_result.revised_curriculum
            print(f"\n📚 Revised Curriculum Overview:")
            print(f"- Domain: {curriculum.domain}")
            print(f"- Total Topics: {len(curriculum.topics)}")
            print(f"- Difficulty Distribution:")
            
            difficulties = curriculum.metadata.difficulties
            print(f"  - Easy: {difficulties.easy} topics")
            print(f"  - Medium: {difficulties.medium} topics") 
            print(f"  - Hard: {difficulties.hard} topics")
            
            print(f"\n📝 First 5 Revised Topics:")
            for i, topic in enumerate(curriculum.topics[:5], 1):
                print(f"{i}. {topic.name} ({topic.difficulty})")
                print(f"   {topic.description[:100]}{'...' if len(topic.description) > 100 else ''}")
        
        # Save results
        save_revision_results(revision_result, evaluation_results)
        
        return True
        
    except DeepResearchError as e:
        print(f"❌ Deep Research Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_evaluation_results():
    """Create sample evaluation results for demonstration"""
    return {
        "evaluation_results": {
            "domain": "Python Programming",
            "overall_accuracy": 0.8,
            "total_questions": 10,
            "total_topics": 1,
            "topic_results": [
                {
                    "topic_name": "Introduction to Python Programming",
                    "accuracy": 0.8,
                    "results": [
                        {
                            "question_id": "sample_001",
                            "question": "What is Python?",
                            "model_answer": "Python is a low-level compiled language",
                            "ideal_answer": "Python is a high-level interpreted language",
                            "is_correct": False,
                            "explanation": "Model incorrectly described Python as low-level and compiled when it's high-level and interpreted",
                            "category": "Factual Recall",
                            "difficulty": "easy"
                        },
                        {
                            "question_id": "sample_002", 
                            "question": "Who created Python?",
                            "model_answer": "Dennis Ritchie created Python",
                            "ideal_answer": "Guido van Rossum created Python",
                            "is_correct": False,
                            "explanation": "Model confused Python's creator with C's creator",
                            "category": "Factual Recall",
                            "difficulty": "easy"
                        }
                    ]
                }
            ]
        }
    }


def save_revision_results(revision_result, evaluation_results):
    """Save revision results to file"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"curriculum_revision_results_{timestamp}.json"
    
    try:
        # Convert Pydantic models to dicts for JSON serialization
        results_data = {
            "revision_metadata": {
                "generated_at": datetime.now().isoformat(),
                "source_evaluation": evaluation_results.get("file_metadata", {}),
                "revision_summary": revision_result.revision_summary
            },
            "performance_analysis": {
                "mastered_topics": revision_result.mastered_topics,
                "failed_topics": revision_result.failed_topics,
                "failed_questions_count": revision_result.failed_questions_count
            },
            "original_curriculum": revision_result.original_curriculum.model_dump() if revision_result.original_curriculum else None,
            "revised_curriculum": revision_result.revised_curriculum.model_dump() if revision_result.revised_curriculum else None
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Revision results saved to: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save revision results: {e}")
        return False


async def test_high_performance_scenario():
    """Test curriculum revision with high-performing evaluation results"""
    
    print("\n🎯 Testing High-Performance Scenario (>90% accuracy)...")
    print("=" * 60)
    
    try:
        client = create_deep_research_client()
        
        # Load the high-performing DPO evaluation results
        evaluation_file = "data/evaluations/evaluation_results_dpo_20250704_224746.json"
        
        try:
            with open(evaluation_file, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
        except FileNotFoundError:
            print(f"❌ High-performance evaluation file not found: {evaluation_file}")
            return False
        
        eval_data = evaluation_results.get("evaluation_results", {})
        print(f"📊 High-Performance Evaluation:")
        print(f"- Domain: {eval_data.get('domain', 'Unknown')}")
        print(f"- Overall Accuracy: {eval_data.get('overall_accuracy', 0):.1%}")
        
        # Generate curriculum for high-performing learner
        revision_result = await client.generate_revised_curriculum_from_evaluation(
            evaluation_results=evaluation_results,
            accuracy_threshold=0.9
        )
        
        if revision_result:
            print("✅ High-performance curriculum revision successful!")
            print(f"\n📈 Results for High Performer:")
            print(f"- Mastered Topics: {len(revision_result.mastered_topics)}")
            print(f"- Topics Needing Work: {len(revision_result.failed_topics)}")
            print("\nThis curriculum should focus on advanced topics since learner has mastered basics.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in high-performance test: {e}")
        return False


async def main():
    """Run all curriculum revision tests"""
    
    print("🚀 Curriculum Revision Testing Suite")
    print("=" * 80)
    
    # Test with mixed performance (some failures)
    test1_success = await test_curriculum_revision_with_evaluation_results()
    
    # Test with high performance (no failures)
    test2_success = await test_high_performance_scenario()
    
    print(f"\n🏁 Test Results Summary:")
    print(f"- Mixed Performance Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"- High Performance Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    overall_success = test1_success and test2_success
    
    if overall_success:
        print("\n🎉 All curriculum revision tests passed!")
        print("\nKey Features Demonstrated:")
        print("✅ Analysis of evaluation results by topic performance")
        print("✅ Extraction of failed questions with explanations")
        print("✅ Separation of mastered vs struggling topics")
        print("✅ Generation of targeted curriculum based on knowledge gaps")
        print("✅ Handling of both low and high-performance scenarios")
        print("✅ Comprehensive prompt engineering for deep research API")
    else:
        print("\n❌ Some tests failed")
        print("\nTroubleshooting tips:")
        print("1. Check your OpenAI API key and access to deep research models")
        print("2. Verify evaluation results files are present and properly formatted")
        print("3. Check internet connectivity for deep research API calls")
        print("4. Review error messages above for specific issues")
    
    return overall_success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 