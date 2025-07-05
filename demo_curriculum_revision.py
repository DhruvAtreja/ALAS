#!/usr/bin/env python3
"""
Simple demonstration of curriculum revision functionality
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.curriculum_revision import (
    revise_curriculum_from_dpo_results,
    analyze_dpo_evaluation_performance,
    CurriculumRevisionEngine
)


async def demo_curriculum_revision():
    """Demonstrate curriculum revision with available evaluation results"""
    
    print("🔄 Curriculum Revision Demo")
    print("=" * 50)
    
    # Available evaluation files (prioritizing DPO results)
    evaluation_files = [
        "data/evaluations/evaluation_results_dpo_20250704_224746.json",  # DPO results (100% accuracy) - PRIMARY
        "evaluation_results_finetuned_test.json",  # Pre-DPO results (80% accuracy) - for comparison
    ]
    
    current_curriculum_file = "curriculum_test_results.json"
    
    for eval_file in evaluation_files:
        try:
            if not Path(eval_file).exists():
                print(f"⚠️  Skipping {eval_file} - file not found")
                continue
                
            print(f"\n📊 Analyzing: {eval_file}")
            print("-" * 40)
            
            # First, analyze the performance
            analysis = analyze_dpo_evaluation_performance(eval_file)
            
            print(f"Domain: {analysis['domain']}")
            print(f"Overall Accuracy: {analysis['overall_accuracy']:.1%}")
            print(f"Topics: {analysis['total_topics']} total, {len(analysis['mastered_topics'])} mastered, {len(analysis['failed_topics'])} failed")
            print(f"Questions: {analysis['total_questions']} total, {analysis['failed_questions_count']} failed")
            
            if analysis['mastered_topics']:
                print(f"✅ Mastered: {', '.join(analysis['mastered_topics'])}")
            
            if analysis['failed_topics']:
                print(f"❌ Failed: {', '.join(analysis['failed_topics'])}")
            
            # Generate revised curriculum
            print(f"\n🧠 Generating revised curriculum...")
            
            revision_result = await revise_curriculum_from_dpo_results(
                evaluation_results_path=eval_file,
                current_curriculum_path=current_curriculum_file if Path(current_curriculum_file).exists() else None,
                accuracy_threshold=0.9,
                iteration=1
            )
            
            if revision_result:
                print("✅ Curriculum revision successful!")
                print(f"\nRevision Summary:")
                print(revision_result.revision_summary)
                
                if revision_result.revised_curriculum:
                    curriculum = revision_result.revised_curriculum
                    print(f"\n📚 Revised Curriculum:")
                    print(f"- {len(curriculum.topics)} topics total")
                    print(f"- Difficulty: {curriculum.metadata.difficulties.easy} easy, "
                          f"{curriculum.metadata.difficulties.medium} medium, "
                          f"{curriculum.metadata.difficulties.hard} hard")
                    
                    print(f"\nSample Topics:")
                    for i, topic in enumerate(curriculum.topics[:3], 1):
                        print(f"{i}. {topic.name} ({topic.difficulty})")
            else:
                print("❌ Curriculum revision failed")
                
        except Exception as e:
            print(f"❌ Error processing {eval_file}: {e}")
    
    print(f"\n🎉 Demo complete! Check the generated curriculum revision files.")


async def demo_different_learner_types():
    """Demonstrate curriculum revision for different learner performance levels"""
    
    print("\n👥 Different Learner Types Demo")
    print("=" * 50)
    
    # Test files and their expected performance levels
    test_scenarios = [
        {
            "file": "evaluation_results_finetuned_test.json",
            "type": "Mixed Performance Learner",
            "expected_accuracy": "~80%",
            "description": "Some topics mastered, some need work"
        },
        {
            "file": "data/evaluations/evaluation_results_dpo_20250704_224746.json", 
            "type": "High Performance Learner",
            "expected_accuracy": "100%",
            "description": "All topics mastered, needs advanced content"
        }
    ]
    
    for scenario in test_scenarios:
        eval_file = scenario["file"]
        
        if not Path(eval_file).exists():
            print(f"⚠️  Skipping {scenario['type']} - file not found: {eval_file}")
            continue
            
        print(f"\n🎯 {scenario['type']} ({scenario['expected_accuracy']})")
        print(f"📝 {scenario['description']}")
        print("-" * 40)
        
        try:
            # Use different thresholds for different learner types
            if "High Performance" in scenario["type"]:
                threshold = 0.9  # Strict threshold for high performers
            elif "Mixed Performance" in scenario["type"]:
                threshold = 0.8  # Moderate threshold for mixed learners
            else:
                threshold = 0.7  # Lenient threshold for struggling learners
            
            engine = CurriculumRevisionEngine(accuracy_threshold=threshold)
            
            result = await engine.revise_curriculum_from_evaluation_file(
                evaluation_file_path=eval_file,
                save_results=True
            )
            
            if result:
                print(f"✅ Generated curriculum for {scenario['type']}")
                print(f"📊 {len(result.mastered_topics)} mastered, {len(result.failed_topics)} need work")
                
                if result.revised_curriculum:
                    curriculum = result.revised_curriculum
                    difficulties = curriculum.metadata.difficulties
                    
                    print(f"📚 New curriculum: {len(curriculum.topics)} topics")
                    print(f"🎚️  Difficulty mix: {difficulties.easy}E, {difficulties.medium}M, {difficulties.hard}H")
                    
                    # Show focus based on learner type
                    if len(result.mastered_topics) > len(result.failed_topics):
                        print("🚀 Focus: Advanced topics building on mastered knowledge")
                    elif len(result.failed_topics) > 0:
                        print("🔧 Focus: Remedial topics addressing knowledge gaps")
                    else:
                        print("⚖️  Focus: Balanced progression")
            else:
                print(f"❌ Failed to generate curriculum for {scenario['type']}")
                
        except Exception as e:
            print(f"❌ Error with {scenario['type']}: {e}")


async def main():
    """
    Demo script showing curriculum revision based on DPO evaluation results
    """
    
    print("🚀 Curriculum Revision Demo")
    print("=" * 60)
    
    try:
        # Test with the DPO evaluation results
        evaluation_file = "data/evaluations/evaluation_results_dpo_20250704_224746.json"
        
        if not Path(evaluation_file).exists():
            print(f"❌ Evaluation file not found: {evaluation_file}")
            print("Please run a DPO evaluation first to generate results.")
            return False
        
        print(f"📁 Loading evaluation results from: {evaluation_file}")
        
        # Use the simplified function with iteration parameter
        result = await revise_curriculum_from_dpo_results(
            evaluation_results_path=evaluation_file,
            current_curriculum_path="curriculum_test_results.json",
            accuracy_threshold=0.9,
            iteration=1  # This is the first iteration after initial training
        )
        
        if not result:
            print("❌ Failed to generate curriculum revision")
            return False
        
        print("✅ Curriculum revision completed successfully!")
        print()
        
        # Display results summary
        print("📊 Revision Summary:")
        print("-" * 40)
        print(result.revision_summary)
        print()
        
        print("🎯 Performance Analysis:")
        print(f"- Mastered topics: {len(result.mastered_topics)}")
        if result.mastered_topics:
            for topic in result.mastered_topics:
                print(f"  ✅ {topic}")
        
        print(f"- Topics needing improvement: {len(result.failed_topics)}")
        if result.failed_topics:
            for topic in result.failed_topics:
                print(f"  ❌ {topic}")
        
        print(f"- Failed questions analyzed: {result.failed_questions_count}")
        print()
        
        # Display learned topics history
        from src.core.deep_research_client import create_deep_research_client
        client = create_deep_research_client()
        
        domain = "Python Programming"  # This should match the evaluation results
        all_learned_topics = client.get_all_learned_topic_names(domain)
        
        print("📚 Historical Learned Topics:")
        print(f"- Total topics learned across all iterations: {len(all_learned_topics)}")
        if all_learned_topics:
            for i, topic in enumerate(all_learned_topics, 1):
                print(f"  {i}. {topic}")
        else:
            print("  (No topics learned yet)")
        print()
        
        # Show new curriculum details
        if result.revised_curriculum:
            curriculum = result.revised_curriculum
            print("📋 New Curriculum Overview:")
            print(f"- Domain: {curriculum.domain}")
            print(f"- Total topics: {curriculum.metadata.total_topics}")
            print(f"- Difficulty distribution:")
            print(f"  - Easy: {curriculum.metadata.difficulties.easy}")
            print(f"  - Medium: {curriculum.metadata.difficulties.medium}")
            print(f"  - Hard: {curriculum.metadata.difficulties.hard}")
            print()
            
            print("📚 New Topics to Learn:")
            print("-" * 40)
            for i, topic in enumerate(curriculum.topics[:5], 1):  # Show first 5 topics
                print(f"{i}. {topic.name} [{topic.difficulty.value}]")
                print(f"   {topic.description[:100]}{'...' if len(topic.description) > 100 else ''}")
                print()
            
            if len(curriculum.topics) > 5:
                print(f"... and {len(curriculum.topics) - 5} more topics")
                print()
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"curriculum_revision_demo_{timestamp}.json"
        
        output_data = {
            "demo_metadata": {
                "generated_at": datetime.now().isoformat(),
                "evaluation_source": evaluation_file,
                "iteration": 1,
                "demo_success": True
            },
            "revision_result": {
                "revision_summary": result.revision_summary,
                "mastered_topics": result.mastered_topics,
                "failed_topics": result.failed_topics,
                "failed_questions_count": result.failed_questions_count,
                "all_learned_topics": all_learned_topics
            },
            "revised_curriculum": result.revised_curriculum.model_dump() if result.revised_curriculum else None
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        print()
        
        print("🎉 Demo completed successfully!")
        print()
        print("Next steps:")
        print("1. Review the revised curriculum topics")
        print("2. Generate training data for the new topics")
        print("3. Run fine-tuning with the new training data")
        print("4. Evaluate the updated model")
        print("5. Repeat the revision process")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main()) 