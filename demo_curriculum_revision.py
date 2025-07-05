#!/usr/bin/env python3
"""
Simple demonstration of curriculum revision functionality
"""

import asyncio
import sys
import json
from pathlib import Path

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
                evaluation_file_path=eval_file,
                current_curriculum_file=current_curriculum_file if Path(current_curriculum_file).exists() else None,
                accuracy_threshold=0.9,
                save_results=True
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
    """Run all demonstrations"""
    
    print("🚀 Curriculum Revision Demonstration Suite")
    print("=" * 60)
    
    # Basic demonstration
    await demo_curriculum_revision()
    
    # Different learner types
    await demo_different_learner_types()
    
    print(f"\n🏁 All demonstrations complete!")
    print(f"\nFiles generated:")
    print("- curriculum_revision_*.json (revised curricula)")
    print("- Check timestamps in filenames for latest results")
    
    print(f"\nKey Features Demonstrated:")
    print("✅ Automatic analysis of DPO evaluation results")
    print("✅ Separation of mastered vs struggling topics")
    print("✅ Extraction of failed questions with explanations")
    print("✅ Generation of targeted curriculum addressing knowledge gaps")
    print("✅ Adaptation for different learner performance levels")
    print("✅ Building advanced topics on mastered knowledge")
    print("✅ Deep research API integration for curriculum generation")


if __name__ == "__main__":
    result = asyncio.run(main()) 