#!/usr/bin/env python3
"""
Compare performance between base model and fine-tuned model
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse

def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {file_path}: {e}")
        return {}

def calculate_overall_accuracy(results: Dict[str, Any]) -> float:
    """Calculate overall accuracy from evaluation results"""
    if not results or 'evaluation_summary' not in results:
        return 0.0
    
    summary = results['evaluation_summary']
    if 'total_questions' in summary and 'total_correct' in summary:
        total = summary['total_questions']
        correct = summary['total_correct']
        return (correct / total * 100) if total > 0 else 0.0
    
    return 0.0

def get_category_breakdown(results: Dict[str, Any]) -> Dict[str, float]:
    """Get accuracy breakdown by category"""
    breakdown = {}
    
    if not results or 'evaluation_results' not in results:
        return breakdown
    
    for topic_result in results['evaluation_results']:
        if 'category_results' in topic_result:
            for category, cat_results in topic_result['category_results'].items():
                if 'accuracy' in cat_results:
                    breakdown[category] = cat_results['accuracy'] * 100
    
    return breakdown

def get_topic_breakdown(results: Dict[str, Any]) -> Dict[str, float]:
    """Get accuracy breakdown by topic"""
    breakdown = {}
    
    if not results or 'evaluation_results' not in results:
        return breakdown
    
    for topic_result in results['evaluation_results']:
        if 'topic_name' in topic_result and 'overall_accuracy' in topic_result:
            topic_name = topic_result['topic_name']
            accuracy = topic_result['overall_accuracy'] * 100
            breakdown[topic_name] = accuracy
    
    return breakdown

def print_comparison_table(base_results: Dict[str, Any], 
                          finetuned_results: Dict[str, Any]):
    """Print comparison table"""
    
    print("\n📊 Model Performance Comparison")
    print("=" * 80)
    
    # Overall accuracy
    base_accuracy = calculate_overall_accuracy(base_results)
    finetuned_accuracy = calculate_overall_accuracy(finetuned_results)
    improvement = finetuned_accuracy - base_accuracy
    
    print(f"\n🎯 Overall Accuracy:")
    print(f"Base Model:       {base_accuracy:.1f}%")
    print(f"Fine-tuned Model: {finetuned_accuracy:.1f}%")
    print(f"Improvement:      {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"🎉 Fine-tuning improved performance by {improvement:.1f}%!")
    elif improvement < 0:
        print(f"⚠️  Fine-tuning decreased performance by {abs(improvement):.1f}%")
    else:
        print("➡️  No change in performance")
    
    # Category breakdown
    base_categories = get_category_breakdown(base_results)
    finetuned_categories = get_category_breakdown(finetuned_results)
    
    if base_categories and finetuned_categories:
        print(f"\n📋 Performance by Category:")
        print("-" * 60)
        print(f"{'Category':<25} {'Base':<10} {'Fine-tuned':<12} {'Change':<10}")
        print("-" * 60)
        
        all_categories = set(base_categories.keys()) | set(finetuned_categories.keys())
        for category in sorted(all_categories):
            base_acc = base_categories.get(category, 0)
            ft_acc = finetuned_categories.get(category, 0)
            change = ft_acc - base_acc
            
            change_str = f"{change:+.1f}%" if change != 0 else "0.0%"
            print(f"{category:<25} {base_acc:<9.1f}% {ft_acc:<11.1f}% {change_str:<10}")
    
    # Topic breakdown (if available)
    base_topics = get_topic_breakdown(base_results)
    finetuned_topics = get_topic_breakdown(finetuned_results)
    
    if base_topics and finetuned_topics:
        print(f"\n📚 Performance by Topic (Top 10):")
        print("-" * 80)
        print(f"{'Topic':<40} {'Base':<10} {'Fine-tuned':<12} {'Change':<10}")
        print("-" * 80)
        
        # Show topics with biggest improvements first
        all_topics = set(base_topics.keys()) | set(finetuned_topics.keys())
        topic_improvements = []
        
        for topic in all_topics:
            base_acc = base_topics.get(topic, 0)
            ft_acc = finetuned_topics.get(topic, 0)
            change = ft_acc - base_acc
            topic_improvements.append((topic, base_acc, ft_acc, change))
        
        # Sort by improvement (descending)
        topic_improvements.sort(key=lambda x: x[3], reverse=True)
        
        for topic, base_acc, ft_acc, change in topic_improvements[:10]:
            topic_short = topic[:37] + "..." if len(topic) > 40 else topic
            change_str = f"{change:+.1f}%" if change != 0 else "0.0%"
            print(f"{topic_short:<40} {base_acc:<9.1f}% {ft_acc:<11.1f}% {change_str:<10}")

def get_model_info(results: Dict[str, Any]) -> Dict[str, str]:
    """Extract model information from results"""
    info = {}
    
    if 'metadata' in results:
        metadata = results['metadata']
        info['model'] = metadata.get('model_tested', 'Unknown')
        info['test_date'] = metadata.get('test_timestamp', 'Unknown')
        info['total_questions'] = str(metadata.get('total_questions', 'Unknown'))
    
    return info

def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between base and fine-tuned models"
    )
    
    parser.add_argument(
        "base_results",
        help="Path to base model evaluation results JSON file"
    )
    
    parser.add_argument(
        "finetuned_results", 
        help="Path to fine-tuned model evaluation results JSON file"
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed breakdown"
    )
    
    args = parser.parse_args()
    
    # Load results
    print("📁 Loading evaluation results...")
    base_results = load_evaluation_results(args.base_results)
    finetuned_results = load_evaluation_results(args.finetuned_results)
    
    if not base_results:
        print(f"❌ Could not load base results from {args.base_results}")
        sys.exit(1)
    
    if not finetuned_results:
        print(f"❌ Could not load fine-tuned results from {args.finetuned_results}")
        sys.exit(1)
    
    # Show model info
    base_info = get_model_info(base_results)
    ft_info = get_model_info(finetuned_results)
    
    print(f"\n🤖 Models Compared:")
    print(f"Base Model:       {base_info.get('model', 'Unknown')}")
    print(f"Fine-tuned Model: {ft_info.get('model', 'Unknown')}")
    print(f"Questions:        {base_info.get('total_questions', 'Unknown')}")
    
    # Print comparison
    print_comparison_table(base_results, finetuned_results)
    
    # Summary
    base_accuracy = calculate_overall_accuracy(base_results)
    ft_accuracy = calculate_overall_accuracy(finetuned_results)
    improvement = ft_accuracy - base_accuracy
    
    print(f"\n💡 Summary:")
    if improvement > 5:
        print(f"🚀 Excellent improvement! Fine-tuning boosted performance by {improvement:.1f}%")
    elif improvement > 1:
        print(f"✅ Good improvement! Fine-tuning helped by {improvement:.1f}%")
    elif improvement > -1:
        print(f"➡️  Minimal change in performance ({improvement:+.1f}%)")
    else:
        print(f"⚠️  Performance decreased by {abs(improvement):.1f}%. Consider:")
        print("   - Adjusting hyperparameters")
        print("   - Increasing training data quality")
        print("   - Reducing epochs to avoid overfitting")

if __name__ == "__main__":
    main() 