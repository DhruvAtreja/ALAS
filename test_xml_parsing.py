#!/usr/bin/env python3
"""
Test script to analyze XML parsing issues in the training data generator
"""

import json
import re
import xml.etree.ElementTree as ET
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MockTrainingQuestion:
    """Mock training question for testing"""
    id: str
    topic_id: str
    question: str
    answer: str
    category: str
    difficulty: str
    explanation: Optional[str]
    source_topic: str

@dataclass
class MockTopic:
    """Mock topic for testing"""
    id: str
    name: str

def clean_xml_content(xml_content: str) -> str:
    """Clean XML content more intelligently"""
    
    # Remove control characters
    xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)
    
    # Only escape unescaped ampersands (not already part of entities)
    # This regex finds & that are not followed by valid entity patterns
    xml_content = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', xml_content)
    
    # Fix common XML structure issues
    xml_content = fix_common_xml_issues(xml_content)
    
    return xml_content

def fix_common_xml_issues(xml_content: str) -> str:
    """Fix common XML structure issues"""
    
    # Fix unclosed <code> tags by ensuring they're properly paired
    # Count opening and closing code tags
    open_code_count = len(re.findall(r'<code[^>]*>', xml_content))
    close_code_count = len(re.findall(r'</code>', xml_content))
    
    # If there are more opening tags than closing, add missing closing tags
    if open_code_count > close_code_count:
        missing_closes = open_code_count - close_code_count
        # Add them at the end before the last </questions> tag
        xml_content = xml_content.replace('</questions>', '</code>' * missing_closes + '</questions>')
    
    # Fix other common issues
    # Remove any stray < or > that aren't part of tags
    xml_content = re.sub(r'(?<![<>])<(?![/!?a-zA-Z])', '&lt;', xml_content)
    xml_content = re.sub(r'(?<![a-zA-Z0-9/\-"\s])>(?![<>])', '&gt;', xml_content)
    
    return xml_content

def extract_element_text(element) -> str:
    """Extract text content from XML element, including nested elements"""
    if element is None:
        return ""
    
    # Get all text content including from nested elements
    text_parts = []
    
    # Add element's direct text
    if element.text:
        text_parts.append(element.text)
    
    # Add text from nested elements
    for child in element:
        if child.tag == 'code':
            # Handle code blocks specially
            code_text = child.text or ""
            text_parts.append(f"`{code_text}`")
        else:
            # For other nested elements, just get their text
            child_text = child.text or ""
            if child_text.strip():
                text_parts.append(child_text)
        
        # Add tail text after the child element
        if child.tail:
            text_parts.append(child.tail)
    
    return " ".join(text_parts)

def fallback_parse_questions(response: str, topic: MockTopic) -> List[MockTrainingQuestion]:
    """Fallback parsing method using regex when XML parsing fails"""
    
    print(f"Attempting fallback parsing for {topic.name}")
    
    questions = []
    
    # Use regex to find question blocks
    question_pattern = r'<question-(\d+)>(.*?)</question-\d+>'
    question_matches = re.findall(question_pattern, response, re.DOTALL)
    
    for i, (question_num, question_content) in enumerate(question_matches, 1):
        try:
            # Extract individual fields using regex
            text_match = re.search(r'<text>(.*?)</text>', question_content, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', question_content, re.DOTALL)
            category_match = re.search(r'<category>(.*?)</category>', question_content, re.DOTALL)
            difficulty_match = re.search(r'<difficulty>(.*?)</difficulty>', question_content, re.DOTALL)
            explanation_match = re.search(r'<explanation>(.*?)</explanation>', question_content, re.DOTALL)
            
            if not text_match or not answer_match:
                print(f"Missing required fields in fallback question {i} for {topic.name}")
                continue
            
            # Clean the extracted text
            question_text = clean_extracted_text(text_match.group(1))
            answer_text = clean_extracted_text(answer_match.group(1))
            
            if not question_text.strip() or not answer_text.strip():
                print(f"Empty question or answer in fallback question {i} for {topic.name}")
                continue
            
            question = MockTrainingQuestion(
                id=f"{topic.id}_q{i:03d}",
                topic_id=topic.id,
                question=question_text.strip(),
                answer=answer_text.strip(),
                category=category_match.group(1).strip() if category_match else "Conceptual Understanding",
                difficulty=difficulty_match.group(1).strip() if difficulty_match else "medium",
                explanation=explanation_match.group(1).strip() if explanation_match else None,
                source_topic=topic.name
            )
            
            questions.append(question)
            
        except Exception as e:
            print(f"Failed to parse fallback question {i} for {topic.name}: {e}")
            continue
    
    print(f"Fallback parsing extracted {len(questions)} questions for {topic.name}")
    return questions

def clean_extracted_text(text: str) -> str:
    """Clean text extracted from XML"""
    if not text:
        return ""
    
    # Decode HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def parse_questions_from_response(response: str, topic: MockTopic) -> List[MockTrainingQuestion]:
    """Parse training questions from XML response with improved error handling"""
    
    try:
        # Extract XML content
        xml_match = re.search(r'<questions>(.*?)</questions>', response, re.DOTALL)
        if not xml_match:
            print(f"No <questions> tags found in response for {topic.name}")
            return []
        
        xml_content = f"<questions>{xml_match.group(1)}</questions>"
        
        # Improved XML cleaning - only escape unescaped ampersands
        xml_content = clean_xml_content(xml_content)
        
        # Parse XML
        root = ET.fromstring(xml_content)
        
        questions = []
        question_elements = [elem for elem in root if elem.tag.startswith('question')]
        
        for i, question_elem in enumerate(question_elements, 1):
            try:
                # Extract question data with better text handling
                text_elem = question_elem.find('text')
                answer_elem = question_elem.find('answer')
                category_elem = question_elem.find('category')
                difficulty_elem = question_elem.find('difficulty')
                explanation_elem = question_elem.find('explanation')
                
                if text_elem is None or answer_elem is None:
                    print(f"Missing required fields in question {i} for {topic.name}")
                    continue
                
                # Extract text content including nested elements
                question_text = extract_element_text(text_elem)
                answer_text = extract_element_text(answer_elem)
                
                # Skip if critical content is missing
                if not question_text.strip() or not answer_text.strip():
                    print(f"Empty question or answer in question {i} for {topic.name}")
                    continue
                
                question = MockTrainingQuestion(
                    id=f"{topic.id}_q{i:03d}",
                    topic_id=topic.id,
                    question=question_text.strip(),
                    answer=answer_text.strip(),
                    category=category_elem.text.strip() if category_elem is not None and category_elem.text else "Conceptual Understanding",
                    difficulty=difficulty_elem.text.strip() if difficulty_elem is not None and difficulty_elem.text else "medium",
                    explanation=explanation_elem.text.strip() if explanation_elem is not None and explanation_elem.text else None,
                    source_topic=topic.name
                )
                
                questions.append(question)
                
            except Exception as e:
                print(f"Failed to parse question {i} for {topic.name}: {e}")
                continue
        
        print(f"Successfully parsed {len(questions)} questions for {topic.name}")
        return questions
        
    except ET.ParseError as e:
        print(f"XML parsing error for {topic.name}: {e}")
        # Try fallback parsing
        return fallback_parse_questions(response, topic)
    except Exception as e:
        print(f"Error parsing questions for {topic.name}: {e}")
        return []

def load_test_data() -> Dict[str, str]:
    """Load XML test data from text files"""
    test_data = {}
    
    # Load each text file
    for i in range(1, 5):  # text1.txt to text4.txt
        file_path = Path(f"text{i}.txt")
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                test_data[f"sample_{i}"] = content
        else:
            print(f"Warning: {file_path} not found")
    
    return test_data

def test_xml_parsing():
    """Test the XML parsing function with sample data"""
    
    print("🧪 Starting XML parsing tests...")
    
    # Load test data
    test_data = load_test_data()
    
    if not test_data:
        print("❌ No test data found! Make sure text1.txt to text4.txt exist in the current directory.")
        return
    
    # Results storage
    results = {
        "test_metadata": {
            "test_run_at": datetime.now().isoformat(),
            "total_samples": len(test_data),
            "samples_tested": 0,
            "successful_parses": 0,
            "failed_parses": 0
        },
        "sample_results": {}
    }
    
    for sample_name, xml_content in test_data.items():
        print(f"\n📝 Testing {sample_name}...")
        
        # Create mock topic for this test
        topic = MockTopic(id=f"test_topic_{sample_name}", name=f"Topic for {sample_name}")
        
        sample_result = {
            "sample_name": sample_name,
            "original_length": len(xml_content),
            "success": False,
            "questions_parsed": 0,
            "error": None,
            "error_details": None,
            "questions": [],
            "parsing_issues": []
        }
        
        try:
            # Test the parsing function
            questions = parse_questions_from_response(xml_content, topic)
            
            sample_result["success"] = True
            sample_result["questions_parsed"] = len(questions)
            
            # Convert questions to dict for JSON serialization
            for q in questions:
                question_dict = {
                    "id": q.id,
                    "topic_id": q.topic_id,
                    "question": q.question[:200] + "..." if len(q.question) > 200 else q.question,  # Truncate for readability
                    "answer": q.answer[:300] + "..." if len(q.answer) > 300 else q.answer,  # Truncate for readability
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "explanation": q.explanation[:200] + "..." if q.explanation and len(q.explanation) > 200 else q.explanation,
                    "source_topic": q.source_topic,
                    "question_length": len(q.question),
                    "answer_length": len(q.answer),
                    "has_explanation": q.explanation is not None
                }
                sample_result["questions"].append(question_dict)
            
            results["test_metadata"]["successful_parses"] += 1
            print(f"✅ Successfully parsed {len(questions)} questions")
            
            # Check for potential issues
            if len(questions) == 0:
                sample_result["parsing_issues"].append("No questions were parsed from the XML")
            
            # Check for empty content
            empty_questions = [q for q in questions if not q.question.strip()]
            if empty_questions:
                sample_result["parsing_issues"].append(f"{len(empty_questions)} questions have empty text")
            
            empty_answers = [q for q in questions if not q.answer.strip()]
            if empty_answers:
                sample_result["parsing_issues"].append(f"{len(empty_answers)} questions have empty answers")
                
        except Exception as e:
            sample_result["success"] = False
            sample_result["error"] = str(e)
            sample_result["error_details"] = traceback.format_exc()
            
            results["test_metadata"]["failed_parses"] += 1
            print(f"❌ Failed to parse: {e}")
        
        results["sample_results"][sample_name] = sample_result
        results["test_metadata"]["samples_tested"] += 1
    
    # Additional analysis
    print(f"\n📊 Test Summary:")
    print(f"   Total samples: {results['test_metadata']['total_samples']}")
    print(f"   Successful: {results['test_metadata']['successful_parses']}")
    print(f"   Failed: {results['test_metadata']['failed_parses']}")
    
    if results["test_metadata"]["successful_parses"] > 0:
        total_questions = sum(r["questions_parsed"] for r in results["sample_results"].values() if r["success"])
        print(f"   Total questions parsed: {total_questions}")
    
    # Save results to JSON
    output_file = f"xml_parsing_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    return results

def analyze_xml_structure():
    """Analyze the structure of the XML content to identify potential issues"""
    
    print("\n🔍 Analyzing XML structure...")
    
    test_data = load_test_data()
    
    for sample_name, content in test_data.items():
        print(f"\n📄 Analyzing {sample_name}:")
        
        # Check for questions tags
        has_questions_tag = "<questions>" in content and "</questions>" in content
        print(f"   Has <questions> tags: {has_questions_tag}")
        
        # Count question elements
        question_pattern = r'<question-\d+>'
        question_matches = re.findall(question_pattern, content)
        print(f"   Question elements found: {len(question_matches)}")
        
        # Check for potential problematic characters
        problematic_chars = ['&', '<', '>', '"', "'"]
        char_counts = {}
        for char in problematic_chars:
            count = content.count(char)
            if count > 0:
                char_counts[char] = count
        
        if char_counts:
            print(f"   Problematic characters: {char_counts}")
        
        # Check for code blocks that might have unescaped content
        code_pattern = r'<code[^>]*>.*?</code>'
        code_matches = re.findall(code_pattern, content, re.DOTALL)
        if code_matches:
            print(f"   Code blocks found: {len(code_matches)}")
        
        # Check for HTML entities
        html_entities = ['&amp;', '&lt;', '&gt;', '&quot;', '&#']
        entity_counts = {}
        for entity in html_entities:
            count = content.count(entity)
            if count > 0:
                entity_counts[entity] = count
        
        if entity_counts:
            print(f"   HTML entities: {entity_counts}")

def analyze_specific_parsing_issues():
    """Analyze specific issues that might cause XML parsing to fail"""
    
    print("\n🔧 Analyzing specific parsing issues...")
    
    test_data = load_test_data()
    
    for sample_name, content in test_data.items():
        print(f"\n🧐 Deep analysis of {sample_name}:")
        
        # Try to extract questions block
        xml_match = re.search(r'<questions>(.*?)</questions>', content, re.DOTALL)
        if xml_match:
            xml_content = f"<questions>{xml_match.group(1)}</questions>"
            print(f"   Extracted XML block length: {len(xml_content)}")
            
            # Check for unescaped ampersands
            unescaped_amps = re.findall(r'&(?!amp;|lt;|gt;|quot;|#)', xml_content)
            if unescaped_amps:
                print(f"   Unescaped ampersands found: {len(unescaped_amps)}")
                print(f"   Examples: {unescaped_amps[:5]}")
            
            # Check for control characters
            control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', xml_content)
            if control_chars:
                print(f"   Control characters found: {len(control_chars)}")
            
            # Try basic XML parsing to see where it fails
            try:
                # Clean XML content like the actual function does
                cleaned_xml = xml_content.replace('&', '&amp;')
                cleaned_xml = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_xml)
                
                root = ET.fromstring(cleaned_xml)
                question_elements = [elem for elem in root if elem.tag.startswith('question')]
                print(f"   ✅ XML parses successfully with {len(question_elements)} question elements")
                
                # Check each question element
                for i, elem in enumerate(question_elements[:3], 1):  # Check first 3
                    text_elem = elem.find('text')
                    answer_elem = elem.find('answer')
                    print(f"   Question {i}: text={'✅' if text_elem is not None else '❌'}, answer={'✅' if answer_elem is not None else '❌'}")
                    
            except ET.ParseError as e:
                print(f"   ❌ XML parsing failed: {e}")
                
                # Try to find the problematic part
                lines = cleaned_xml.split('\n')
                if hasattr(e, 'lineno') and e.lineno is not None:
                    problem_line = e.lineno
                    if problem_line <= len(lines):
                        print(f"   Problem around line {problem_line}: {lines[problem_line-1][:100]}")
        else:
            print(f"   ❌ No <questions> block found")

if __name__ == "__main__":
    print("🚀 XML Parsing Test Suite")
    print("=" * 50)
    
    # First analyze structure
    analyze_xml_structure()
    
    print("\n" + "=" * 50)
    
    # Analyze specific parsing issues
    analyze_specific_parsing_issues()
    
    print("\n" + "=" * 50)
    
    # Then test parsing
    test_xml_parsing()
    
    print("\n✨ Test complete!") 