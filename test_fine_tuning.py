#!/usr/bin/env python3
"""
Test script for fine-tuning functionality
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.fine_tuner import create_fine_tuner, FineTuningHyperparameters, FineTuningError
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_file_upload():
    """Test file upload functionality"""
    print("\n📂 Testing File Upload...")
    print("=" * 50)
    
    try:
        fine_tuner = create_fine_tuner()
        
        # Use our existing training file
        training_file = "training_data_python_programming_20250630_021709_openai.jsonl"
        
        if not Path(training_file).exists():
            print(f"❌ Training file not found: {training_file}")
            print("Please make sure you have generated training data first.")
            return False, None
        
        print(f"📁 Uploading: {training_file}")
        
        # Upload the file
        result = fine_tuner.upload_training_file(training_file)
        
        print("✅ File upload successful!")
        print(f"File ID: {result.file_id}")
        print(f"Filename: {result.filename}")
        print(f"Size: {result.bytes:,} bytes")
        print(f"Status: {result.status}")
        
        return True, result.file_id
        
    except FineTuningError as e:
        print(f"❌ Fine-tuning error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None


async def test_job_creation(training_file_id: str):
    """Test fine-tuning job creation"""
    print("\n🏗️  Testing Job Creation...")
    print("=" * 50)
    
    try:
        fine_tuner = create_fine_tuner()
        
        # Create hyperparameters for testing
        hyperparameters = FineTuningHyperparameters(
            n_epochs=3,  # Small number for testing
            batch_size="auto",
            learning_rate_multiplier="auto"
        )
        
        print(f"📊 Creating job with training file: {training_file_id}")
        print(f"🤖 Model: gpt-4.1-nano-2025-04-14 (cheapest for testing)")
        print(f"⚙️  Epochs: {hyperparameters.n_epochs}")
        
        # Create job (without waiting for completion)
        result = fine_tuner.create_fine_tuning_job(
            training_file_id=training_file_id,
            model="gpt-4.1-nano-2025-04-14",  # Use nano for testing (cheaper)
            hyperparameters=hyperparameters,
            suffix="python-test"
        )
        
        print("✅ Job creation successful!")
        print(f"Job ID: {result.job_id}")
        print(f"Status: {result.status}")
        print(f"Model: {result.model}")
        
        return True, result.job_id
        
    except FineTuningError as e:
        print(f"❌ Fine-tuning error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None


async def test_job_status(job_id: str):
    """Test job status checking"""
    print("\n📊 Testing Job Status...")
    print("=" * 50)
    
    try:
        fine_tuner = create_fine_tuner()
        
        print(f"🔍 Checking status for: {job_id}")
        
        result = fine_tuner.get_job_status(job_id)
        
        print("✅ Status check successful!")
        print(f"Job ID: {result.job_id}")
        print(f"Status: {result.status}")
        print(f"Model: {result.model}")
        
        if result.fine_tuned_model:
            print(f"Fine-tuned model: {result.fine_tuned_model}")
            
        if result.trained_tokens:
            print(f"Tokens trained: {result.trained_tokens:,}")
            
        if result.error:
            print(f"Error: {result.error}")
            
        return True
        
    except FineTuningError as e:
        print(f"❌ Fine-tuning error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def test_list_jobs():
    """Test listing jobs"""
    print("\n📋 Testing Job Listing...")
    print("=" * 50)
    
    try:
        fine_tuner = create_fine_tuner()
        
        print("📋 Fetching recent jobs...")
        
        jobs = fine_tuner.list_fine_tuning_jobs(limit=5)
        
        print(f"✅ Found {len(jobs)} jobs")
        
        if jobs:
            print("\nRecent jobs:")
            for i, job in enumerate(jobs[:3], 1):  # Show first 3
                print(f"{i}. {job.job_id} - {job.status} ({job.model})")
                if job.fine_tuned_model:
                    print(f"   → {job.fine_tuned_model}")
        else:
            print("No jobs found.")
            
        return True
        
    except FineTuningError as e:
        print(f"❌ Fine-tuning error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def test_training_file_validation():
    """Test training file validation"""
    print("\n🔍 Testing Training File Validation...")
    print("=" * 50)
    
    training_file = "training_data_python_programming_20250630_021709_openai.jsonl"
    
    if not Path(training_file).exists():
        print(f"❌ Training file not found: {training_file}")
        return False
    
    try:
        # Check file format
        with open(training_file, 'r') as f:
            lines_checked = 0
            valid_lines = 0
            
            for line in f:
                if lines_checked >= 5:  # Check first 5 lines
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Validate required fields
                    if "messages" not in data:
                        print(f"❌ Line {lines_checked + 1}: Missing 'messages' field")
                        continue
                        
                    messages = data["messages"]
                    if not isinstance(messages, list):
                        print(f"❌ Line {lines_checked + 1}: 'messages' must be a list")
                        continue
                        
                    # Check message format
                    valid_message = True
                    for msg in messages:
                        if not isinstance(msg, dict):
                            valid_message = False
                            break
                        if "role" not in msg or "content" not in msg:
                            valid_message = False
                            break
                            
                    if valid_message:
                        valid_lines += 1
                        print(f"✅ Line {lines_checked + 1}: Valid format")
                    else:
                        print(f"❌ Line {lines_checked + 1}: Invalid message format")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Line {lines_checked + 1}: JSON decode error: {e}")
                    
                lines_checked += 1
            
            print(f"\n📊 Validation Summary:")
            print(f"Lines checked: {lines_checked}")
            print(f"Valid lines: {valid_lines}")
            print(f"Success rate: {valid_lines/lines_checked*100:.1f}%")
            
            return valid_lines > 0
            
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False


async def main():
    """Run all tests"""
    print("🧪 Fine-Tuning Module Tests")
    print("=" * 60)
    
    # Test 1: Validate training file
    validation_success = await test_training_file_validation()
    if not validation_success:
        print("\n❌ Training file validation failed. Cannot proceed with API tests.")
        return False
    
    # Test 2: List existing jobs
    list_success = await test_list_jobs()
    
    # Test 3: File upload
    upload_success, file_id = await test_file_upload()
    if not upload_success or file_id is None:
        print("\n❌ File upload failed. Skipping remaining tests.")
        return False
    
    # Test 4: Job creation
    job_success, job_id = await test_job_creation(file_id)
    if not job_success or job_id is None:
        print("\n❌ Job creation failed. Skipping status test.")
        return False
    
    # Test 5: Job status
    status_success = await test_job_status(job_id)
    
    # Summary
    print("\n🎯 Test Summary")
    print("=" * 30)
    print(f"Validation: {'✅' if validation_success else '❌'}")
    print(f"List jobs: {'✅' if list_success else '❌'}")
    print(f"File upload: {'✅' if upload_success else '❌'}")
    print(f"Job creation: {'✅' if job_success else '❌'}")
    print(f"Status check: {'✅' if status_success else '❌'}")
    
    overall_success = all([validation_success, upload_success, job_success, status_success])
    
    if overall_success:
        print("\n🎉 All tests passed!")
        print(f"\n💡 Your fine-tuning job has been created: {job_id}")
        print("You can check its status with:")
        print(f"python run_fine_tuning.py --status {job_id}")
    else:
        print("\n❌ Some tests failed")
        
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 