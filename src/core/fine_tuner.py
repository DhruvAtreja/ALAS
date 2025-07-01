"""
OpenAI Fine-Tuning Module for Supervised Fine-Tuning
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import os

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from enum import Enum

try:
    from ..config.settings import settings
    from ..utils.logger import get_logger, log_api_call, log_cost, log_error
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import settings
    from utils.logger import get_logger, log_api_call, log_cost, log_error

logger = get_logger(__name__)


class FineTuningStatus(str, Enum):
    """Fine-tuning job status"""
    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuningMethod(str, Enum):
    """Fine-tuning methods"""
    SUPERVISED = "supervised"
    DPO = "dpo"  # Direct Preference Optimization


class FileUploadResult(BaseModel):
    """Result of file upload to OpenAI"""
    file_id: str = Field(..., description="OpenAI file ID")
    filename: str = Field(..., description="Original filename")
    bytes: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="File processing status")
    purpose: str = Field(..., description="File purpose (fine-tune)")
    created_at: int = Field(..., description="Upload timestamp")


class FineTuningJobResult(BaseModel):
    """Result of fine-tuning job creation/status"""
    job_id: str = Field(..., description="Fine-tuning job ID")
    model: str = Field(..., description="Base model used")
    status: FineTuningStatus = Field(..., description="Current job status")
    fine_tuned_model: Optional[str] = Field(default=None, description="Fine-tuned model ID when completed")
    training_file: str = Field(..., description="Training file ID")
    validation_file: Optional[str] = Field(default=None, description="Validation file ID")
    created_at: int = Field(..., description="Job creation timestamp")
    finished_at: Optional[int] = Field(default=None, description="Job completion timestamp")
    trained_tokens: Optional[int] = Field(default=None, description="Number of tokens trained")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Training hyperparameters")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error details if failed")
    estimated_finish: Optional[int] = Field(default=None, description="Estimated completion time")


class FineTuningHyperparameters(BaseModel):
    """Hyperparameters for fine-tuning"""
    n_epochs: Optional[int] = Field(default=None, ge=1, le=50, description="Number of training epochs")
    batch_size: Optional[Union[int, str]] = Field(default="auto", description="Batch size or 'auto'")
    learning_rate_multiplier: Optional[Union[float, str]] = Field(default="auto", description="Learning rate multiplier or 'auto'")


class FineTuningError(Exception):
    """Custom exception for fine-tuning errors"""
    pass


class OpenAIFineTuner:
    """Client for OpenAI's Fine-Tuning API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai.api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required for fine-tuning")
            
        self.client = OpenAI(api_key=self.api_key, timeout=settings.openai.timeout)
        self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=settings.openai.timeout)
        
        # Supported models for fine-tuning
        self.supported_models = [
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14", 
            "gpt-4.1-nano-2025-04-14"
        ]
    
    def _estimate_cost(self, training_tokens: int, model: str) -> float:
        """Estimate fine-tuning cost based on tokens and model"""
        # These are estimated rates - actual rates may vary
        cost_per_1k_tokens = {
            "gpt-4.1-2025-04-14": 0.025,
            "gpt-4.1-mini-2025-04-14": 0.003,
            "gpt-4.1-nano-2025-04-14": 0.0008
        }
        
        rate = cost_per_1k_tokens.get(model, 0.025)  # Default to highest rate
        return (training_tokens / 1000) * rate
    
    def upload_training_file(self, file_path: Union[str, Path]) -> FileUploadResult:
        """
        Upload training data file to OpenAI
        
        Args:
            file_path: Path to JSONL training file
            
        Returns:
            FileUploadResult with upload details
        """
        start_time = datetime.now()
        
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"Training file not found: {file_path_obj}")
            
            if file_path_obj.suffix.lower() != '.jsonl':
                raise ValueError("Training file must be in JSONL format")
            
            logger.info(f"Uploading training file: {file_path_obj}")
            
            with open(file_path_obj, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            # Log metrics
            duration = (datetime.now() - start_time).total_seconds()
            log_api_call("OpenAI File Upload", "files", {"filename": file_path_obj.name}, duration)
            
            result = FileUploadResult(
                file_id=response.id,
                filename=response.filename,
                bytes=response.bytes,
                status=response.status,
                purpose=response.purpose,
                created_at=response.created_at
            )
            
            logger.info(f"File uploaded successfully: {result.file_id}")
            return result
            
        except Exception as e:
            log_error(e, {"file_path": str(file_path)})
            raise FineTuningError(f"Failed to upload training file: {e}")
    
    def create_fine_tuning_job(self,
                             training_file_id: str,
                             model: str = "gpt-4.1-2025-04-14",
                             validation_file_id: Optional[str] = None,
                             hyperparameters: Optional[FineTuningHyperparameters] = None,
                             suffix: Optional[str] = None) -> FineTuningJobResult:
        """
        Create fine-tuning job
        
        Args:
            training_file_id: ID of uploaded training file
            model: Base model to fine-tune
            validation_file_id: Optional validation file ID
            hyperparameters: Training hyperparameters
            suffix: Optional suffix for model name
            
        Returns:
            FineTuningJobResult with job details
        """
        start_time = datetime.now()
        
        try:
            if model not in self.supported_models:
                logger.warning(f"Model {model} may not support fine-tuning")
            
            # Prepare job parameters
            job_params = {
                "training_file": training_file_id,
                "model": model,
                "method": {
                    "type": "supervised"
                }
            }
            
            if validation_file_id:
                job_params["validation_file"] = validation_file_id
                
            if suffix:
                job_params["suffix"] = suffix
            
            # Add hyperparameters if provided
            if hyperparameters:
                hyperparams_dict = {}
                if hyperparameters.n_epochs is not None:
                    hyperparams_dict["n_epochs"] = hyperparameters.n_epochs
                if hyperparameters.batch_size is not None:
                    hyperparams_dict["batch_size"] = hyperparameters.batch_size
                if hyperparameters.learning_rate_multiplier is not None:
                    hyperparams_dict["learning_rate_multiplier"] = hyperparameters.learning_rate_multiplier
                
                if hyperparams_dict:
                    job_params["method"]["supervised"] = {
                        "hyperparameters": hyperparams_dict
                    }
            
            logger.info(f"Creating fine-tuning job for model: {model}")
            logger.debug(f"Job parameters: {job_params}")
            
            response = self.client.fine_tuning.jobs.create(**job_params)
            
            # Log metrics
            duration = (datetime.now() - start_time).total_seconds()
            log_api_call("OpenAI Fine-Tuning Job", model, {"training_file": training_file_id}, duration)
            
            # Convert hyperparameters and error to dicts if they exist
            hyperparams_dict = None
            if hasattr(response, 'hyperparameters') and response.hyperparameters:
                hyperparams_dict = response.hyperparameters.__dict__ if hasattr(response.hyperparameters, '__dict__') else dict(response.hyperparameters)
            
            error_dict = None
            if hasattr(response, 'error') and response.error:
                error_dict = response.error.__dict__ if hasattr(response.error, '__dict__') else dict(response.error)
            
            result = FineTuningJobResult(
                job_id=response.id,
                model=response.model,
                status=FineTuningStatus(response.status),
                fine_tuned_model=response.fine_tuned_model,
                training_file=response.training_file,
                validation_file=response.validation_file,
                created_at=response.created_at,
                finished_at=response.finished_at,
                trained_tokens=response.trained_tokens,
                hyperparameters=hyperparams_dict,
                error=error_dict,
                estimated_finish=response.estimated_finish if hasattr(response, 'estimated_finish') else None
            )
            
            logger.info(f"Fine-tuning job created: {result.job_id}")
            return result
            
        except Exception as e:
            log_error(e, {"model": model, "training_file": training_file_id})
            raise FineTuningError(f"Failed to create fine-tuning job: {e}")
    
    def get_job_status(self, job_id: str) -> FineTuningJobResult:
        """
        Get current status of fine-tuning job
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            FineTuningJobResult with current status
        """
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            # Convert hyperparameters and error to dicts if they exist
            hyperparams_dict = None
            if hasattr(response, 'hyperparameters') and response.hyperparameters:
                hyperparams_dict = response.hyperparameters.__dict__ if hasattr(response.hyperparameters, '__dict__') else dict(response.hyperparameters)
            
            error_dict = None
            if hasattr(response, 'error') and response.error:
                error_dict = response.error.__dict__ if hasattr(response.error, '__dict__') else dict(response.error)
            
            result = FineTuningJobResult(
                job_id=response.id,
                model=response.model,
                status=FineTuningStatus(response.status),
                fine_tuned_model=response.fine_tuned_model,
                training_file=response.training_file,
                validation_file=response.validation_file,
                created_at=response.created_at,
                finished_at=response.finished_at,
                trained_tokens=response.trained_tokens,
                hyperparameters=hyperparams_dict,
                error=error_dict,
                estimated_finish=response.estimated_finish if hasattr(response, 'estimated_finish') else None
            )
            
            return result
            
        except Exception as e:
            log_error(e, {"job_id": job_id})
            raise FineTuningError(f"Failed to get job status: {e}")
    
    def wait_for_job_completion(self, 
                              job_id: str,
                              poll_interval: int = 60,
                              timeout: Optional[int] = None) -> FineTuningJobResult:
        """
        Wait for fine-tuning job to complete
        
        Args:
            job_id: Fine-tuning job ID
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
            
        Returns:
            FineTuningJobResult when job completes
        """
        start_time = time.time()
        logger.info(f"Waiting for fine-tuning job {job_id} to complete...")
        
        while True:
            try:
                status = self.get_job_status(job_id)
                
                logger.info(f"Job {job_id} status: {status.status}")
                
                if status.status in [FineTuningStatus.SUCCEEDED, FineTuningStatus.FAILED, FineTuningStatus.CANCELLED]:
                    if status.status == FineTuningStatus.SUCCEEDED:
                        logger.info(f"Fine-tuning completed! Model: {status.fine_tuned_model}")
                        if status.trained_tokens:
                            estimated_cost = self._estimate_cost(status.trained_tokens, status.model)
                            log_cost("Fine-Tuning", "training", estimated_cost)
                    else:
                        logger.error(f"Fine-tuning failed with status: {status.status}")
                        if status.error:
                            logger.error(f"Error details: {status.error}")
                    
                    return status
                
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise FineTuningError(f"Job {job_id} timed out after {timeout} seconds")
                
                # Wait before next check
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                return self.get_job_status(job_id)
            except Exception as e:
                logger.error(f"Error checking job status: {e}")
                time.sleep(poll_interval)
    
    def list_fine_tuning_jobs(self, limit: int = 20) -> List[FineTuningJobResult]:
        """
        List recent fine-tuning jobs
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of FineTuningJobResult objects
        """
        try:
            response = self.client.fine_tuning.jobs.list(limit=limit)
            
            jobs = []
            for job in response.data:
                # Convert hyperparameters and error to dicts if they exist
                hyperparams_dict = None
                if hasattr(job, 'hyperparameters') and job.hyperparameters:
                    hyperparams_dict = job.hyperparameters.__dict__ if hasattr(job.hyperparameters, '__dict__') else dict(job.hyperparameters)
                
                error_dict = None
                if hasattr(job, 'error') and job.error:
                    error_dict = job.error.__dict__ if hasattr(job.error, '__dict__') else dict(job.error)
                
                result = FineTuningJobResult(
                    job_id=job.id,
                    model=job.model,
                    status=FineTuningStatus(job.status),
                    fine_tuned_model=job.fine_tuned_model,
                    training_file=job.training_file,
                    validation_file=job.validation_file,
                    created_at=job.created_at,
                    finished_at=job.finished_at,
                    trained_tokens=job.trained_tokens,
                    hyperparameters=hyperparams_dict,
                    error=error_dict,
                    estimated_finish=job.estimated_finish if hasattr(job, 'estimated_finish') else None
                )
                jobs.append(result)
            
            return jobs
            
        except Exception as e:
            log_error(e, {"limit": limit})
            raise FineTuningError(f"Failed to list fine-tuning jobs: {e}")
    
    def cancel_job(self, job_id: str) -> FineTuningJobResult:
        """
        Cancel a fine-tuning job
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            FineTuningJobResult with updated status
        """
        try:
            response = self.client.fine_tuning.jobs.cancel(job_id)
            
            # Convert hyperparameters and error to dicts if they exist
            hyperparams_dict = None
            if hasattr(response, 'hyperparameters') and response.hyperparameters:
                hyperparams_dict = response.hyperparameters.__dict__ if hasattr(response.hyperparameters, '__dict__') else dict(response.hyperparameters)
            
            error_dict = None
            if hasattr(response, 'error') and response.error:
                error_dict = response.error.__dict__ if hasattr(response.error, '__dict__') else dict(response.error)
            
            result = FineTuningJobResult(
                job_id=response.id,
                model=response.model,
                status=FineTuningStatus(response.status),
                fine_tuned_model=response.fine_tuned_model,
                training_file=response.training_file,
                validation_file=response.validation_file,
                created_at=response.created_at,
                finished_at=response.finished_at,
                trained_tokens=response.trained_tokens,
                hyperparameters=hyperparams_dict,
                error=error_dict,
                estimated_finish=response.estimated_finish if hasattr(response, 'estimated_finish') else None
            )
            
            logger.info(f"Fine-tuning job cancelled: {job_id}")
            return result
            
        except Exception as e:
            log_error(e, {"job_id": job_id})
            raise FineTuningError(f"Failed to cancel job: {e}")
    
    def fine_tune_from_file(self,
                          training_file_path: str,
                          model: str = "gpt-4.1-2025-04-14",
                          validation_file_path: Optional[str] = None,
                          hyperparameters: Optional[FineTuningHyperparameters] = None,
                          suffix: Optional[str] = None,
                          wait_for_completion: bool = True,
                          poll_interval: int = 60) -> FineTuningJobResult:
        """
        Complete fine-tuning workflow: upload file, create job, and optionally wait
        
        Args:
            training_file_path: Path to training JSONL file
            model: Base model to fine-tune
            validation_file_path: Optional validation file path
            hyperparameters: Training hyperparameters
            suffix: Optional suffix for model name
            wait_for_completion: Whether to wait for job completion
            poll_interval: Seconds between status checks
            
        Returns:
            FineTuningJobResult with final status
        """
        logger.info(f"Starting fine-tuning workflow for {training_file_path}")
        
        # Upload training file
        training_upload = self.upload_training_file(training_file_path)
        
        # Upload validation file if provided
        validation_file_id = None
        if validation_file_path:
            validation_upload = self.upload_training_file(validation_file_path)
            validation_file_id = validation_upload.file_id
        
        # Create fine-tuning job
        job = self.create_fine_tuning_job(
            training_file_id=training_upload.file_id,
            model=model,
            validation_file_id=validation_file_id,
            hyperparameters=hyperparameters,
            suffix=suffix
        )
        
        # Wait for completion if requested
        if wait_for_completion:
            job = self.wait_for_job_completion(job.job_id, poll_interval)
        
        return job


# Factory function
def create_fine_tuner(api_key: Optional[str] = None) -> OpenAIFineTuner:
    """Create a configured fine-tuner"""
    return OpenAIFineTuner(api_key=api_key) 