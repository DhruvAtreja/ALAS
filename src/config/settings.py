"""
Configuration settings for the ALAS (Autonomous Learning Agent System)
"""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration"""
    api_key: str = ""
    deep_research_model: str = "o3"
    deep_research_mini_model: str = "o4-mini"
    fine_tuning_model: str = "gpt-4.1-2025-04-14"
    timeout: int = 3600  # 1 hour for deep research
    max_concurrent_requests: int = 5
    
    class Config:
        env_prefix = "OPENAI_"


class LangSmithConfig(BaseSettings):
    """LangSmith monitoring configuration"""
    api_key: str = ""
    project: str = "self-learning-agent"
    tracing_v2: bool = True
    
    class Config:
        env_prefix = "LANGCHAIN_"


class ExaConfig(BaseSettings):
    """Exa Search API configuration"""
    api_key: str = ""
    
    class Config:
        env_prefix = "EXA_"


class RedisConfig(BaseSettings):
    """Redis configuration for state persistence"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}"
        return f"redis://{self.host}:{self.port}"
    
    class Config:
        env_prefix = "REDIS_"


class CostControlConfig(BaseSettings):
    """Cost control and budget configuration"""
    max_budget_per_domain: float = 2000.0
    cost_per_deep_research: float = 50.0
    cost_per_fine_tuning: float = 100.0
    
    class Config:
        env_prefix = ""


class LearningConfig(BaseSettings):
    """Learning algorithm configuration"""
    initial_topics_breadth: int = 10
    topics_per_iteration: int = 5
    evaluation_threshold: float = 0.7
    mastery_threshold: float = 0.85
    max_iterations: int = 20
    min_training_examples_per_topic: int = 20
    
    class Config:
        env_prefix = "LEARNING_"


class Settings(BaseSettings):
    """Main settings class combining all configurations"""
    
    # Sub-configurations  
    openai: OpenAIConfig = Field(default_factory=lambda: OpenAIConfig())
    langsmith: LangSmithConfig = Field(default_factory=lambda: LangSmithConfig())
    exa: ExaConfig = Field(default_factory=lambda: ExaConfig())
    redis: RedisConfig = Field(default_factory=lambda: RedisConfig())
    cost_control: CostControlConfig = Field(default_factory=lambda: CostControlConfig())
    learning: LearningConfig = Field(default_factory=lambda: LearningConfig())
    
    # General settings
    environment: str = "development"
    webhook_base_url: Optional[str] = None
    log_level: str = "INFO"
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    class Config:
        env_prefix = ""
        case_sensitive = False


# Singleton instance
settings = Settings()

# Initialize LangSmith tracing
if settings.langsmith.api_key and settings.langsmith.tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith.api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project


# Helper function to get nested config
def get_config(path: str) -> Any:
    """
    Get configuration value by dot-separated path
    Example: get_config("openai.api_key")
    """
    parts = path.split(".")
    value = settings
    
    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        else:
            raise KeyError(f"Configuration key '{path}' not found")
    
    return value 