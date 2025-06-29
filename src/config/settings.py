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
    api_key: str = Field(..., env="OPENAI_API_KEY")
    deep_research_model: str = Field(default="o3", env="OPENAI_DEEP_RESEARCH_MODEL")
    deep_research_mini_model: str = Field(default="o4-mini", env="OPENAI_DEEP_RESEARCH_MINI_MODEL")
    fine_tuning_model: str = Field(default="gpt-4o-2024-08-06", env="OPENAI_FINE_TUNING_MODEL")
    timeout: int = Field(default=3600, env="OPENAI_TIMEOUT")  # 1 hour for deep research
    max_concurrent_requests: int = Field(default=5, env="OPENAI_MAX_CONCURRENT")
    
    class Config:
        env_prefix = "OPENAI_"


class LangSmithConfig(BaseSettings):
    """LangSmith monitoring configuration"""
    api_key: str = Field(..., env="LANGCHAIN_API_KEY")
    project: str = Field(default="self-learning-agent", env="LANGCHAIN_PROJECT")
    tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    
    class Config:
        env_prefix = "LANGCHAIN_"


class ExaConfig(BaseSettings):
    """Exa Search API configuration"""
    api_key: str = Field(..., env="EXA_API_KEY")
    
    class Config:
        env_prefix = "EXA_"


class RedisConfig(BaseSettings):
    """Redis configuration for state persistence"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}"
        return f"redis://{self.host}:{self.port}"
    
    class Config:
        env_prefix = "REDIS_"


class CostControlConfig(BaseSettings):
    """Cost control and budget configuration"""
    max_budget_per_domain: float = Field(default=2000.0, env="MAX_BUDGET_PER_DOMAIN")
    cost_per_deep_research: float = Field(default=50.0, env="COST_PER_DEEP_RESEARCH")
    cost_per_fine_tuning: float = Field(default=100.0, env="COST_PER_FINE_TUNING")
    
    class Config:
        env_prefix = ""


class LearningConfig(BaseSettings):
    """Learning algorithm configuration"""
    initial_topics_breadth: int = Field(default=10, env="INITIAL_TOPICS_BREADTH")
    topics_per_iteration: int = Field(default=5, env="TOPICS_PER_ITERATION")
    evaluation_threshold: float = Field(default=0.7, env="EVALUATION_THRESHOLD")
    mastery_threshold: float = Field(default=0.85, env="MASTERY_THRESHOLD")
    max_iterations: int = Field(default=20, env="MAX_ITERATIONS")
    min_training_examples_per_topic: int = Field(default=20, env="MIN_TRAINING_EXAMPLES")
    
    class Config:
        env_prefix = "LEARNING_"


class Settings(BaseSettings):
    """Main settings class combining all configurations"""
    
    # Sub-configurations
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    exa: ExaConfig = Field(default_factory=ExaConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    cost_control: CostControlConfig = Field(default_factory=CostControlConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    
    # General settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    webhook_base_url: Optional[str] = Field(default=None, env="WEBHOOK_BASE_URL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
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