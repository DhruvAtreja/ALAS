"""
Logging configuration for ALAS
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Optional
from loguru import logger

# Try to import settings, with fallbacks for LangGraph Studio
try:
    from ..config.settings import settings
except ImportError:
    # Fallback for when running directly
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.settings import settings
    except ImportError:
        # Final fallback - create minimal settings object
        class FallbackSettings:
            log_level = "INFO"
            environment = "development"
        
        settings = FallbackSettings()


def is_langgraph_studio() -> bool:
    """Detect if we're running in LangGraph Studio environment"""
    # Check for common LangGraph Studio environment indicators
    indicators = [
        os.getenv("LANGGRAPH_STUDIO") == "true",
        os.getenv("LANGGRAPH_API_WORKER") == "true", 
        os.getenv("LANGGRAPH_API_VERSION") is not None,
        "langgraph" in os.getenv("PYTHONPATH", "").lower(),
        "langgraph-cli" in sys.executable.lower(),
        any("langgraph" in arg.lower() for arg in sys.argv),
        "blockbuster" in str(sys.modules.keys())  # blockbuster is used by LangGraph Studio
    ]
    
    return any(indicators)


def setup_logger():
    """Configure loguru logger with appropriate settings"""
    
    # Remove default logger
    logger.remove()
    
    # Get log level with fallback
    log_level = getattr(settings, 'log_level', 'INFO')
    environment = getattr(settings, 'environment', 'development')
    
    # Console handler with color (always safe, doesn't block)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Only add file handlers if NOT running in LangGraph Studio to avoid blocking operations
    if not is_langgraph_studio():
        # File handler for all logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "alas_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True
        )
        
        # Error log file
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True
        )
        
        # Production settings
        if environment == "production":
            # JSON logs for production
            logger.add(
                log_dir / "alas_json_{time:YYYY-MM-DD}.log",
                rotation="1 day",
                retention="30 days",
                level="INFO",
                format="{message}",
                serialize=True
            )
    else:
        # Running in LangGraph Studio - log to console only to avoid blocking
        logger.info("🔧 Running in LangGraph Studio - file logging disabled to prevent blocking operations")
    
    return logger


# Async logger helper for critical file operations when needed
async def async_log_to_file(message: str, filename: str = "async_log.txt"):
    """Async file logging for critical operations"""
    if not is_langgraph_studio():
        # If not in studio, use regular file operations
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
    else:
        # In studio, use async file operations to avoid blocking
        def _write_log():
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / filename, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        
        # Move to thread to avoid blocking
        await asyncio.to_thread(_write_log)


# Initialize logger on import
setup_logger()


def get_logger(name: str):
    """Get a logger instance with a specific name"""
    return logger.bind(name=name)


# Utility functions for structured logging
def log_api_call(api_name: str, endpoint: str, params: dict, response_time: float):
    """Log API call details"""
    logger.info(
        "API Call",
        extra={
            "api": api_name,
            "endpoint": endpoint,
            "params": params,
            "response_time_ms": response_time * 1000
        }
    )


def log_iteration_start(domain: str, iteration: int):
    """Log the start of a learning iteration"""
    logger.info(
        f"Starting iteration {iteration} for domain: {domain}",
        extra={
            "domain": domain,
            "iteration": iteration,
            "event": "iteration_start"
        }
    )


def log_iteration_complete(domain: str, iteration: int, metrics: dict):
    """Log the completion of a learning iteration"""
    logger.info(
        f"Completed iteration {iteration} for domain: {domain}",
        extra={
            "domain": domain,
            "iteration": iteration,
            "metrics": metrics,
            "event": "iteration_complete"
        }
    )


def log_error(error: Exception, context: Optional[dict] = None):
    """Log an error with context"""
    logger.error(
        f"Error: {str(error)}",
        extra={
            "error_type": type(error).__name__,
            "context": context or {},
            "event": "error"
        }
    )


def log_cost(domain: str, operation: str, cost: float):
    """Log cost tracking information"""
    logger.info(
        f"Cost incurred: ${cost:.2f} for {operation}",
        extra={
            "domain": domain,
            "operation": operation,
            "cost": cost,
            "event": "cost_tracking"
        }
    ) 