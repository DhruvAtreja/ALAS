"""
Logging configuration for ALAS
"""

import sys
from pathlib import Path
from loguru import logger

try:
    from ..config.settings import settings
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import settings


def setup_logger():
    """Configure loguru logger with appropriate settings"""
    
    # Remove default logger
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
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
    if settings.environment == "production":
        # JSON logs for production
        logger.add(
            log_dir / "alas_json_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="INFO",
            format="{message}",
            serialize=True
        )
    
    return logger


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


def log_error(error: Exception, context: dict = None):
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