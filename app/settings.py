"""
Settings and configuration for the Influencer Discovery Tool.

This module provides configuration for CLIP embeddings, vector storage, 
and other application components without LLM dependencies.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
try:
    load_dotenv(encoding='utf-8')
except UnicodeDecodeError:
    # Fallback for encoding issues
    load_dotenv(encoding='utf-8-sig')  # This handles BOM
except Exception:
    # If .env file has issues, continue without it (env vars might be set elsewhere)
    pass

logger = logging.getLogger(__name__)


class AppConfig:
    """Application configuration class."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # CLIP Configuration
        self.clip_model_name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        self.clip_device = os.getenv("CLIP_DEVICE", "cpu")  # or "cuda" if GPU available
        
        # Data Configuration
        self.data_dir = Path(os.getenv("DATA_DIR", "data"))
        self.storage_dir = Path(os.getenv("STORAGE_DIR", "storage"))
        self.vector_store_path = self.storage_dir / "vector_index"
        
        # Application Settings
        self.app_env = os.getenv("APP_ENV", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        
        # Search Configuration
        self.default_similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "50"))
        self.default_search_limit = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
        
        # Performance Configuration
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.cache_embeddings = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "512"))
        
        # Embedding weights for multimodal search
        self.text_weight = float(os.getenv("TEXT_WEIGHT", "0.4"))
        self.profile_weight = float(os.getenv("PROFILE_WEIGHT", "0.3"))
        self.content_weight = float(os.getenv("CONTENT_WEIGHT", "0.3"))
        
        # Validate configuration
        self._validate_config()
        
        # Create directories
        self._create_directories()
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if self.clip_device not in ["cpu", "cuda"]:
            logger.warning(f"Invalid CLIP_DEVICE '{self.clip_device}', defaulting to 'cpu'")
            self.clip_device = "cpu"
        
        if self.default_similarity_threshold < 0 or self.default_similarity_threshold > 1:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        
        # Validate embedding weights
        total_weight = self.text_weight + self.profile_weight + self.content_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            logger.warning(f"Embedding weights don't sum to 1.0 (sum: {total_weight}), normalizing")
            self.text_weight /= total_weight
            self.profile_weight /= total_weight
            self.content_weight /= total_weight
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Storage directory: {self.storage_dir}")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env.lower() == "development"


# Global configuration instance
config = AppConfig()


def init_settings() -> None:
    """
    Initialize application settings without LLM dependencies.
    
    This function configures logging and validates the configuration.
    """
    try:
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Initialized settings with CLIP model: {config.clip_model_name}")
        logger.info(f"Using device: {config.clip_device}")
        logger.info(f"Environment: {config.app_env}")
        logger.info(f"Embedding weights - Text: {config.text_weight}, Profile: {config.profile_weight}, Content: {config.content_weight}")
        
    except Exception as e:
        logger.error(f"Failed to initialize settings: {e}")
        raise


def get_embedding_dimension() -> int:
    """
    Get the embedding dimension for the configured CLIP model.
    
    Returns:
        int: Embedding dimension (typically 512 for ViT-B/32)
    """
    return config.embedding_dimension


def get_vector_store_path() -> Path:
    """
    Get the path to the vector store directory.
    
    Returns:
        Path: Vector store directory path
    """
    return config.vector_store_path


# Initialize settings when module is imported
if __name__ != "__main__":
    init_settings()

