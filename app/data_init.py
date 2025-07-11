"""
Data Initialization Module

This module handles loading and initialization of influencer data on application startup.
It checks if data is already loaded and processes the Instagram CSV if needed.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from app.vector_store import VectorStore, vector_store  # Import global instance
from app.instagram_data_processor import InstagramDataProcessor, process_instagram_data_file
from app.settings import config

logger = logging.getLogger(__name__)


class DataInitializer:
    """Handles data initialization for the application."""
    
    def __init__(self):
        """Initialize the data initializer."""
        self.vector_store = vector_store  # Use global instance
        self.data_path = Path("data/instagram_influencers_final_20250708.csv")
        self.vector_store_path = Path(config.storage_dir) / "vector_index"
        
    def is_data_loaded(self) -> bool:
        """Check if data is already loaded in the vector store."""
        try:
            stats = self.vector_store.get_stats()
            return stats.get('total_influencers', 0) > 0
        except Exception as e:
            logger.warning(f"Error checking data status: {e}")
            return False
    
    def has_saved_index(self) -> bool:
        """Check if there's a saved vector index on disk."""
        try:
            # Check for various index files
            index_files = [
                self.vector_store_path / "metadata.json",
                self.vector_store_path / "text_index.faiss",
                self.vector_store_path / "profile_index.faiss",
                self.vector_store_path / "content_index.faiss"
            ]
            return any(f.exists() for f in index_files)
        except Exception:
            return False
    
    def load_saved_index(self) -> bool:
        """Load a previously saved vector index."""
        try:
            logger.info("Loading saved vector index...")
            success = self.vector_store.load(str(self.vector_store_path))
            if success:
                stats = self.vector_store.get_stats()
                logger.info(f"Successfully loaded {stats.get('total_influencers', 0)} influencers from saved index")
                return True
            else:
                logger.warning("Failed to load saved index")
                return False
        except Exception as e:
            logger.error(f"Error loading saved index: {e}")
            return False
    
    def process_instagram_data(self) -> bool:
        """Process the Instagram CSV data and create embeddings."""
        try:
            if not self.data_path.exists():
                logger.error(f"Instagram data file not found: {self.data_path}")
                return False
            
            logger.info("Processing Instagram influencer data...")
            stats = process_instagram_data_file(str(self.data_path))
            
            if stats.get('status') == 'completed':
                logger.info(f"Successfully processed Instagram data: {stats}")
                
                # Reload the global vector store to pick up the newly processed data
                logger.info("Reloading global vector store with processed data...")
                if self.vector_store.load(str(self.vector_store_path)):
                    reload_stats = self.vector_store.get_stats()
                    logger.info(f"Global vector store reloaded: {reload_stats.get('total_influencers', 0)} influencers")
                else:
                    logger.warning("Failed to reload global vector store")
                
                return True
            else:
                logger.error(f"Failed to process Instagram data: {stats}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing Instagram data: {e}")
            return False
    
    def initialize_data(self) -> Dict[str, Any]:
        """
        Initialize data for the application.
        
        Returns:
            Dictionary with initialization status and statistics
        """
        logger.info("Starting data initialization...")
        
        # Check if data is already loaded in memory
        if self.is_data_loaded():
            stats = self.vector_store.get_stats()
            logger.info(f"Data already loaded: {stats.get('total_influencers', 0)} influencers")
            return {
                'status': 'already_loaded',
                'source': 'memory',
                'stats': stats
            }
        
        # Try to load from saved index
        if self.has_saved_index():
            if self.load_saved_index():
                # Make sure global vector store is also updated
                self.vector_store.load(str(self.vector_store_path))
                stats = self.vector_store.get_stats()
                return {
                    'status': 'loaded_from_disk',
                    'source': 'saved_index',
                    'stats': stats
                }
        
        # Process Instagram data from scratch
        logger.info("No saved index found, processing Instagram data from scratch...")
        if self.process_instagram_data():
            stats = self.vector_store.get_stats()
            return {
                'status': 'processed_from_csv',
                'source': 'instagram_csv',
                'stats': stats
            }
        
        # Fallback - no data available
        logger.error("Failed to initialize any data")
        return {
            'status': 'failed',
            'source': 'none',
            'stats': {'total_influencers': 0}
        }
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get current data status information."""
        return {
            'data_loaded': self.is_data_loaded(),
            'has_saved_index': self.has_saved_index(),
            'csv_file_exists': self.data_path.exists(),
            'vector_store_stats': self.vector_store.get_stats() if self.is_data_loaded() else {},
            'csv_file_path': str(self.data_path),
            'vector_store_path': str(self.vector_store_path)
        }


# Global data initializer instance
_data_initializer = None


def get_data_initializer() -> DataInitializer:
    """Get the global data initializer instance."""
    global _data_initializer
    if _data_initializer is None:
        _data_initializer = DataInitializer()
    return _data_initializer


def ensure_data_loaded() -> Dict[str, Any]:
    """
    Ensure that influencer data is loaded in the application.
    
    This function is called during application startup to initialize data.
    
    Returns:
        Dictionary with initialization results
    """
    initializer = get_data_initializer()
    return initializer.initialize_data()


def get_data_status() -> Dict[str, Any]:
    """Get current status of data loading."""
    initializer = get_data_initializer()
    return initializer.get_data_status()


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize data
    result = ensure_data_loaded()
    print(f"Data initialization result: {result}")
    
    # Print status
    status = get_data_status()
    print(f"Data status: {status}") 