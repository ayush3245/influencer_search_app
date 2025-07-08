"""
Instagram Data Processor for the Influencer Discovery Tool.

This module processes the Instagram influencers CSV data and generates embeddings
for both text and images, then stores them in the vector database.
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from app.data_loader import InfluencerDataLoader
from app.embedding_service import CLIPEmbeddingService, LocalImagePathResolver
from app.vector_store import VectorStore
from app.schemas import InfluencerData
from app.settings import config

logger = logging.getLogger(__name__)


class InstagramDataProcessor:
    """Process Instagram influencer data and generate embeddings."""
    
    def __init__(self):
        """Initialize the processor with required services."""
        self.data_loader = InfluencerDataLoader(validate_images=False)
        self.embedding_service = CLIPEmbeddingService()
        self.vector_store = VectorStore()
        self.image_path_resolver = LocalImagePathResolver()
        
        # Log local image availability
        image_stats = self.image_path_resolver.get_available_images_count()
        logger.info(f"Local images available: {image_stats}")
        
    def load_instagram_data(self, csv_path: str) -> List[InfluencerData]:
        """
        Load Instagram influencer data from CSV.
        
        Args:
            csv_path: Path to the Instagram CSV file
            
        Returns:
            List of validated InfluencerData objects
        """
        logger.info(f"Loading Instagram data from {csv_path}")
        
        try:
            influencers = self.data_loader.load_from_csv(csv_path)
            
            # Log validation report
            validation_report = self.data_loader.get_validation_report()
            logger.info(f"Loaded {validation_report['total_loaded']} influencers successfully")
            
            if validation_report['total_errors'] > 0:
                logger.warning(f"Found {validation_report['total_errors']} validation errors")
                
                # Save errors to file for review
                error_file = Path(csv_path).parent / "loading_errors.json"
                self.data_loader.save_errors_to_file(error_file)
                logger.info(f"Saved validation errors to {error_file}")
            
            return influencers
            
        except Exception as e:
            logger.error(f"Failed to load Instagram data: {e}")
            raise
    
    def replace_urls_with_local_paths(self, influencers: List[InfluencerData]) -> List[InfluencerData]:
        """
        Replace expired Instagram URLs with local image paths.
        
        Args:
            influencers: List of influencer data with URLs
            
        Returns:
            List of influencer data with local paths where available
        """
        logger.info("Replacing expired URLs with local image paths...")
        
        updated_influencers = []
        local_replacements = 0
        
        for i, influencer in enumerate(influencers):
            # Get local paths for this CSV row
            local_paths = self.image_path_resolver.get_local_image_paths(i)
            
            # Create updated influencer data
            updated_data = influencer.model_copy()
            
            # Replace profile photo URL if local file exists
            if local_paths['profile']:
                updated_data.profile_photo_url = local_paths['profile']
                local_replacements += 1
                logger.debug(f"Row {i}: Using local profile image {local_paths['profile']}")
            
            # Replace content thumbnail URL if local file exists
            if local_paths['content']:
                updated_data.content_thumbnail_url = local_paths['content']
                local_replacements += 1
                logger.debug(f"Row {i}: Using local content image {local_paths['content']}")
            
            updated_influencers.append(updated_data)
        
        logger.info(f"Replaced {local_replacements} URLs with local image paths")
        return updated_influencers
    
    def process_embeddings_batch(self, influencers: List[InfluencerData], batch_size: int = 5) -> Dict[str, Any]:
        """
        Process embeddings for a batch of influencers.
        
        Args:
            influencers: List of influencer data
            batch_size: Number of influencers to process at once
            
        Returns:
            Processing statistics
        """
        total_influencers = len(influencers)
        processed_count = 0
        failed_count = 0
        start_time = time.time()
        
        logger.info(f"Starting embedding generation for {total_influencers} influencers")
        
        # Process in batches
        for i in range(0, total_influencers, batch_size):
            batch = influencers[i:i + batch_size]
            batch_start = time.time()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_influencers + batch_size - 1)//batch_size}")
            
            try:
                # Generate embeddings for the batch
                embeddings_data = self.embedding_service.batch_generate_influencer_embeddings(batch)
                
                # Add to vector store
                text_embeddings = []
                image_embeddings = []
                
                # Track embedding success for this batch
                batch_text_success = 0
                batch_image_success = 0
                
                for j, influencer in enumerate(batch):
                    embeddings = embeddings_data[j]
                    
                    # Fix: Use correct key for text embedding (bio)
                    text_emb = embeddings.get('bio')
                    text_embeddings.append(text_emb)
                    if text_emb is not None:
                        batch_text_success += 1
                    
                    # Fix: Handle image embeddings properly - try both profile and content
                    profile_emb = embeddings.get('profile_photo')
                    content_emb = embeddings.get('content_thumbnail')
                    
                    # Use profile image if available, otherwise use content, otherwise None
                    image_emb = profile_emb if profile_emb is not None else content_emb
                    image_embeddings.append(image_emb)
                    if image_emb is not None:
                        batch_image_success += 1
                
                logger.info(f"Batch embeddings: text {batch_text_success}/{len(batch)}, images {batch_image_success}/{len(batch)}")
                
                # Add batch to vector store
                self.vector_store.add_influencers_batch(
                    influencers=batch,
                    text_embeddings=text_embeddings,
                    image_embeddings=image_embeddings
                )
                
                processed_count += len(batch)
                batch_time = time.time() - batch_start
                logger.info(f"Batch completed in {batch_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                failed_count += len(batch)
                continue
        
        total_time = time.time() - start_time
        
        stats = {
            'total_influencers': total_influencers,
            'processed_successfully': processed_count,
            'failed': failed_count,
            'total_time_seconds': total_time,
            'average_time_per_influencer': total_time / total_influencers if total_influencers > 0 else 0
        }
        
        logger.info(f"Embedding processing completed: {stats}")
        return stats
    
    def save_vector_store(self, path: Optional[str] = None) -> None:
        """Save the vector store to disk."""
        try:
            self.vector_store.save(path)
            logger.info(f"Vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return self.vector_store.get_stats()
    
    def process_instagram_csv(self, csv_path: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete processing pipeline for Instagram CSV data.
        
        Args:
            csv_path: Path to Instagram CSV file
            save_path: Optional path to save vector store
            
        Returns:
            Processing statistics
        """
        logger.info("Starting complete Instagram data processing pipeline")
        
        # Step 1: Load data
        influencers = self.load_instagram_data(csv_path)
        
        if not influencers:
            logger.warning("No valid influencers loaded, stopping processing")
            return {'status': 'no_data', 'influencers_loaded': 0}
        
        # Step 1.5: Replace URLs with local paths
        influencers = self.replace_urls_with_local_paths(influencers)
        
        # Step 2: Process embeddings
        embedding_stats = self.process_embeddings_batch(influencers)
        
        # Step 3: Save vector store
        if save_path or embedding_stats['processed_successfully'] > 0:
            self.save_vector_store(save_path)
        
        # Step 4: Get final stats
        vector_stats = self.get_vector_store_stats()
        
        final_stats = {
            'status': 'completed',
            'influencers_loaded': len(influencers),
            'embedding_stats': embedding_stats,
            'vector_store_stats': vector_stats
        }
        
        logger.info(f"Complete processing finished: {final_stats}")
        return final_stats


def process_instagram_data_file(csv_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to process the Instagram data file.
    
    Args:
        csv_path: Path to CSV file (defaults to the standard location)
        
    Returns:
        Processing statistics
    """
    if csv_path is None:
        csv_path = "data/instagram_influencers_final_20250708.csv"
    
    processor = InstagramDataProcessor()
    return processor.process_instagram_csv(csv_path)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process the Instagram data
    stats = process_instagram_data_file()
    print(f"Processing completed with stats: {stats}") 