"""
Index Building Script for Influencer Search

This script loads influencer data, generates CLIP embeddings, and builds the vector store
for efficient similarity search.
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import List, Optional
import time
import numpy as np

from app.data_loader import InfluencerDataLoader
from app.embedding_service import CLIPEmbeddingService
from app.vector_store import vector_store
from app.settings import config
from app.schemas import InfluencerData

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builds the search index by combining data loading, embedding generation,
    and vector store population.
    """
    
    def __init__(self):
        """Initialize the index builder."""
        self.data_loader = InfluencerDataLoader()
        self.embedding_service = CLIPEmbeddingService()
        self.vector_store = vector_store
        
    def build_index(
        self,
        data_file: str,
        force_rebuild: bool = False,
        save_index: bool = True
    ) -> bool:
        """
        Build the complete search index.
        
        Args:
            data_file: Path to influencer data file (CSV/Excel)
            force_rebuild: Whether to rebuild even if index exists
            save_index: Whether to save index to disk
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Starting index build for {data_file}")
            start_time = time.time()
            
            # Check if index already exists
            if not force_rebuild and self._index_exists():
                logger.info("Index already exists, loading from disk...")
                if self.vector_store.load():
                    logger.info("Successfully loaded existing index")
                    return True
                else:
                    logger.warning("Failed to load existing index, rebuilding...")
            
            # Step 1: Load influencer data
            logger.info("Step 1: Loading influencer data...")
            
            # Determine file type and load accordingly
            if data_file.lower().endswith('.csv'):
                influencers = self.data_loader.load_from_csv(data_file)
            elif data_file.lower().endswith(('.xlsx', '.xls')):
                influencers = self.data_loader.load_from_excel(data_file)
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
                
            if not influencers:
                logger.error("No influencer data loaded")
                return False
                
            logger.info(f"Loaded {len(influencers)} influencers")
            
            # Step 2: Generate embeddings
            logger.info("Step 2: Generating embeddings...")
            text_embeddings, image_embeddings = self._generate_embeddings(influencers)
            
            # Step 3: Build vector store
            logger.info("Step 3: Building vector store...")
            self._build_vector_store(influencers, text_embeddings, image_embeddings)
            
            # Step 4: Save index if requested
            if save_index:
                logger.info("Step 4: Saving index to disk...")
                self.vector_store.save()
                
            # Log completion stats
            elapsed_time = time.time() - start_time
            stats = self.vector_store.get_stats()
            
            logger.info(f"Index build completed in {elapsed_time:.2f}s")
            logger.info(f"Final stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            return False
    
    def _index_exists(self) -> bool:
        """Check if index files exist on disk."""
        base_path = self.vector_store.index_path
        metadata_path = f"{base_path}_metadata.json"
        return os.path.exists(metadata_path)
    
    def _generate_embeddings(
        self, 
        influencers: List[InfluencerData]
    ) -> tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """Generate text and image embeddings for all influencers."""
        
        text_embeddings = []
        image_embeddings = []
        
        total = len(influencers)
        success_text = 0
        success_image = 0
        
        for i, influencer in enumerate(influencers):
            if i % 5 == 0:  # Progress every 5 items
                logger.info(f"Processing embeddings: {i+1}/{total}")
            
            # Generate text embedding from bio
            try:
                text_emb = self.embedding_service.generate_text_embedding(influencer.bio)
                text_embeddings.append(text_emb)
                if text_emb is not None:
                    success_text += 1
            except Exception as e:
                logger.warning(f"Failed to generate text embedding for {influencer.influencer_id}: {e}")
                text_embeddings.append(None)
            
            # Generate image embedding from profile photo
            try:
                if influencer.profile_photo_url:
                    image_emb = self.embedding_service.generate_image_embedding(str(influencer.profile_photo_url))
                    if image_emb is not None:
                        image_embeddings.append(image_emb)
                        success_image += 1
                    else:
                        image_embeddings.append(None)
                else:
                    image_embeddings.append(None)
            except Exception as e:
                logger.warning(f"Failed to generate image embedding for {influencer.influencer_id}: {e}")
                image_embeddings.append(None)
        
        logger.info(f"Generated embeddings - Text: {success_text}/{total}, Image: {success_image}/{total}")
        return text_embeddings, image_embeddings
    
    def _build_vector_store(
        self,
        influencers: List[InfluencerData],
        text_embeddings: List[Optional[np.ndarray]],
        image_embeddings: List[Optional[np.ndarray]]
    ) -> None:
        """Build the vector store with influencer data and embeddings."""
        
        # Clear existing data
        self.vector_store.clear()
        
        # Add influencers in batch
        self.vector_store.add_influencers_batch(
            influencers=influencers,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings
        )
        
        logger.info("Vector store populated successfully")
    
    def rebuild_index(self, data_file: str) -> bool:
        """Force rebuild the entire index."""
        return self.build_index(data_file, force_rebuild=True, save_index=True)
    
    def update_index(self, new_influencers: List[InfluencerData]) -> bool:
        """Update index with new influencers (incremental update)."""
        try:
            logger.info(f"Updating index with {len(new_influencers)} new influencers")
            
            # Generate embeddings for new influencers
            text_embeddings, image_embeddings = self._generate_embeddings(new_influencers)
            
            # Add to vector store
            for i, influencer in enumerate(new_influencers):
                self.vector_store.add_influencer(
                    influencer=influencer,
                    text_embedding=text_embeddings[i],
                    image_embedding=image_embeddings[i]
                )
            
            # Save updated index
            self.vector_store.save()
            
            logger.info("Index updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update index: {e}", exc_info=True)
            return False
    
    def get_index_stats(self) -> dict:
        """Get current index statistics."""
        return self.vector_store.get_stats()


# CLI interface
def main():
    """Main CLI interface for building the index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build influencer search index")
    parser.add_argument(
        "data_file",
        help="Path to influencer data file (CSV/Excel)",
        default="data/sample_influencers.csv",
        nargs="?"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save index to disk"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Build index
    builder = IndexBuilder()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        logger.error(f"Data file not found: {args.data_file}")
        return 1
    
    success = builder.build_index(
        data_file=args.data_file,
        force_rebuild=args.force,
        save_index=not args.no_save
    )
    
    if success:
        # Print final stats
        stats = builder.get_index_stats()
        print("\n" + "="*50)
        print("INDEX BUILD COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Total influencers: {stats['total_influencers']}")
        print(f"Text embeddings: {stats['text_embeddings']}")
        print(f"Image embeddings: {stats['image_embeddings']}")
        print(f"Storage path: {stats['storage_path']}")
        print("="*50)
        return 0
    else:
        print("\n" + "="*50)
        print("INDEX BUILD FAILED")
        print("="*50)
        return 1


if __name__ == "__main__":
    exit(main()) 