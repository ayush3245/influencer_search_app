"""
Build LlamaIndex-compatible storage from influencer CSV data.
"""

import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser

from app.data_loader import InfluencerDataLoader
from app.settings import init_settings

logger = logging.getLogger(__name__)


def create_documents_from_influencers() -> List[Document]:
    """Create LlamaIndex Documents from influencer data."""
    data_loader = InfluencerDataLoader()
    
    # Load influencer data from CSV
    influencers = data_loader.load_from_csv("data/sample_influencers.csv")
    
    documents = []
    for influencer in influencers:
        # Create a very concise text representation for each influencer (under 77 tokens)
        # Use just the category to avoid tokenization issues
        text = influencer.category
        
        # Create document with metadata
        doc = Document(
            text=text,
            metadata={
                "influencer_id": influencer.influencer_id,
                "name": influencer.name,
                "category": influencer.category,
                "follower_count": influencer.follower_count,
                "bio": influencer.bio,  # Keep full bio in metadata
                "profile_photo_url": influencer.profile_photo_url,
                "content_thumbnail_url": influencer.content_thumbnail_url,
            }
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} documents from influencer data")
    return documents


def build_llamaindex():
    """Build LlamaIndex-compatible storage from influencer data."""
    load_dotenv()
    init_settings()
    
    logger.info("Building LlamaIndex from influencer data...")
    
    # Clear existing storage
    storage_dir = "storage"
    if os.path.exists(storage_dir):
        import shutil
        shutil.rmtree(storage_dir)
    
    # Create documents
    documents = create_documents_from_influencers()
    
    if not documents:
        logger.error("No documents created!")
        return False
    
    # Build index
    logger.info("Creating vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    # Save to storage
    logger.info(f"Saving index to {storage_dir}...")
    index.storage_context.persist(storage_dir)
    
    logger.info("LlamaIndex storage created successfully!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = build_llamaindex()
    if success:
        print("✅ LlamaIndex storage built successfully!")
    else:
        print("❌ Failed to build LlamaIndex storage") 