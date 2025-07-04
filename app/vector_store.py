"""
Vector Store Service for Influencer Search

This module provides vector storage and similarity search capabilities using FAISS.
It handles embedding storage, metadata management, and efficient similarity search.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd

from app.schemas import InfluencerData, SearchFilters
from app.settings import config

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Internal result class for vector store searches."""
    influencer_id: str
    influencer_data: InfluencerData
    score: float
    search_type: str
    metadata: Optional[Dict[str, Any]] = None


class VectorStore:
    """
    Vector store for influencer embeddings using FAISS.
    
    Supports:
    - Text and image embedding storage
    - Efficient similarity search
    - Metadata filtering
    - Incremental updates
    - Persistence to disk
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize vector store with optional persistent storage."""
        self.index_path = index_path or os.path.join(config.storage_dir, "vector_index")
        self.embedding_dim = 512  # CLIP embedding dimension
        
        # FAISS index for similarity search
        self.text_index: Optional[faiss.Index] = None
        self.image_index: Optional[faiss.Index] = None
        
        # Metadata storage
        self.influencer_data: List[InfluencerData] = []
        self.text_embeddings: List[np.ndarray] = []
        self.image_embeddings: List[np.ndarray] = []
        
        # ID mapping
        self.influencer_id_to_index: Dict[str, int] = {}
        self.index_to_influencer_id: Dict[int, str] = {}
        
        self._ensure_storage_dir()
        self._initialize_indices()
        
    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
    def _initialize_indices(self) -> None:
        """Initialize FAISS indices."""
        # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.text_index = faiss.IndexFlatIP(self.embedding_dim)
        self.image_index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Initialized FAISS indices with dimension {self.embedding_dim}")
        
    def add_influencer(
        self,
        influencer: InfluencerData,
        text_embedding: Optional[np.ndarray] = None,
        image_embedding: Optional[np.ndarray] = None
    ) -> None:
        """Add influencer with embeddings to the vector store."""
        if influencer.influencer_id in self.influencer_id_to_index:
            logger.warning(f"Influencer {influencer.influencer_id} already exists, skipping")
            return
            
        current_index = len(self.influencer_data)
        
        # Store influencer data
        self.influencer_data.append(influencer)
        self.influencer_id_to_index[influencer.influencer_id] = current_index
        self.index_to_influencer_id[current_index] = influencer.influencer_id
        
        # Add text embedding
        if text_embedding is not None:
            # Normalize for cosine similarity
            normalized_embedding = text_embedding / np.linalg.norm(text_embedding)
            self.text_embeddings.append(normalized_embedding)
            self.text_index.add(normalized_embedding.reshape(1, -1))
        else:
            self.text_embeddings.append(None)
            
        # Add image embedding
        if image_embedding is not None:
            # Normalize for cosine similarity
            normalized_embedding = image_embedding / np.linalg.norm(image_embedding)
            self.image_embeddings.append(normalized_embedding)
            self.image_index.add(normalized_embedding.reshape(1, -1))
        else:
            self.image_embeddings.append(None)
            
        logger.debug(f"Added influencer {influencer.influencer_id} to vector store")
        
    def add_influencers_batch(
        self,
        influencers: List[InfluencerData],
        text_embeddings: List[Optional[np.ndarray]] = None,
        image_embeddings: List[Optional[np.ndarray]] = None
    ) -> None:
        """Add multiple influencers efficiently."""
        if text_embeddings is None:
            text_embeddings = [None] * len(influencers)
        if image_embeddings is None:
            image_embeddings = [None] * len(influencers)
            
        for influencer, text_emb, image_emb in zip(influencers, text_embeddings, image_embeddings):
            self.add_influencer(influencer, text_emb, image_emb)
            
        logger.info(f"Added {len(influencers)} influencers to vector store")
        
    def search_text(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """Search for similar influencers based on text embedding."""
        return self._search(
            query_embedding, self.text_index, self.text_embeddings,
            k=k, filters=filters, threshold=threshold, search_type="text"
        )
        
    def search_image(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """Search for similar influencers based on image embedding."""
        return self._search(
            query_embedding, self.image_index, self.image_embeddings,
            k=k, filters=filters, threshold=threshold, search_type="image"
        )
        
    def search_multimodal(
        self,
        text_embedding: Optional[np.ndarray] = None,
        image_embedding: Optional[np.ndarray] = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """Search using both text and image embeddings with weighted combination."""
        if text_embedding is None and image_embedding is None:
            raise ValueError("At least one embedding must be provided")
            
        all_scores = {}
        
        # Text search
        if text_embedding is not None:
            text_results = self.search_text(text_embedding, k=min(k*2, 50), filters=filters)
            for result in text_results:
                all_scores[result.influencer_id] = {
                    'text_score': result.score,
                    'image_score': 0.0,
                    'data': result
                }
                
        # Image search
        if image_embedding is not None:
            image_results = self.search_image(image_embedding, k=min(k*2, 50), filters=filters)
            for result in image_results:
                if result.influencer_id in all_scores:
                    all_scores[result.influencer_id]['image_score'] = result.score
                else:
                    all_scores[result.influencer_id] = {
                        'text_score': 0.0,
                        'image_score': result.score,
                        'data': result
                    }
                    
        # Calculate combined scores
        combined_results = []
        for influencer_id, scores in all_scores.items():
            combined_score = (scores['text_score'] * text_weight + 
                            scores['image_score'] * image_weight)
            
            if combined_score >= threshold:
                result = scores['data']
                result.score = combined_score
                result.metadata = result.metadata or {}
                result.metadata.update({
                    'text_score': scores['text_score'],
                    'image_score': scores['image_score'],
                    'text_weight': text_weight,
                    'image_weight': image_weight
                })
                combined_results.append(result)
                
        # Sort by combined score and return top k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:k]
        
    def _search(
        self,
        query_embedding: np.ndarray,
        index: faiss.Index,
        embeddings: List[Optional[np.ndarray]],
        k: int,
        filters: Optional[SearchFilters],
        threshold: float,
        search_type: str
    ) -> List[VectorSearchResult]:
        """Internal search method."""
        if index.ntotal == 0:
            logger.warning(f"No {search_type} embeddings in index")
            return []
            
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS index
        search_k = min(k * 3, index.ntotal)  # Search more to allow for filtering
        scores, indices = index.search(query_normalized.reshape(1, -1), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:  # FAISS returns -1 for invalid indices
                continue
                
            influencer_id = self.index_to_influencer_id.get(idx)
            if influencer_id is None:
                continue
                
            influencer = self.influencer_data[idx]
            
            # Apply filters
            if filters and not self._apply_filters(influencer, filters):
                continue
                
            result = VectorSearchResult(
                influencer_id=influencer_id,
                influencer_data=influencer,
                score=float(score),
                search_type=search_type,
                metadata={
                    'embedding_available': embeddings[idx] is not None,
                    'original_rank': len(results)
                }
            )
            results.append(result)
            
            if len(results) >= k:
                break
                
        logger.debug(f"Found {len(results)} results for {search_type} search")
        return results
        
    def _apply_filters(self, influencer: InfluencerData, filters: SearchFilters) -> bool:
        """Apply search filters to influencer data."""
        if filters.category and influencer.category != filters.category:
            return False
            
        if filters.min_followers and influencer.follower_count < filters.min_followers:
            return False
            
        if filters.max_followers and influencer.follower_count > filters.max_followers:
            return False
            
        return True
        
    def get_influencer(self, influencer_id: str) -> Optional[InfluencerData]:
        """Get influencer by ID."""
        idx = self.influencer_id_to_index.get(influencer_id)
        if idx is not None:
            return self.influencer_data[idx]
        return None
        
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        text_count = sum(1 for emb in self.text_embeddings if emb is not None)
        image_count = sum(1 for emb in self.image_embeddings if emb is not None)
        
        return {
            'total_influencers': len(self.influencer_data),
            'text_embeddings': text_count,
            'image_embeddings': image_count,
            'text_index_size': self.text_index.ntotal if self.text_index else 0,
            'image_index_size': self.image_index.ntotal if self.image_index else 0,
            'embedding_dimension': self.embedding_dim,
            'storage_path': self.index_path
        }
        
    def save(self, path: Optional[str] = None) -> None:
        """Save vector store to disk."""
        save_path = path or self.index_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indices
        if self.text_index and self.text_index.ntotal > 0:
            faiss.write_index(self.text_index, f"{save_path}_text.faiss")
            
        if self.image_index and self.image_index.ntotal > 0:
            faiss.write_index(self.image_index, f"{save_path}_image.faiss")
            
        # Save metadata and mappings
        metadata = {
            'influencer_data': [inf.model_dump() for inf in self.influencer_data],
            'influencer_id_to_index': self.influencer_id_to_index,
            'index_to_influencer_id': self.index_to_influencer_id,
            'embedding_dim': self.embedding_dim
        }
        
        with open(f"{save_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        # Save embeddings
        with open(f"{save_path}_embeddings.pkl", 'wb') as f:
            pickle.dump({
                'text_embeddings': self.text_embeddings,
                'image_embeddings': self.image_embeddings
            }, f)
            
        logger.info(f"Saved vector store to {save_path}")
        
    def load(self, path: Optional[str] = None) -> bool:
        """Load vector store from disk."""
        load_path = path or self.index_path
        
        try:
            # Load metadata
            with open(f"{load_path}_metadata.json", 'r') as f:
                metadata = json.load(f)
                
            self.embedding_dim = metadata['embedding_dim']
            self.influencer_id_to_index = metadata['influencer_id_to_index']
            # Convert string keys back to int for reverse mapping
            self.index_to_influencer_id = {int(k): v for k, v in metadata['index_to_influencer_id'].items()}
            
            # Reconstruct influencer data
            self.influencer_data = [InfluencerData(**data) for data in metadata['influencer_data']]
            
            # Load embeddings
            with open(f"{load_path}_embeddings.pkl", 'rb') as f:
                embeddings = pickle.load(f)
                self.text_embeddings = embeddings['text_embeddings']
                self.image_embeddings = embeddings['image_embeddings']
                
            # Reinitialize indices
            self._initialize_indices()
            
            # Load FAISS indices if they exist
            text_path = f"{load_path}_text.faiss"
            if os.path.exists(text_path):
                self.text_index = faiss.read_index(text_path)
                
            image_path = f"{load_path}_image.faiss"
            if os.path.exists(image_path):
                self.image_index = faiss.read_index(image_path)
                
            logger.info(f"Loaded vector store from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store from {load_path}: {e}")
            return False
            
    def clear(self) -> None:
        """Clear all data from vector store."""
        self.influencer_data.clear()
        self.text_embeddings.clear()
        self.image_embeddings.clear()
        self.influencer_id_to_index.clear()
        self.index_to_influencer_id.clear()
        self._initialize_indices()
        logger.info("Cleared vector store")


# Global vector store instance
vector_store = VectorStore()


if __name__ == "__main__":
    # Test vector store
    logging.basicConfig(level=logging.INFO)
    
    print("Testing VectorStore...")
    
    # Create test data
    test_influencer = InfluencerData(
        influencer_id="test_001",
        name="Test Influencer",
        bio="Fitness enthusiast and lifestyle blogger",
        category="fitness",
        follower_count=50000,
        profile_photo_url="https://example.com/profile.jpg",
        content_thumbnail_url="https://example.com/content.jpg"
    )
    
    # Test embedding (random for demo)
    test_embedding = np.random.rand(512).astype(np.float32)
    
    # Add to vector store
    vector_store.add_influencer(test_influencer, text_embedding=test_embedding)
    
    # Test search
    query_embedding = np.random.rand(512).astype(np.float32)
    results = vector_store.search_text(query_embedding, k=5)
    
    print(f"Found {len(results)} results")
    for result in results:
        print(f"  - {result.influencer_data.name}: {result.score:.3f}")
        
    # Test stats
    stats = vector_store.get_stats()
    print(f"Vector store stats: {stats}")
    
    print("VectorStore test completed!") 