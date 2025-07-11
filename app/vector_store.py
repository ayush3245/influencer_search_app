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
    - Text embedding storage
    - Separate profile and content image embedding storage
    - Efficient similarity search across all embedding types
    - Metadata filtering
    - Incremental updates
    - Persistence to disk
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize vector store with optional persistent storage."""
        self.index_path = index_path or os.path.join(config.storage_dir, "vector_index")
        self.embedding_dim = 512  # CLIP embedding dimension
        
        # FAISS indices for similarity search
        self.text_index: Optional[faiss.Index] = None
        self.profile_index: Optional[faiss.Index] = None
        self.content_index: Optional[faiss.Index] = None
        
        # Metadata storage
        self.influencer_data: List[InfluencerData] = []
        self.text_embeddings: List[np.ndarray] = []
        self.profile_embeddings: List[np.ndarray] = []
        self.content_embeddings: List[np.ndarray] = []
        
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
        self.profile_index = faiss.IndexFlatIP(self.embedding_dim)
        self.content_index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Initialized FAISS indices with dimension {self.embedding_dim}")
        
    def add_influencer(
        self,
        influencer: InfluencerData,
        text_embedding: Optional[np.ndarray] = None,
        profile_embedding: Optional[np.ndarray] = None,
        content_embedding: Optional[np.ndarray] = None
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
            
        # Add profile image embedding
        if profile_embedding is not None:
            # Normalize for cosine similarity
            normalized_embedding = profile_embedding / np.linalg.norm(profile_embedding)
            self.profile_embeddings.append(normalized_embedding)
            self.profile_index.add(normalized_embedding.reshape(1, -1))
        else:
            self.profile_embeddings.append(None)
            
        # Add content image embedding
        if content_embedding is not None:
            # Normalize for cosine similarity
            normalized_embedding = content_embedding / np.linalg.norm(content_embedding)
            self.content_embeddings.append(normalized_embedding)
            self.content_index.add(normalized_embedding.reshape(1, -1))
        else:
            self.content_embeddings.append(None)
            
        logger.debug(f"Added influencer {influencer.influencer_id} to vector store")
        
    def add_influencers_batch(
        self,
        influencers: List[InfluencerData],
        text_embeddings: List[Optional[np.ndarray]] = None,
        profile_embeddings: List[Optional[np.ndarray]] = None,
        content_embeddings: List[Optional[np.ndarray]] = None
    ) -> None:
        """Add multiple influencers efficiently."""
        if text_embeddings is None:
            text_embeddings = [None] * len(influencers)
        if profile_embeddings is None:
            profile_embeddings = [None] * len(influencers)
        if content_embeddings is None:
            content_embeddings = [None] * len(influencers)
            
        for influencer, text_emb, profile_emb, content_emb in zip(
            influencers, text_embeddings, profile_embeddings, content_embeddings
        ):
            self.add_influencer(influencer, text_emb, profile_emb, content_emb)
            
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
        
    def search_profile_image(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """Search for similar influencers based on profile image embedding."""
        return self._search(
            query_embedding, self.profile_index, self.profile_embeddings,
            k=k, filters=filters, threshold=threshold, search_type="profile_image"
        )
        
    def search_content_image(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """Search for similar influencers based on content image embedding."""
        return self._search(
            query_embedding, self.content_index, self.content_embeddings,
            k=k, filters=filters, threshold=threshold, search_type="content_image"
        )
        
    def search_all_images(
        self,
        profile_embedding: Optional[np.ndarray] = None,
        content_embedding: Optional[np.ndarray] = None,
        profile_weight: float = 0.5,
        content_weight: float = 0.5,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """Search using both profile and content image embeddings with weighted combination."""
        if profile_embedding is None and content_embedding is None:
            raise ValueError("At least one image embedding must be provided")
            
        all_scores = {}
        
        # Profile image search
        if profile_embedding is not None:
            profile_results = self.search_profile_image(profile_embedding, k=min(k*2, 50), filters=filters)
            for result in profile_results:
                all_scores[result.influencer_id] = {
                    'profile_score': result.score,
                    'content_score': 0.0,
                    'data': result
                }
                
        # Content image search
        if content_embedding is not None:
            content_results = self.search_content_image(content_embedding, k=min(k*2, 50), filters=filters)
            for result in content_results:
                if result.influencer_id in all_scores:
                    all_scores[result.influencer_id]['content_score'] = result.score
                else:
                    all_scores[result.influencer_id] = {
                        'profile_score': 0.0,
                        'content_score': result.score,
                        'data': result
                    }
                    
        # Calculate combined scores
        combined_results = []
        for influencer_id, scores in all_scores.items():
            combined_score = (scores['profile_score'] * profile_weight + 
                            scores['content_score'] * content_weight)
            
            if combined_score >= threshold:
                result = scores['data']
                result.score = combined_score
                result.metadata = result.metadata or {}
                result.metadata.update({
                    'profile_score': scores['profile_score'],
                    'content_score': scores['content_score'],
                    'profile_weight': profile_weight,
                    'content_weight': content_weight
                })
                combined_results.append(result)
                
        # Sort by combined score and return top k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:k]
        
    def search_multimodal(
        self,
        text_embedding: Optional[np.ndarray] = None,
        profile_embedding: Optional[np.ndarray] = None,
        content_embedding: Optional[np.ndarray] = None,
        text_weight: float = 0.4,
        profile_weight: float = 0.3,
        content_weight: float = 0.3,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """Search using text, profile, and content embeddings with weighted combination."""
        if text_embedding is None and profile_embedding is None and content_embedding is None:
            raise ValueError("At least one embedding must be provided")
            
        all_scores = {}
        
        # Text search
        if text_embedding is not None:
            text_results = self.search_text(text_embedding, k=min(k*2, 50), filters=filters)
            for result in text_results:
                all_scores[result.influencer_id] = {
                    'text_score': result.score,
                    'profile_score': 0.0,
                    'content_score': 0.0,
                    'data': result
                }
                
        # Profile image search
        if profile_embedding is not None:
            profile_results = self.search_profile_image(profile_embedding, k=min(k*2, 50), filters=filters)
            for result in profile_results:
                if result.influencer_id in all_scores:
                    all_scores[result.influencer_id]['profile_score'] = result.score
                else:
                    all_scores[result.influencer_id] = {
                        'text_score': 0.0,
                        'profile_score': result.score,
                        'content_score': 0.0,
                        'data': result
                    }
                    
        # Content image search
        if content_embedding is not None:
            content_results = self.search_content_image(content_embedding, k=min(k*2, 50), filters=filters)
            for result in content_results:
                if result.influencer_id in all_scores:
                    all_scores[result.influencer_id]['content_score'] = result.score
                else:
                    all_scores[result.influencer_id] = {
                        'text_score': 0.0,
                        'profile_score': 0.0,
                        'content_score': result.score,
                        'data': result
                    }
                    
        # Calculate combined scores
        combined_results = []
        for influencer_id, scores in all_scores.items():
            combined_score = (scores['text_score'] * text_weight + 
                            scores['profile_score'] * profile_weight +
                            scores['content_score'] * content_weight)
            
            if combined_score >= threshold:
                result = scores['data']
                result.score = combined_score
                result.metadata = result.metadata or {}
                result.metadata.update({
                    'text_score': scores['text_score'],
                    'profile_score': scores['profile_score'],
                    'content_score': scores['content_score'],
                    'text_weight': text_weight,
                    'profile_weight': profile_weight,
                    'content_weight': content_weight
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
        profile_count = sum(1 for emb in self.profile_embeddings if emb is not None)
        content_count = sum(1 for emb in self.content_embeddings if emb is not None)
        
        return {
            'total_influencers': len(self.influencer_data),
            'text_embeddings': text_count,
            'profile_embeddings': profile_count,
            'content_embeddings': content_count,
            'text_index_size': self.text_index.ntotal if self.text_index else 0,
            'profile_index_size': self.profile_index.ntotal if self.profile_index else 0,
            'content_index_size': self.content_index.ntotal if self.content_index else 0,
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
            
        if self.profile_index and self.profile_index.ntotal > 0:
            faiss.write_index(self.profile_index, f"{save_path}_profile.faiss")
            
        if self.content_index and self.content_index.ntotal > 0:
            faiss.write_index(self.content_index, f"{save_path}_content.faiss")
            
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
                'profile_embeddings': self.profile_embeddings,
                'content_embeddings': self.content_embeddings
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
                self.profile_embeddings = embeddings['profile_embeddings']
                self.content_embeddings = embeddings['content_embeddings']
                
            # Reinitialize indices
            self._initialize_indices()
            
            # Load FAISS indices if they exist
            text_path = f"{load_path}_text.faiss"
            if os.path.exists(text_path):
                self.text_index = faiss.read_index(text_path)
                
            profile_path = f"{load_path}_profile.faiss"
            if os.path.exists(profile_path):
                self.profile_index = faiss.read_index(profile_path)
                
            content_path = f"{load_path}_content.faiss"
            if os.path.exists(content_path):
                self.content_index = faiss.read_index(content_path)
                
            logger.info(f"Loaded vector store from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store from {load_path}: {e}")
            return False
            
    def clear(self) -> None:
        """Clear all data from vector store."""
        self.influencer_data.clear()
        self.text_embeddings.clear()
        self.profile_embeddings.clear()
        self.content_embeddings.clear()
        self.influencer_id_to_index.clear()
        self.index_to_influencer_id.clear()
        self._initialize_indices()
        logger.info("Cleared vector store")


# Global vector store instance
vector_store = VectorStore()
# Automatically load existing data if available
vector_store.load()


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
    
    # Test embeddings (random for demo)
    text_embedding = np.random.rand(512).astype(np.float32)
    profile_embedding = np.random.rand(512).astype(np.float32)
    content_embedding = np.random.rand(512).astype(np.float32)
    
    # Add to vector store
    vector_store.add_influencer(
        test_influencer, 
        text_embedding=text_embedding,
        profile_embedding=profile_embedding,
        content_embedding=content_embedding
    )
    
    # Test text search
    query_embedding = np.random.rand(512).astype(np.float32)
    text_results = vector_store.search_text(query_embedding, k=5)
    
    print(f"Text search found {len(text_results)} results")
    for result in text_results:
        print(f"  - {result.influencer_data.name}: {result.score:.3f}")
    
    # Test profile image search
    profile_results = vector_store.search_profile_image(query_embedding, k=5)
    print(f"Profile image search found {len(profile_results)} results")
    
    # Test content image search
    content_results = vector_store.search_content_image(query_embedding, k=5)
    print(f"Content image search found {len(content_results)} results")
    
    # Test combined image search
    combined_results = vector_store.search_all_images(
        profile_embedding=query_embedding, 
        content_embedding=query_embedding, 
        k=5
    )
    print(f"Combined image search found {len(combined_results)} results")
        
    # Test stats
    stats = vector_store.get_stats()
    print(f"Vector store stats: {stats}")
    
    print("VectorStore test completed!") 