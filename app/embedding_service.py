"""
CLIP Embedding Service for the Influencer Discovery Tool.

This module provides functions to generate embeddings for text and images
using the CLIP model, with support for caching and batch processing.
"""

import logging
import asyncio
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import hashlib
import pickle
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel

from app.settings import config
from app.schemas import InfluencerData
import glob

logger = logging.getLogger(__name__)


class LocalImagePathResolver:
    """Resolves CSV row indices to local image file paths."""
    
    def __init__(self, base_image_dir: str = "data/images/downloaded_images"):
        """
        Initialize path resolver.
        
        Args:
            base_image_dir: Base directory containing downloaded images
        """
        self.base_dir = Path(base_image_dir)
        self._validate_directory()
        
    def _validate_directory(self):
        """Validate that the image directory exists."""
        if not self.base_dir.exists():
            logger.warning(f"Local image directory not found: {self.base_dir}")
        else:
            logger.info(f"Local image directory found: {self.base_dir}")
    
    def get_local_image_paths(self, csv_row_index: int) -> Dict[str, Optional[str]]:
        """
        Get local image paths for a given CSV row index.
        
        Args:
            csv_row_index: 0-based CSV row index
            
        Returns:
            Dictionary with 'profile' and 'content' keys mapping to local paths or None
        """
        # Convert 0-based to 1-based with zero padding
        row_num = f"{csv_row_index + 1:02d}"
        
        result = {
            'profile': self._find_image_file(f"influencer_{row_num}_profile"),
            'content': self._find_image_file(f"influencer_{row_num}_thumb")
        }
        
        logger.debug(f"Row {csv_row_index} -> {result}")
        return result
    
    def _find_image_file(self, base_filename: str) -> Optional[str]:
        """
        Find image file with various extensions.
        
        Args:
            base_filename: Base filename without extension
            
        Returns:
            Full path to image file or None if not found
        """
        # Common image extensions
        extensions = ['jpg', 'jpeg', 'png', 'webp']
        
        for ext in extensions:
            file_path = self.base_dir / f"{base_filename}.{ext}"
            if file_path.exists():
                return str(file_path)
        
        # If not found, log warning
        logger.debug(f"Image file not found for pattern: {base_filename}.*")
        return None
    
    def get_available_images_count(self) -> Dict[str, int]:
        """
        Get count of available local images.
        
        Returns:
            Dictionary with counts of profile and content images
        """
        profile_count = len(list(self.base_dir.glob("*_profile.*")))
        content_count = len(list(self.base_dir.glob("*_thumb.*")))
        
        return {
            'profile_images': profile_count,
            'content_images': content_count,
            'total_images': profile_count + content_count
        }


class EmbeddingCache:
    """Simple file-based cache for embeddings."""
    
    def __init__(self, cache_dir: Path = None):
        """Initialize cache with specified directory."""
        self.cache_dir = cache_dir or (config.storage_dir / "embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config.cache_embeddings
        logger.info(f"Embedding cache: {'enabled' if self.enabled else 'disabled'}")
    
    def _get_cache_key(self, content: Union[str, bytes]) -> str:
        """Generate cache key from content."""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if not self.enabled:
            return None
        
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {key}: {e}")
        return None
    
    def set(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        if not self.enabled:
            return
        
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding for {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cleared embedding cache")


class ImageProcessor:
    """Handles image downloading and preprocessing for CLIP."""
    
    def __init__(self, max_image_size: Tuple[int, int] = (224, 224)):
        """Initialize image processor with max size."""
        self.max_image_size = max_image_size
        self.session = requests.Session()
        
        # Set up session with reasonable timeout and retries
        self.session.timeout = 30
    
    def download_image(self, url: str) -> Optional[Image.Image]:
        """
        Download and preprocess image from URL.
        
        Args:
            url: Image URL to download
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Load image
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            return image
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to download image from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to process image from {url}: {e}")
            return None
    
    def load_local_image(self, file_path: str) -> Optional[Image.Image]:
        """
        Load and preprocess local image file.
        
        Args:
            file_path: Path to local image file
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            image = Image.open(file_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load local image {file_path}: {e}")
            return None
    
    def is_local_path(self, path: str) -> bool:
        """
        Check if a path is a local file path.
        
        Args:
            path: Path to check
            
        Returns:
            True if local path, False if URL
        """
        return path.startswith(('/', './', '../')) or '://' not in path


class CLIPEmbeddingService:
    """CLIP embedding service for text and image embeddings."""
    
    def __init__(self):
        """Initialize CLIP model and processor."""
        self.device = config.clip_device
        self.model_name = config.clip_model_name
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # Initialize cache and image processor
        self.cache = EmbeddingCache()
        self.image_processor = ImageProcessor()
        
        logger.info(f"Initialized CLIP model: {self.model_name} on {self.device}")
    
    def generate_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate text embedding using CLIP."""
        try:
            # Check cache first
            cache_key = f"text_{hashlib.md5(text.encode()).hexdigest()}"
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding
            
            # Process text
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                embedding = text_features.cpu().numpy().flatten()
            
            # Cache the result
            self.cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return None
    
    def generate_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate image embedding using CLIP."""
        try:
            # Check cache first
            cache_key = f"image_{hashlib.md5(image_path.encode()).hexdigest()}"
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding
            
            # Load and process image
            if self.image_processor.is_local_path(image_path):
                image = self.image_processor.load_local_image(image_path)
            else:
                image = self.image_processor.download_image(image_path)
            
            if image is None:
                return None
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy().flatten()
            
            # Cache the result
            self.cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            return None
    
    def generate_influencer_embeddings(self, influencer: InfluencerData) -> Dict[str, Optional[np.ndarray]]:
        """
        Generate all embeddings for an influencer.
        
        Args:
            influencer: Influencer data
            
        Returns:
            Dictionary with text, profile, and content embeddings
        """
        embeddings = {
            'text': None,
            'profile': None,
            'content': None
        }
        
        # Generate text embedding from bio
        if influencer.bio:
            text_content = f"{influencer.name} {influencer.bio} {influencer.category}"
            embeddings['text'] = self.generate_text_embedding(text_content)
        
        # Generate profile image embedding
        if influencer.profile_photo_url:
            embeddings['profile'] = self.generate_image_embedding(influencer.profile_photo_url)
        
        # Generate content image embedding
        if influencer.content_thumbnail_url:
            embeddings['content'] = self.generate_image_embedding(influencer.content_thumbnail_url)
        
        return embeddings
    
    def batch_generate_text_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts efficiently."""
        embeddings = []
        for text in texts:
            embedding = self.generate_text_embedding(text)
            embeddings.append(embedding)
        return embeddings

    
    def batch_generate_influencer_embeddings(self, influencers: List[InfluencerData]) -> List[Dict[str, Optional[np.ndarray]]]:
        """Generate embeddings for multiple influencers efficiently."""
        embeddings_list = []
        for influencer in influencers:
            embeddings = self.generate_influencer_embeddings(influencer)
            embeddings_list.append(embeddings)
        return embeddings_list
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache.cache_dir.glob("*.pkl"))
        return {
            'cache_enabled': self.cache.enabled,
            'cache_dir': str(self.cache.cache_dir),
            'cached_embeddings': len(cache_files),
            'cache_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        }


# Global embedding service instance
_embedding_service = None


def get_embedding_service() -> CLIPEmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = CLIPEmbeddingService()
    return _embedding_service


async def test_embedding_service():
    """Test the embedding service functionality."""
    print("\n🔍 Testing Embedding Service...")
    
    try:
        service = get_embedding_service()
        
        # Test text embedding
        text_embedding = service.generate_text_embedding("fitness influencer")
        if text_embedding is not None:
            print(f"✅ Text embedding generated: {text_embedding.shape}")
        else:
            print("❌ Text embedding failed")
            return False
        
        # Test cache stats
        cache_stats = service.get_cache_stats()
        print(f"✅ Cache stats: {cache_stats}")
        
        print("✅ Embedding service tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False 