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

from llama_index.embeddings.clip import ClipEmbedding
from app.settings import config, get_clip_embedding_model
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
        Load and preprocess image from local file path.
        
        Args:
            file_path: Local path to image file
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            # Convert to Path object for proper cross-platform handling
            path = Path(file_path).resolve()
            
            # Check if file exists
            if not path.exists():
                logger.warning(f"Local image file not found: {file_path}")
                return None
            
            # Log file details for debugging
            logger.debug(f"Loading image: {path} (size: {path.stat().st_size} bytes)")
            
            # Load image with explicit path string conversion
            image = Image.open(str(path))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.debug(f"Converted image from {image.mode} to RGB")
            
            # Resize if too large
            original_size = image.size
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {original_size} to {image.size}")
            
            logger.debug(f"Successfully loaded image: {path}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load local image from {file_path}: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            return None
    
    def is_local_path(self, path: str) -> bool:
        """
        Determine if a path is a local file path or a URL.
        
        Args:
            path: Path or URL string
            
        Returns:
            True if local path, False if URL
        """
        if not path:
            return False
        
        # Check for URL schemes
        if path.startswith(('http://', 'https://', 'ftp://')):
            return False
        
        # Check for local path indicators
        if any(indicator in path for indicator in ['\\', '/', '.']):
            # Could be a local path
            return True
        
        return False


class CLIPEmbeddingService:
    """Service for generating CLIP embeddings for text and images."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.clip_model = get_clip_embedding_model()
        self.cache = EmbeddingCache()
        self.image_processor = ImageProcessor()
        self.batch_size = config.batch_size
        logger.info(f"Initialized CLIP embedding service with model: {config.clip_model_name}")
    
    def generate_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text content.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array embedding or None if failed
        """
        if not text or not text.strip():
            return None
        
        # Check cache first
        cache_key = self.cache._get_cache_key(text)
        cached_embedding = self.cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Generate embedding using CLIP
            embedding = self.clip_model.get_text_embedding(text.strip())
            
            # Convert to numpy array
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            elif torch.is_tensor(embedding):
                embedding = embedding.cpu().numpy().astype(np.float32)
            
            # Cache the result
            self.cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return None
    
    def generate_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate embedding for image from URL or local file path.
        
        Args:
            image_path: URL or local file path of image to embed
            
        Returns:
            Numpy array embedding or None if failed
        """
        if not image_path:
            return None
        
        # Check cache first
        cache_key = self.cache._get_cache_key(image_path)
        cached_embedding = self.cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Handle local paths vs URLs differently for CLIP
            if self.image_processor.is_local_path(image_path):
                # For local paths, pass the path directly to CLIP (LlamaIndex expects file paths)
                logger.debug(f"Using local image path: {image_path}")
                # Convert to absolute path for CLIP
                absolute_path = str(Path(image_path).resolve())
                embedding = self.clip_model.get_image_embedding(absolute_path)
            else:
                # For URLs, download first then save temporarily for CLIP
                logger.debug(f"Downloading image from URL: {image_path}")
                image = self.image_processor.download_image(image_path)
                if image is None:
                    return None
                
                # Save to temporary file for CLIP processing
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image.save(tmp_file.name, 'JPEG')
                    tmp_path = tmp_file.name
                
                try:
                    embedding = self.clip_model.get_image_embedding(tmp_path)
                finally:
                    # Clean up temporary file
                    Path(tmp_path).unlink(missing_ok=True)
            
            # Convert to numpy array
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            elif torch.is_tensor(embedding):
                embedding = embedding.cpu().numpy().astype(np.float32)
            
            # Cache the result
            self.cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding for {image_path}: {e}")
            return None
    
    def generate_influencer_embeddings(self, influencer: InfluencerData) -> Dict[str, Optional[np.ndarray]]:
        """
        Generate all embeddings for an influencer (bio, profile photo, content thumbnail).
        
        Args:
            influencer: InfluencerData object
            
        Returns:
            Dictionary with embedding types as keys and numpy arrays as values
        """
        embeddings = {}
        
        # Generate text embedding from bio
        bio_embedding = self.generate_text_embedding(influencer.bio)
        embeddings['bio'] = bio_embedding
        
        # Generate profile photo embedding
        profile_embedding = self.generate_image_embedding(str(influencer.profile_photo_url))
        embeddings['profile_photo'] = profile_embedding
        
        # Generate content thumbnail embedding
        content_embedding = self.generate_image_embedding(str(influencer.content_thumbnail_url))
        embeddings['content_thumbnail'] = content_embedding
        
        # Log results
        success_count = sum(1 for emb in embeddings.values() if emb is not None)
        logger.info(f"Generated {success_count}/3 embeddings for {influencer.name}")
        
        return embeddings
    
    def batch_generate_text_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (same order as input)
        """
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_text_embedding(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            logger.info(f"Processed text batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
        
        return embeddings
    
    def batch_generate_image_embeddings(self, image_urls: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            image_urls: List of image URLs to embed
            
        Returns:
            List of embeddings (same order as input)
        """
        embeddings = []
        
        for i in range(0, len(image_urls), self.batch_size):
            batch = image_urls[i:i + self.batch_size]
            batch_embeddings = []
            
            for url in batch:
                embedding = self.generate_image_embedding(url)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            logger.info(f"Processed image batch {i//self.batch_size + 1}/{(len(image_urls) + self.batch_size - 1)//self.batch_size}")
        
        return embeddings
    
    def batch_generate_influencer_embeddings(self, influencers: List[InfluencerData]) -> List[Dict[str, Optional[np.ndarray]]]:
        """
        Generate embeddings for multiple influencers.
        
        Args:
            influencers: List of InfluencerData objects
            
        Returns:
            List of embedding dictionaries (same order as input)
        """
        all_embeddings = []
        
        for i, influencer in enumerate(influencers):
            embeddings = self.generate_influencer_embeddings(influencer)
            all_embeddings.append(embeddings)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(influencers)} influencers")
        
        return all_embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'enabled': self.cache.enabled,
            'cache_dir': str(self.cache.cache_dir),
            'cached_items': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024)
        }


# Global embedding service instance
embedding_service = CLIPEmbeddingService()


def get_embedding_service() -> CLIPEmbeddingService:
    """Get the global embedding service instance."""
    return embedding_service


# Test function for development
async def test_embedding_service():
    """Test the embedding service with sample data."""
    logger.info("Testing CLIP embedding service...")
    
    # Test text embedding
    test_text = "Fitness enthusiast with curly hair"
    text_embedding = embedding_service.generate_text_embedding(test_text)
    logger.info(f"Text embedding shape: {text_embedding.shape if text_embedding is not None else 'None'}")
    
    # Test image embedding
    test_image_url = "https://images.unsplash.com/photo-1594736797933-d0200b5d2c84?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
    image_embedding = embedding_service.generate_image_embedding(test_image_url)
    logger.info(f"Image embedding shape: {image_embedding.shape if image_embedding is not None else 'None'}")
    
    # Test cache stats
    cache_stats = embedding_service.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")


if __name__ == "__main__":
    asyncio.run(test_embedding_service()) 