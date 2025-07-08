"""
Data loading utilities for the Influencer Discovery Tool.

This module provides functions to load, validate, and process influencer data
from various sources including CSV files, Excel files, and databases.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.schemas import InfluencerData, InfluencerCategory
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass


class ImageURLValidator:
    """Validates image URLs and checks if they're accessible."""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """Initialize with timeout and retry settings."""
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if a URL is properly formatted and accessible.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid and accessible, False otherwise
        """
        try:
            # Basic URL format validation
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check if URL is accessible (HEAD request)
            response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
            return response.status_code == 200
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"URL validation failed for {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating URL {url}: {e}")
            return False
    
    def batch_validate(self, urls: List[str]) -> Dict[str, bool]:
        """
        Validate multiple URLs in batch.
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            Dictionary mapping URLs to their validation status
        """
        results = {}
        for url in urls:
            results[url] = self.validate_url(url)
        return results


class InfluencerDataLoader:
    """Loads and validates influencer data from various sources."""
    
    def __init__(self, validate_images: bool = False):
        """
        Initialize the data loader.
        
        Args:
            validate_images: Whether to validate image URLs (slower but more thorough)
        """
        self.validate_images = validate_images
        self.image_validator = ImageURLValidator() if validate_images else None
        self.loaded_data: List[InfluencerData] = []
        self.validation_errors: List[Dict[str, Any]] = []
    
    def load_from_csv(self, file_path: Union[str, Path]) -> List[InfluencerData]:
        """
        Load influencer data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of validated InfluencerData objects
            
        Raises:
            DataLoadError: If the file cannot be loaded or contains invalid data
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DataLoadError(f"File not found: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            return self._process_dataframe(df, str(file_path))
            
        except pd.errors.EmptyDataError:
            raise DataLoadError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise DataLoadError(f"Error parsing CSV file: {e}")
        except Exception as e:
            raise DataLoadError(f"Unexpected error loading CSV: {e}")
    
    def load_from_excel(self, file_path: Union[str, Path], sheet_name: str = None) -> List[InfluencerData]:
        """
        Load influencer data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Sheet name to load (default: first sheet)
            
        Returns:
            List of validated InfluencerData objects
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DataLoadError(f"File not found: {file_path}")
            
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            return self._process_dataframe(df, str(file_path))
            
        except Exception as e:
            raise DataLoadError(f"Error loading Excel file: {e}")
    
    def load_from_dict_list(self, data: List[Dict[str, Any]]) -> List[InfluencerData]:
        """
        Load influencer data from a list of dictionaries.
        
        Args:
            data: List of dictionaries containing influencer data
            
        Returns:
            List of validated InfluencerData objects
        """
        df = pd.DataFrame(data)
        return self._process_dataframe(df, "dict_list")
    
    def _process_dataframe(self, df: pd.DataFrame, source: str) -> List[InfluencerData]:
        """
        Process a pandas DataFrame and validate the data.
        
        Args:
            df: DataFrame containing influencer data
            source: Source identifier for logging
            
        Returns:
            List of validated InfluencerData objects
        """
        self.loaded_data = []
        self.validation_errors = []
        
        # Check if this is the new Instagram CSV format
        if 'Handle' in df.columns and 'Full_Name' in df.columns:
            return self._process_instagram_dataframe(df, source)
        
        # Original format check
        required_columns = [
            'influencer_id', 'name', 'bio', 'category', 
            'follower_count', 'profile_photo_url', 'content_thumbnail_url'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataLoadError(f"Missing required columns: {missing_columns}")
        
        # Validate image URLs if requested
        if self.validate_images:
            self._validate_image_urls(df)
        
        # Process each row
        for index, row in df.iterrows():
            try:
                # Create InfluencerData object
                influencer_data = InfluencerData(
                    influencer_id=str(row['influencer_id']),
                    name=str(row['name']),
                    bio=str(row['bio']),
                    category=str(row['category']).lower(),
                    follower_count=int(row['follower_count']),
                    profile_photo_url=str(row['profile_photo_url']),
                    content_thumbnail_url=str(row['content_thumbnail_url'])
                )
                
                self.loaded_data.append(influencer_data)
                
            except ValidationError as e:
                self.validation_errors.append({
                    'row': index,
                    'data': row.to_dict(),
                    'errors': e.errors(),
                    'source': source
                })
                logger.warning(f"Validation error for row {index}: {e}")
            except Exception as e:
                self.validation_errors.append({
                    'row': index,
                    'data': row.to_dict(),
                    'errors': [{'msg': str(e)}],
                    'source': source
                })
                logger.error(f"Unexpected error processing row {index}: {e}")
        
        logger.info(f"Successfully processed {len(self.loaded_data)} influencers from {source}")
        if self.validation_errors:
            logger.warning(f"Found {len(self.validation_errors)} validation errors")
        
        return self.loaded_data

    def _process_instagram_dataframe(self, df: pd.DataFrame, source: str) -> List[InfluencerData]:
        """
        Process the Instagram CSV format DataFrame.
        
        Args:
            df: DataFrame containing Instagram influencer data
            source: Source identifier for logging
            
        Returns:
            List of validated InfluencerData objects
        """
        self.loaded_data = []
        self.validation_errors = []
        
        # Category mapping for normalization
        category_mapping = {
            'gaming': 'gaming',
            'lifestyle': 'lifestyle', 
            'tech': 'tech',
            'fitness': 'fitness',
            'food': 'food',
            'beauty': 'beauty',
            'fashion': 'fashion',
            'travel': 'travel',
            'wellness': 'wellness'
        }
        
        # Validate image URLs if requested
        if self.validate_images:
            profile_urls = df['Profile_Photo_URL'].tolist()
            content_urls = df['Latest_Post_Thumbnail'].tolist()
            all_urls = profile_urls + content_urls
            validation_results = self.image_validator.batch_validate(all_urls)
            logger.info(f"Image validation completed for {len(all_urls)} URLs")
        
        # Process each row
        for index, row in df.iterrows():
            try:
                # Normalize category
                category = str(row['Category']).lower().strip()
                normalized_category = category_mapping.get(category, category)
                
                # Handle boolean fields
                is_verified = row.get('Is_Verified', False)
                if isinstance(is_verified, str):
                    is_verified = is_verified.lower() in ['true', '1', 'yes']
                
                is_private = row.get('Is_Private', False)
                if isinstance(is_private, str):
                    is_private = is_private.lower() in ['true', '1', 'yes']
                
                # Create InfluencerData object
                influencer_data = InfluencerData(
                    influencer_id=str(row['Handle']),
                    username=str(row['Username']) if pd.notna(row.get('Username')) else None,
                    name=str(row['Full_Name']),
                    bio=str(row['Bio']),
                    category=normalized_category,
                    follower_count=int(row['Follower_Count']),
                    following_count=int(row['Following_Count']) if pd.notna(row.get('Following_Count')) else None,
                    post_count=int(row['Post_Count']) if pd.notna(row.get('Post_Count')) else None,
                    profile_photo_url=str(row['Profile_Photo_URL']),
                    content_thumbnail_url=str(row['Latest_Post_Thumbnail']),
                    instagram_url=str(row['Instagram_URL']) if pd.notna(row.get('Instagram_URL')) else None,
                    is_verified=is_verified,
                    is_private=is_private
                )
                
                self.loaded_data.append(influencer_data)
                
            except ValidationError as e:
                self.validation_errors.append({
                    'row': index,
                    'data': row.to_dict(),
                    'errors': e.errors(),
                    'source': source
                })
                logger.warning(f"Validation error for row {index}: {e}")
            except Exception as e:
                self.validation_errors.append({
                    'row': index,
                    'data': row.to_dict(),
                    'errors': [{'msg': str(e)}],
                    'source': source
                })
                logger.error(f"Unexpected error processing row {index}: {e}")
        
        logger.info(f"Successfully processed {len(self.loaded_data)} influencers from Instagram CSV")
        if self.validation_errors:
            logger.warning(f"Found {len(self.validation_errors)} validation errors")
        
        return self.loaded_data
    
    def _validate_image_urls(self, df: pd.DataFrame) -> None:
        """Validate all image URLs in the DataFrame."""
        logger.info("Validating image URLs...")
        
        # Collect all unique URLs
        profile_urls = df['profile_photo_url'].dropna().unique().tolist()
        content_urls = df['content_thumbnail_url'].dropna().unique().tolist()
        all_urls = list(set(profile_urls + content_urls))
        
        # Validate URLs
        url_results = self.image_validator.batch_validate(all_urls)
        
        # Log results
        valid_count = sum(1 for result in url_results.values() if result)
        logger.info(f"Image URL validation: {valid_count}/{len(all_urls)} URLs are accessible")
        
        # Log invalid URLs
        invalid_urls = [url for url, valid in url_results.items() if not valid]
        if invalid_urls:
            logger.warning(f"Invalid or inaccessible image URLs: {invalid_urls}")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get a detailed validation report.
        
        Returns:
            Dictionary containing validation statistics and errors
        """
        return {
            'total_loaded': len(self.loaded_data),
            'total_errors': len(self.validation_errors),
            'success_rate': len(self.loaded_data) / (len(self.loaded_data) + len(self.validation_errors)) if (len(self.loaded_data) + len(self.validation_errors)) > 0 else 0,
            'errors': self.validation_errors
        }
    
    def save_errors_to_file(self, file_path: Union[str, Path]) -> None:
        """Save validation errors to a file for review."""
        if not self.validation_errors:
            logger.info("No validation errors to save")
            return
        
        error_df = pd.DataFrame(self.validation_errors)
        error_df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(self.validation_errors)} validation errors to {file_path}")


def load_sample_data(validate_images: bool = False) -> List[InfluencerData]:
    """
    Convenience function to load the sample influencer data.
    
    Args:
        validate_images: Whether to validate image URLs
        
    Returns:
        List of validated InfluencerData objects
    """
    data_dir = Path(__file__).parent.parent / "data"
    csv_file = data_dir / "sample_influencers.csv"
    
    loader = InfluencerDataLoader(validate_images=validate_images)
    return loader.load_from_csv(csv_file)


def get_category_distribution(influencers: List[InfluencerData]) -> Dict[str, int]:
    """
    Get the distribution of influencers by category.
    
    Args:
        influencers: List of influencer data
        
    Returns:
        Dictionary mapping categories to counts
    """
    distribution = {}
    for influencer in influencers:
        category = influencer.category
        distribution[category] = distribution.get(category, 0) + 1
    
    return distribution


def get_follower_stats(influencers: List[InfluencerData]) -> Dict[str, Any]:
    """
    Get follower count statistics.
    
    Args:
        influencers: List of influencer data
        
    Returns:
        Dictionary containing follower statistics
    """
    if not influencers:
        return {}
    
    follower_counts = [inf.follower_count for inf in influencers]
    
    return {
        'total_influencers': len(influencers),
        'min_followers': min(follower_counts),
        'max_followers': max(follower_counts),
        'avg_followers': sum(follower_counts) / len(follower_counts),
        'total_reach': sum(follower_counts)
    }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load sample data
        influencers = load_sample_data(validate_images=False)
        print(f"Loaded {len(influencers)} influencers")
        
        # Print statistics
        print("\nCategory distribution:")
        for category, count in get_category_distribution(influencers).items():
            print(f"  {category}: {count}")
        
        print("\nFollower statistics:")
        stats = get_follower_stats(influencers)
        for key, value in stats.items():
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.1f}")
        
    except Exception as e:
        print(f"Error: {e}") 