"""
Data validation schemas for the Influencer Discovery Tool.

This module contains Pydantic models for validating influencer data,
search requests, and API responses.
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class InfluencerCategory(str, Enum):
    """Valid influencer categories."""
    FITNESS = "fitness"
    BEAUTY = "beauty"
    TECH = "tech"
    LIFESTYLE = "lifestyle"
    FOOD = "food"
    FASHION = "fashion"
    TRAVEL = "travel"
    GAMING = "gaming"
    WELLNESS = "wellness"


class InfluencerData(BaseModel):
    """Model for influencer data validation."""
    
    influencer_id: str = Field(..., description="Unique identifier for the influencer")
    username: Optional[str] = Field(None, description="Username without @ symbol")
    name: str = Field(..., min_length=1, max_length=100, description="Influencer's full name")
    bio: str = Field(..., min_length=1, max_length=1000, description="Influencer's bio/description")
    category: InfluencerCategory = Field(..., description="Primary content category")
    follower_count: int = Field(..., ge=0, le=100_000_000, description="Number of followers")
    following_count: Optional[int] = Field(None, ge=0, description="Number of accounts following")
    post_count: Optional[int] = Field(None, ge=0, description="Number of posts")
    profile_photo_url: Union[HttpUrl, str] = Field(..., description="URL or local path to profile photo")
    content_thumbnail_url: Union[HttpUrl, str] = Field(..., description="URL or local path to content thumbnail")
    instagram_url: Optional[HttpUrl] = Field(None, description="Instagram profile URL")
    is_verified: Optional[bool] = Field(False, description="Whether the account is verified")
    is_private: Optional[bool] = Field(False, description="Whether the account is private")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate name contains only allowed characters."""
        if not v.strip():
            raise ValueError('Name cannot be empty or whitespace only')
        return v.strip()
    
    @validator('bio')
    def validate_bio(cls, v):
        """Validate bio content."""
        if not v.strip():
            raise ValueError('Bio cannot be empty or whitespace only')
        return v.strip()
    
    @validator('profile_photo_url', 'content_thumbnail_url', pre=True)
    def validate_image_path(cls, v):
        """Accept both URLs and local file paths for image locations."""
        if isinstance(v, str):
            # Check if it's a local path (contains path separators and no http)
            if ('\\' in v or '/' in v) and not v.startswith(('http://', 'https://')):
                # It's a local path - return as string
                return v
            # Try to validate as URL
            try:
                from pydantic import HttpUrl
                return HttpUrl(v)
            except:
                # If URL validation fails, return as string (might be a malformed URL)
                return v
        return v
    
    class Config:
        """Pydantic config."""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "influencer_id": "@pewdiepie",
                "username": "pewdiepie",
                "name": "PewDiePie",
                "bio": "Gamer;Subscribe to my newsletter with @itsmarziapie",
                "category": "gaming",
                "follower_count": 20400000,
                "following_count": 1,
                "post_count": 788,
                "profile_photo_url": "https://images.unsplash.com/photo-1594736797933-d0200b5d2c84",
                "content_thumbnail_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b",
                "instagram_url": "https://www.instagram.com/pewdiepie/",
                "is_verified": True,
                "is_private": False
            }
        }


class SearchFilters(BaseModel):
    """Model for search filters."""
    
    category: Optional[InfluencerCategory] = Field(None, description="Filter by category")
    min_followers: Optional[int] = Field(None, ge=0, description="Minimum follower count")
    max_followers: Optional[int] = Field(None, ge=0, description="Maximum follower count")
    
    @validator('max_followers')
    def validate_follower_range(cls, v, values):
        """Ensure max_followers is greater than min_followers."""
        if v is not None and 'min_followers' in values and values['min_followers'] is not None:
            if v < values['min_followers']:
                raise ValueError('max_followers must be greater than min_followers')
        return v


class SearchRequest(BaseModel):
    """Model for search requests."""
    
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    filters: Optional[SearchFilters] = Field(None, description="Additional search filters")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Result offset for pagination")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate search query."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "query": "fitness influencers with curly hair",
                "filters": {
                    "category": "fitness",
                    "min_followers": 50000
                },
                "limit": 10,
                "offset": 0
            }
        }


class SearchResult(BaseModel):
    """Model for individual search results."""
    
    influencer: InfluencerData = Field(..., description="Influencer data")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    match_reasons: List[str] = Field(default_factory=list, description="Reasons for the match")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "influencer": {
                    "influencer_id": "INF001",
                    "name": "Emma Rodriguez",
                    "bio": "Fitness enthusiast 💪 | Personal trainer | Healthy lifestyle advocate | Curly hair, don't care! 🌿",
                    "category": "fitness",
                    "follower_count": 125000,
                    "profile_photo_url": "https://images.unsplash.com/photo-1594736797933-d0200b5d2c84",
                    "content_thumbnail_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b"
                },
                "similarity_score": 0.87,
                "match_reasons": ["Category match: fitness", "Bio contains: curly hair"]
            }
        }


class SearchResponse(BaseModel):
    """Model for search responses."""
    
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., ge=0, description="Total number of matching results")
    page_info: Dict[str, Any] = Field(..., description="Pagination information")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "query": "fitness influencers with curly hair",
                "results": [],
                "total_count": 15,
                "page_info": {
                    "limit": 10,
                    "offset": 0,
                    "has_next": True,
                    "has_previous": False
                },
                "processing_time_ms": 124.5
            }
        }


class InfluencerBatch(BaseModel):
    """Model for batch influencer data upload."""
    
    influencers: List[InfluencerData] = Field(..., min_items=1, max_items=100, description="List of influencers")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "influencers": [
                    {
                        "influencer_id": "INF001",
                        "name": "Emma Rodriguez",
                        "bio": "Fitness enthusiast 💪 | Personal trainer | Healthy lifestyle advocate | Curly hair, don't care! 🌿",
                        "category": "fitness",
                        "follower_count": 125000,
                        "profile_photo_url": "https://images.unsplash.com/photo-1594736797933-d0200b5d2c84",
                        "content_thumbnail_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b"
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Model for error responses."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "error": "Validation error",
                "detail": "Query cannot be empty",
                "code": "INVALID_INPUT"
            }
        }


class HealthResponse(BaseModel):
    """Model for health check responses."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "components": {
                    "database": "healthy",
                    "vector_store": "healthy",
                    "clip_model": "healthy"
                }
            }
        } 
