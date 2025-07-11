#!/usr/bin/env python3
"""
Influencer Discovery Tool - Web Application

A FastAPI-based web application for discovering influencers using
multimodal search with CLIP embeddings.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.search_engine import get_search_engine
from app.schemas import SearchRequest, SearchFilters, SearchResponse
from app.workflow import SimpleInfluencerResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Influencer Discovery Tool",
    description="AI-powered influencer discovery using multimodal search",
    version="1.0.0"
)
app.mount("/data", StaticFiles(directory="data"), name="data")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
search_engine = get_search_engine()
response_handler = SimpleInfluencerResponse()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create directories if they don't exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("static/css").mkdir(exist_ok=True)
Path("static/js").mkdir(exist_ok=True)


# API Models
class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    category: Optional[str] = None
    min_followers: Optional[int] = None
    max_followers: Optional[int] = None


class SearchResultAPI(BaseModel):
    influencer_id: str
    name: str
    username: Optional[str]
    bio: str
    category: str
    follower_count: int
    following_count: Optional[int]
    post_count: Optional[int]
    profile_photo_url: str
    content_thumbnail_url: str
    instagram_url: Optional[str]
    is_verified: bool
    is_private: bool
    similarity_score: float
    match_reasons: List[str]


class SearchResponseAPI(BaseModel):
    query: str
    results: List[SearchResultAPI]
    total_count: int
    processing_time_ms: float
    page_info: Dict[str, Any]


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main search interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = search_engine.get_stats()
        return {
            "status": "healthy",
            "total_influencers": stats.get("total_influencers", 0),
            "embedding_types": {
                "text": stats.get("text_embeddings", 0),
                "profile": stats.get("profile_embeddings", 0),
                "content": stats.get("content_embeddings", 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable")


@app.post("/api/search", response_model=SearchResponseAPI)
async def search_influencers(search_query: SearchQuery):
    """Search for influencers using the multimodal search engine."""
    try:
        # Create search request
        filters = None
        if search_query.category or search_query.min_followers or search_query.max_followers:
            filters = SearchFilters(
                category=search_query.category,
                min_followers=search_query.min_followers,
                max_followers=search_query.max_followers
            )
        
        request = SearchRequest(
            query=search_query.query,
            limit=search_query.limit,
            offset=0,
            filters=filters
        )
        
        # Perform search
        response = search_engine.search(request)
        
        # Convert to API format
        api_results = []
        for result in response.results:
            inf = result.influencer
            api_results.append(SearchResultAPI(
                influencer_id=inf.influencer_id,
                name=inf.name,
                username=inf.username,
                bio=inf.bio,
                category=inf.category,
                follower_count=inf.follower_count,
                following_count=inf.following_count,
                post_count=inf.post_count,
                profile_photo_url=inf.profile_photo_url,
                content_thumbnail_url=inf.content_thumbnail_url,
                instagram_url=str(inf.instagram_url) if inf.instagram_url is not None else None,
                is_verified=inf.is_verified,
                is_private=inf.is_private,
                similarity_score=result.similarity_score,
                match_reasons=result.match_reasons
            ))
        
        return SearchResponseAPI(
            query=response.query,
            results=api_results,
            total_count=response.total_count,
            processing_time_ms=response.processing_time_ms,
            page_info=response.page_info
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/categories")
async def get_categories():
    """Get available influencer categories."""
    try:
        stats = search_engine.get_stats()
        # Extract categories from vector store data
        categories = set()
        for influencer in search_engine.vector_store.influencer_data:
            categories.add(influencer.category)
        
        return {"categories": sorted(list(categories))}
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        return {"categories": []}


@app.get("/api/suggestions")
async def get_search_suggestions(q: str = Query(..., min_length=1)):
    """Get search suggestions based on partial query."""
    try:
        suggestions = search_engine.get_search_suggestions(q)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        return {"suggestions": []}


@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        stats = search_engine.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system stats")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested resource was not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    # Load search index on startup
    logger.info("Starting Influencer Discovery Tool...")
    
    # Check if vector store is loaded
    stats = search_engine.get_stats()
    if stats.get("total_influencers", 0) == 0:
        logger.info("Loading search index...")
        if not search_engine.vector_store.load():
            logger.error("Failed to load search index. Please ensure data is initialized.")
            sys.exit(1)
    
    logger.info(f"Search index loaded with {stats.get('total_influencers', 0)} influencers")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
