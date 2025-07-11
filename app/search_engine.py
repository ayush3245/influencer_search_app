"""
Search Engine for Influencer Discovery

This module provides high-level search functionality combining text/image embeddings
with metadata filtering and result ranking.
"""

import logging
from typing import List, Optional, Dict, Any, Union
import time
import numpy as np
from app.embedding_service import get_embedding_service
from app.vector_store import vector_store
from app.schemas import SearchRequest, SearchResponse, SearchResult, SearchFilters, InfluencerData
from app.vector_store import VectorSearchResult

logger = logging.getLogger(__name__)


class InfluencerSearchEngine:
    """
    High-level search engine for influencer discovery.
    
    Provides semantic search across influencer bios, profile photos, and content
    with intelligent ranking and filtering capabilities.
    """
    
    def __init__(self):
        """Initialize the search engine."""
        self.embedding_service = get_embedding_service()
        self.vector_store = vector_store
        self._search_history: List[Dict[str, Any]] = []
    
    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform comprehensive influencer search using text query that leverages
        bio, profile image, and content image embeddings in the background.
        
        Args:
            request: SearchRequest with text query and parameters
            
        Returns:
            SearchResponse with ranked results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing text search: '{request.query}' with {request.limit} results")
            
            # Generate text embedding for the query
            text_embedding = self.embedding_service.generate_text_embedding(request.query)
            if text_embedding is None:
                logger.warning("Failed to generate text embedding for query")
                # Fallback to metadata-only search
                results = self._metadata_search(request)
            else:
                # Perform comprehensive search using text embedding against all embedding types
                raw_results = self._search_comprehensive(text_embedding, request)
                
                # Convert to schema format
                results = [
                    SearchResult(
                        influencer=result.influencer_data,
                        similarity_score=result.score,
                        match_reasons=[f"Semantic similarity: {result.score:.3f}"]
                    )
                    for result in raw_results
                ]
            
            # Post-process and rank results
            ranked_results = self._rank_results(results, request)
            
            # Create response
            elapsed_time = time.time() - start_time
            response = SearchResponse(
                query=request.query,
                results=ranked_results,
                total_count=len(ranked_results),
                page_info={
                    "limit": request.limit,
                    "offset": request.offset,
                    "has_next": False,
                    "has_previous": False
                },
                processing_time_ms=elapsed_time * 1000
            )
            
            # Log search
            self._log_search(request, response)
            
            logger.info(f"Search completed: {len(ranked_results)} results in {elapsed_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return SearchResponse(
                query=request.query,
                results=[],
                total_count=0,
                page_info={
                    "limit": request.limit,
                    "offset": request.offset,
                    "has_next": False,
                    "has_previous": False
                },
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def search_text(
        self,
        query: str,
        limit: int = 10,
        categories: Optional[List[str]] = None,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Simplified text search interface that leverages bio, profile image, and content image embeddings.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            categories: Filter by categories
            min_followers: Minimum follower count
            max_followers: Maximum follower count
            
        Returns:
            List of search results
        """
        filters = None
        if categories or min_followers or max_followers:
            filters = SearchFilters(
                category=categories[0] if categories else None,  # Schema only supports single category
                min_followers=min_followers,
                max_followers=max_followers
            )
        
        request = SearchRequest(
            query=query,
            limit=limit,
            offset=0,
            filters=filters
        )
        
        response = self.search(request)
        return response.results
    
    def search_by_category(
        self,
        category: str,
        limit: int = 10,
        query: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search influencers by category with optional text query.
        
        Args:
            category: Influencer category to filter by
            limit: Maximum number of results
            query: Optional additional text query
            
        Returns:
            List of search results
        """
        return self.search_text(
            query=query or f"{category} influencer",
            limit=limit,
            categories=[category]
        )
    
    def find_similar_influencers(
        self,
        influencer_id: str,
        limit: int = 5,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find influencers similar to a given influencer.
        
        Args:
            influencer_id: ID of the reference influencer
            limit: Maximum number of results
            exclude_self: Whether to exclude the reference influencer
            
        Returns:
            List of similar influencers
        """
        # Get reference influencer
        ref_influencer = self.vector_store.get_influencer(influencer_id)
        if not ref_influencer:
            logger.warning(f"Influencer {influencer_id} not found")
            return []
        
        # Use bio as search query
        results = self.search_text(
            query=ref_influencer.bio,
            limit=limit + (1 if exclude_self else 0)
        )
        
        # Filter out the reference influencer if requested
        if exclude_self:
            results = [r for r in results if r.influencer.influencer_id != influencer_id]
            results = results[:limit]
        
        return results
    
    def _metadata_search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Fallback search using only metadata when embeddings fail.
        
        Args:
            request: Search request
            
        Returns:
            List of search results based on metadata
        """
        logger.info("Performing metadata-only search")
        
        all_influencers = []
        for i in range(len(self.vector_store.influencer_data)):
            influencer = self.vector_store.influencer_data[i]
            
            # Apply filters
            if request.filters and not self.vector_store._apply_filters(influencer, request.filters):
                continue
            
            # Simple text matching if query provided
            score = 1.0
            if request.query:
                query_lower = request.query.lower()
                bio_lower = influencer.bio.lower()
                name_lower = influencer.name.lower()
                
                # Basic text similarity scoring
                if query_lower in bio_lower:
                    score += 0.5
                if query_lower in name_lower:
                    score += 0.3
                if any(word in bio_lower for word in query_lower.split()):
                    score += 0.2
            
            result = SearchResult(
                influencer=influencer,
                similarity_score=score,
                match_reasons=[f"Text match: {score:.3f}"]
            )
            all_influencers.append(result)
        
        # Sort by score and return top results
        all_influencers.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_influencers[:request.limit]
    
    def _rank_results(self, results: List[SearchResult], request: SearchRequest) -> List[SearchResult]:
        """
        Apply additional ranking and filtering to search results.
        
        Args:
            results: Raw search results
            request: Original search request
            
        Returns:
            Ranked and filtered results
        """
        if not results:
            return results
        
        # Enhanced ranking with category boost
        enhanced_results = []
        query_lower = request.query.lower() if request.query else ""
        
        # Define category boost factor
        CATEGORY_BOOST = 0.15  # Add 15% boost for category matches
        
        for i, result in enumerate(results):
            enhanced_result = result
            original_score = result.similarity_score
            boosted_score = original_score
            boost_reasons = []
            
            # Category Boost Logic
            if query_lower and result.influencer.category:
                influencer_category = result.influencer.category.lower()
                
                # 1. Direct category match boost (highest priority)
                if query_lower == influencer_category:
                    boosted_score += CATEGORY_BOOST
                    boost_reasons.append(f"Exact category match: +{CATEGORY_BOOST:.3f}")
                
                # 2. Partial match for multi-word queries or partial words
                elif query_lower in influencer_category or influencer_category in query_lower:
                    partial_boost = CATEGORY_BOOST * 0.7  # 70% of full boost
                    boosted_score += partial_boost
                    boost_reasons.append(f"Partial category match: +{partial_boost:.3f}")
            
            # Update similarity score
            enhanced_result.similarity_score = boosted_score
            
            # Update match reasons for debugging and transparency
            rank_info = f"Rank: {i + 1}/{len(results)}"
            match_reasons = result.match_reasons.copy()
            
            if boost_reasons:
                match_reasons.extend(boost_reasons)
                match_reasons.append(f"Boosted score: {original_score:.4f} → {boosted_score:.4f}")
            
            if rank_info not in match_reasons:
                match_reasons.append(rank_info)
            
            enhanced_result.match_reasons = match_reasons
            enhanced_results.append(enhanced_result)
        
        # Re-sort by boosted scores
        enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks after re-sorting
        for i, result in enumerate(enhanced_results):
            # Update rank in match reasons
            result.match_reasons = [reason for reason in result.match_reasons if not reason.startswith("Rank:")]
            result.match_reasons.append(f"Final rank: {i + 1}/{len(enhanced_results)}")
        
        return enhanced_results
    
    def _log_search(self, request: SearchRequest, response: SearchResponse) -> None:
        """Log search for analytics and debugging."""
        search_log = {
            'timestamp': time.time(),
            'query': request.query,
            'results_count': len(response.results),
            'processing_time_ms': response.processing_time_ms,
            'filters': request.filters.model_dump() if request.filters else None
        }
        
        self._search_history.append(search_log)
        
        # Keep only last 100 searches
        if len(self._search_history) > 100:
            self._search_history = self._search_history[-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        total_searches = len(self._search_history)
        
        if total_searches == 0:
            return {
                'total_searches': 0,
                'vector_store_stats': self.vector_store.get_stats()
            }
        
        avg_results = sum(s['results_count'] for s in self._search_history) / total_searches
        avg_time = sum(s['processing_time_ms'] for s in self._search_history) / total_searches
        
        return {
            'total_searches': total_searches,
            'avg_results_per_search': round(avg_results, 1),
            'avg_search_time_ms': round(avg_time, 1),
            'vector_store_stats': self.vector_store.get_stats(),
            'recent_searches': self._search_history[-5:]  # Last 5 searches
        }
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions based on available influencer data.
        
        Args:
            partial_query: Partial search query
            
        Returns:
            List of suggested queries
        """
        suggestions = []
        
        # Category-based suggestions
        categories = set()
        for influencer in self.vector_store.influencer_data:
            categories.add(influencer.category)
        
        for category in categories:
            if partial_query.lower() in category.lower():
                suggestions.append(f"{category} influencer")
                suggestions.append(f"find {category} creators")
        
        # Bio keyword suggestions
        common_keywords = ['fitness', 'beauty', 'lifestyle', 'food', 'fashion', 'tech', 'travel']
        for keyword in common_keywords:
            if partial_query.lower() in keyword.lower():
                suggestions.append(f"{keyword} content creator")
                suggestions.append(f"{keyword} influencer with high engagement")
        
        return suggestions[:5]  # Return top 5 suggestions

    def _search_comprehensive(self, text_embedding: np.ndarray, request: SearchRequest) -> List[VectorSearchResult]:
        """
        Search using text embedding against bio, profile image, and content image embeddings
        with intelligent weighting to provide the best matches.
        
        Args:
            text_embedding: Text embedding of the search query
            request: Search request with parameters
            
        Returns:
            List of search results with combined scores
        """
        
        all_scores = {}

        # 1. Get results from all three embedding types
        text_results = self.vector_store.search_text(text_embedding, k=request.limit * 3, filters=request.filters)
        profile_results = self.vector_store.search_profile_image(text_embedding, k=request.limit * 3, filters=request.filters)
        content_results = self.vector_store.search_content_image(text_embedding, k=request.limit * 3, filters=request.filters)

        # 2. Populate the scores dictionary
        for result in text_results:
            if result.influencer_id not in all_scores:
                all_scores[result.influencer_id] = {'text_score': 0.0, 'profile_score': 0.0, 'content_score': 0.0, 'data': result}
            all_scores[result.influencer_id]['text_score'] = result.score

        for result in profile_results:
            if result.influencer_id not in all_scores:
                all_scores[result.influencer_id] = {'text_score': 0.0, 'profile_score': 0.0, 'content_score': 0.0, 'data': result}
            all_scores[result.influencer_id]['profile_score'] = result.score

        for result in content_results:
            if result.influencer_id not in all_scores:
                all_scores[result.influencer_id] = {'text_score': 0.0, 'profile_score': 0.0, 'content_score': 0.0, 'data': result}
            all_scores[result.influencer_id]['content_score'] = result.score
    
        # 3. Calculate combined scores and create final list
        combined_results = []
        for influencer_id, scores in all_scores.items():
            combined_score = (
                scores['text_score'] * 0.4 +
                scores['profile_score'] * 0.3 +
                scores['content_score'] * 0.3
            )
        
            result = scores['data']
            result.score = combined_score
            result.metadata = result.metadata or {}
            result.metadata.update({
                'text_score': scores['text_score'],
                'profile_score': scores['profile_score'],
                'content_score': scores['content_score'],
            })
            combined_results.append(result)

        # 4. Sort and return the final results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:request.limit]

# Global search engine instance
search_engine = InfluencerSearchEngine()


def get_search_engine() -> InfluencerSearchEngine:
    """Get the global search engine instance."""
    return search_engine


# CLI interface for testing
def main():
    """CLI interface for testing the search engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test influencer search engine")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--min-followers", type=int, help="Minimum followers")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load index if not already loaded
    if search_engine.vector_store.get_stats()['total_influencers'] == 0:
        print("Loading search index...")
        if not search_engine.vector_store.load():
            print("❌ Failed to load search index. Data may not be initialized.")
            return 1
    
    # Perform search
    print(f"\n🔍 Searching for: '{args.query}'")
    print("=" * 50)
    
    results = search_engine.search_text(
        query=args.query,
        limit=args.limit,
        categories=[args.category] if args.category else None,
        min_followers=args.min_followers
    )
    
    if not results:
        print("❌ No results found")
        return 0
    
    # Display results
    for i, result in enumerate(results, 1):
        inf = result.influencer
        print(f"\n{i}. {inf.name} (Score: {result.similarity_score:.3f})")
        print(f"   Category: {inf.category}")
        print(f"   Followers: {inf.follower_count:,}")
        print(f"   Bio: {inf.bio}")
        print(f"   ID: {inf.influencer_id}")
        if result.match_reasons:
            print(f"   Match: {', '.join(result.match_reasons)}")
    
    # Show stats
    stats = search_engine.get_stats()
    print(f"\n📊 Search completed in {stats.get('avg_search_time_ms', 0):.0f}ms")
    
    return 0


if __name__ == "__main__":
    exit(main())