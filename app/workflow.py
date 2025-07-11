from typing import Optional
from app.search_engine import get_search_engine


class SimpleInfluencerResponse:
    """Simple response handler for influencer search without LLM."""
    
    def __init__(self):
        self.search_engine = get_search_engine()
    
    def process_query(self, query: str) -> str:
        """Process a user query and return search results."""
        # Extract search intent and parameters
        limit = 5
        
        # Check for limit keywords
        if "top 10" in query.lower() or "10 influencers" in query.lower():
            limit = 10
        elif "top 3" in query.lower() or "3 influencers" in query.lower():
            limit = 3
        
        # Perform search
        results = self.search_engine.search_text(query, limit=limit)
        
        if not results:
            return "I couldn't find any influencers matching your search criteria. Try refining your search with different keywords like 'fitness', 'beauty', 'tech', or describe the type of content or style you're looking for."
        
        # Format enhanced response with complete data
        response_lines = [f"I found {len(results)} influencer(s) matching your search:\n"]
        for i, result in enumerate(results, 1):
            inf = result.influencer
            
            # Header with name and verification
            verification = " ✅" if inf.is_verified else ""
            privacy = " 🔒" if inf.is_private else ""
            username_display = f" (@{inf.username})" if inf.username else ""
            response_lines.append(f"**{i}. {inf.name}{username_display}{verification}{privacy}**")
            
            # Core information
            response_lines.append(f"🏷️ **Category:** {inf.category.title()}")
            response_lines.append(f"📝 **Bio:** {inf.bio}")
            
            # Metrics section
            metrics = []
            if inf.follower_count:
                metrics.append(f"👥 {inf.follower_count:,} followers")
            if inf.following_count:
                metrics.append(f"➡️ {inf.following_count:,} following")
            if inf.post_count:
                metrics.append(f"📸 {inf.post_count:,} posts")
            
            if metrics:
                response_lines.append(f"📊 **Stats:** {' | '.join(metrics)}")
            
            # Links and media
            if inf.instagram_url:
                response_lines.append(f"🔗 **Instagram:** {inf.instagram_url}")
            
            # Match quality
            response_lines.append(f"🎯 **Match Score:** {result.similarity_score:.1%}")
            response_lines.append("")  # Empty line for spacing
        
        response_lines.append("Would you like me to search for something else or refine these results?")
        return "\n".join(response_lines)
