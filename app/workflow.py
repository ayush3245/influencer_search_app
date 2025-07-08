from typing import Optional

from app.search_engine import get_search_engine
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.server.api.models import ChatRequest
from llama_index.core.tools import FunctionTool


def influencer_search_tool(query: str, limit: int = 5) -> str:
    """Search for influencers based on text query."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔍 INFLUENCER SEARCH CALLED: '{query}' (limit: {limit})")
    
    search_engine = get_search_engine()
    results = search_engine.search_text(query, limit=limit)
    
    logger.info(f"📊 LOCAL SEARCH RESULTS: {len(results)} influencers found")
    
    if not results:
        return "No influencers found matching your query from our local database."
    
    # Log the top results for debugging
    for i, result in enumerate(results[:3], 1):
        logger.info(f"  {i}. {result.influencer.name} ({result.similarity_score:.1%})")
    
    response_lines = [f"Found {len(results)} influencer(s) from our database:\n"]
    for i, result in enumerate(results, 1):
        inf = result.influencer
        
        # Format comprehensive influencer information
        response_lines.append(f"{i}. **{inf.name}**")
        
        # Username and verification status
        if inf.username:
            verification = " ✅" if inf.is_verified else ""
            privacy = " 🔒" if inf.is_private else ""
            response_lines.append(f"   📱 @{inf.username}{verification}{privacy}")
        
        # Category and bio
        response_lines.append(f"   🏷️ Category: {inf.category.title()}")
        response_lines.append(f"   📝 Bio: {inf.bio}")
        
        # Follower metrics
        if inf.follower_count:
            response_lines.append(f"   👥 Followers: {inf.follower_count:,}")
        if inf.following_count:
            response_lines.append(f"   ➡️ Following: {inf.following_count:,}")
        if inf.post_count:
            response_lines.append(f"   📸 Posts: {inf.post_count:,}")
        
        # Links
        if inf.instagram_url:
            response_lines.append(f"   🔗 Instagram: {inf.instagram_url}")
        if inf.profile_photo_url:
            response_lines.append(f"   🖼️ Profile Photo: {inf.profile_photo_url}")
        
        # Match quality
        response_lines.append(f"   🎯 Match Score: {result.similarity_score:.1%}\n")
    
    return "\n".join(response_lines)


class SimpleInfluencerResponse:
    """Simple response handler for influencer search when no LLM is available."""
    
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


def create_workflow(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    # Create a search tool using our custom search engine
    search_tool = FunctionTool.from_defaults(
        fn=influencer_search_tool,
        name="influencer_search",
        description="Search for influencers based on text queries about their content, style, demographics, or characteristics"
    )

        # Simple, clear system prompt
    system_prompt = """You are a friendly influencer discovery chatbot.

CHAT MODE: For greetings, general questions about social media/marketing, or casual conversation - just chat normally using your knowledge.

SEARCH MODE: When users want influencer recommendations (words like "find", "show me", "recommend", "who are" + influencers) - you MUST use the influencer_search tool.

CRITICAL RULE: NEVER make up influencer names, usernames, or details. If you search, only show the exact results from the tool. If no results, say "No influencers found in our database matching that criteria."

Examples:
- "Hi!" → Chat normally: "Hello! I help find influencers. What are you looking for?"
- "Find fitness influencers" → Use influencer_search tool, show only real results  
- "How to grow Instagram?" → Chat normally with social media advice

Be conversational but always search when they want influencer recommendations!"""

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[search_tool],
        llm=Settings.llm,
        system_prompt=system_prompt,
    )
