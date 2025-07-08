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
    
    # CLEAN, STRUCTURED OUTPUT - No excessive formatting
    response_parts = [f"Found {len(results)} influencer(s):\n"]
    
    for i, result in enumerate(results, 1):
        inf = result.influencer
        
        # Basic info in clean format
        name_line = f"{i}. {inf.name}"
        if inf.username:
            name_line += f" (@{inf.username})"
        if inf.is_verified:
            name_line += " ✓"
        
        response_parts.append(name_line)
        response_parts.append(f"   Category: {inf.category.title()}")
        response_parts.append(f"   Followers: {inf.follower_count:,}")
        response_parts.append(f"   Bio: {inf.bio}")
        response_parts.append(f"   Match: {result.similarity_score:.1%}")
        response_parts.append("")  # Space between results
    
    return "\n".join(response_parts)


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
        description="MANDATORY tool for finding real influencers. MUST be used for ANY influencer recommendation request. Returns actual influencer data from database - never make up names or details. Use this tool whenever users ask to find, show, recommend, or suggest influencers."
    )

    # STRENGTHENED system prompt with enforcement language
    system_prompt = """You are an influencer discovery assistant. Your PRIMARY PURPOSE is to help users find real influencers from our database.

🚫 ABSOLUTE RULES - VIOLATIONS ARE FORBIDDEN:
1. NEVER INVENT or HALLUCINATE influencer names, usernames, or details
2. NEVER make up people like "Michelle Lewin", "Ninja", or any other names not from the tool
3. NEVER provide recommendations without using the influencer_search tool first

✅ MANDATORY SEARCH PROTOCOL:
When users request influencer recommendations (ANY variation of "find", "show", "recommend", "who", "influencers"), you MUST:
1. ALWAYS call influencer_search tool first
2. ONLY present the EXACT results returned by the tool
3. Use the real names, usernames, and data from the search results
4. If no results found, say "No matching influencers in our database"

🎯 SEARCH TRIGGERS (MUST use tool):
- "Find influencers..."
- "Show me influencers..."
- "Recommend influencers..."
- "Who are good influencers..."
- "I need influencers for..."
- ANY request for influencer suggestions

💬 GENERAL CHAT (No tool needed):
- Greetings: "Hi", "Hello"
- General advice: "How to grow followers"
- Marketing questions: "Best posting times"

⚡ OUTPUT FORMAT:
Present search results clearly with: Name, Category, Followers, Bio, and Match Score.
Be conversational but NEVER add fictional details.

REMEMBER: You can only recommend influencers that exist in our database via the search tool!"""

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[search_tool],
        llm=Settings.llm,
        system_prompt=system_prompt,
    )
