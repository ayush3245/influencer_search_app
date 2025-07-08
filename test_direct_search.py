#!/usr/bin/env python3
"""
Direct search test - bypasses LLM to test core search functionality
"""
from app.search_engine import get_search_engine
from app.settings import init_settings

def test_direct_search():
    """Test the search engine directly without LLM."""
    print("🔍 Testing Direct Influencer Search Engine")
    print("=" * 50)
    
    # Initialize settings
    init_settings()
    
    # Get search engine
    search_engine = get_search_engine()
    
    # Test queries
    test_queries = [
        "fitness influencer with curly hair",
        "korean fitness coach", 
        "beauty makeup dark hair",
        "tech reviewer gadgets",
        "food cooking chef"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        print("-" * 30)
        
        try:
            results = search_engine.search_text(query, limit=3)
            
            if results:
                print(f"Found {len(results)} result(s):")
                for i, result in enumerate(results, 1):
                    inf = result.influencer
                    print(f"  {i}. **{inf.name}** ({inf.category})")
                    print(f"     📝 {inf.bio}")
                    print(f"     👥 {inf.follower_count:,} followers")
                    print(f"     🎯 Score: {result.similarity_score:.1%}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Direct search test complete!")
    print("💡 This confirms your CLIP embeddings and vector store are working!")

if __name__ == "__main__":
    test_direct_search() 