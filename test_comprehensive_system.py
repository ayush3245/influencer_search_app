#!/usr/bin/env python3
"""
Comprehensive system test for the Influencer Discovery Tool.

This script tests all major components of the system:
- Data loading and processing
- Embedding generation
- Vector store operations
- Search functionality
- Response handling
"""

import logging
import sys
import asyncio
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.data_loader import load_sample_data
from app.embedding_service import get_embedding_service, test_embedding_service
from app.vector_store import vector_store
from app.search_engine import get_search_engine
from app.workflow import SimpleInfluencerResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing Data Loading...")
    
    try:
        # Load influencer data
        influencers = load_sample_data()
        
        if not influencers:
            print("❌ No influencers loaded")
            return False
        
        print(f"✅ Loaded {len(influencers)} influencers")
        
        # Check data quality
        for i, influencer in enumerate(influencers[:3]):
            print(f"  {i+1}. {influencer.name} ({influencer.category})")
            print(f"     Bio: {influencer.bio[:50]}...")
            print(f"     Followers: {influencer.follower_count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def test_embedding_service():
    """Test embedding service functionality."""
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


def test_vector_store():
    """Test vector store functionality."""
    print("\n🗄️  Testing Vector Store...")
    
    try:
        # Check if vector store is loaded
        stats = vector_store.get_stats()
        print(f"Vector store stats: {stats}")
        
        if stats['total_influencers'] == 0:
            print("❌ No influencers in vector store")
            return False
        
        # Test search functionality
        search_engine = get_search_engine()
        results = search_engine.search_text("fitness", limit=3)
        
        if results:
            print(f"✅ Search returned {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                print(f"  {i}. {result.influencer.name} (Score: {result.similarity_score:.3f})")
        else:
            print("❌ Search returned no results")
            return False
        
        print("✅ Vector store tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_search_engine():
    """Test search engine functionality."""
    print("\n🔍 Testing Search Engine...")
    
    try:
        search_engine = get_search_engine()
        
        # Test different search queries
        test_queries = [
            "fitness influencers",
            "beauty content",
            "lifestyle bloggers"
        ]
        
        for query in test_queries:
            results = search_engine.search_text(query, limit=3)
            print(f"✅ Query '{query}': {len(results)} results")
            
            if results:
                for i, result in enumerate(results[:2], 1):
                    print(f"    {i}. {result.influencer.name} ({result.similarity_score:.3f})")
        
        print("✅ Search engine tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Search engine test failed: {e}")
        return False


def test_simple_response():
    """Test the simple response handler."""
    print("\n⚙️  Testing Simple Response Handler...")
    
    try:
        response_handler = SimpleInfluencerResponse()
        result = response_handler.process_query("fitness influencers")
        print(f"✅ Simple response handler processed query")
        print(f"📝 Response length: {len(result)} characters")
        return True
    except Exception as e:
        print(f"❌ Simple response handler failed: {e}")
        return False


async def run_all_tests():
    """Run all system tests."""
    print("🚀 Starting Comprehensive System Tests...")
    print("=" * 60)
    
    # Test registry
    test_registry = [
        ("Data Loading", test_data_loading),
        ("Embedding Service", test_embedding_service),
        ("Vector Store", test_vector_store),
        ("Search Engine", test_search_engine),
        ("Simple Response Handler", test_simple_response),
    ]
    
    # Run tests
    passed = 0
    total = len(test_registry)
    
    for test_name, test_func in test_registry:
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                passed += 1
            else:
                print(f"❌ {test_name} failed")
                
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            logger.error(f"Test {test_name} failed", exc_info=True)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        print("✅ Data loading is functional")
        print("✅ Embedding service is operational")
        print("✅ Vector store is working")
        print("✅ Search engine is responsive")
        print("✅ Response handler is ready")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


def main():
    """Main test runner."""
    try:
        success = asyncio.run(run_all_tests())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        logger.error("Test runner failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 