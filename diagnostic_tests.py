#!/usr/bin/env python3
"""
Diagnostic Tests for Influencer Search Engine
============================================

This script runs comprehensive tests to identify why the search engine
isn't finding fitness influencers despite them being in the database.
"""

import sys
import numpy as np
from typing import List
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def test_1_direct_search_engine():
    """Test 1: Direct search engine without LLM"""
    print("\n" + "="*60)
    print("TEST 1: DIRECT SEARCH ENGINE")
    print("="*60)
    
    try:
        from app.search_engine import get_search_engine
        from app.settings import init_settings
        
        search_engine = get_search_engine()
        print("✅ Search engine initialized")
        
        # Test 'fitness' search
        results = search_engine.search_text('fitness', limit=10)
        print(f"📊 Results for 'fitness': {len(results)} found")
        
        if len(results) > 0:
            print("\n🎯 FOUND RESULTS:")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.influencer.name} (@{r.influencer.username}) - {r.influencer.category}")
                print(f"   Score: {r.similarity_score:.4f}")
                print(f"   Bio: {r.influencer.bio[:100]}...")
                print()
        else:
            print("🚨 NO RESULTS FOUND - This confirms our hypothesis!")
            
        return results
        
    except Exception as e:
        print(f"❌ Error in Test 1: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_2_embedding_similarity():
    """Test 2: Direct embedding similarity calculation"""
    print("\n" + "="*60)
    print("TEST 2: EMBEDDING SIMILARITY")
    print("="*60)
    
    try:
        from app.embedding_service import embedding_service
        
        # Generate embedding for query
        print("🔍 Generating embedding for 'fitness'...")
        fitness_emb = embedding_service.generate_text_embedding("fitness")
        
        if fitness_emb is None:
            print("❌ Failed to generate fitness embedding")
            return
            
        print(f"✅ Fitness embedding shape: {fitness_emb.shape}")
        
        # Test against fitness influencer bios
        fitness_bios = [
            "Realistic health and fitness ♀️♀️@eltmethod coaching⭐️@pescience @runna soheefit",
            "🦋 celebrating all things curly and fit #curlsandfitness DM for any/all Q's or feature!! 🔥🔥 @dladla11"
        ]
        
        print("\n📊 SIMILARITY SCORES:")
        for i, bio in enumerate(fitness_bios, 1):
            bio_emb = embedding_service.generate_text_embedding(bio)
            if bio_emb is not None:
                # Calculate cosine similarity
                similarity = np.dot(fitness_emb, bio_emb) / (np.linalg.norm(fitness_emb) * np.linalg.norm(bio_emb))
                print(f"{i}. Bio: {bio[:50]}...")
                print(f"   Similarity: {similarity:.4f}")
                print()
            else:
                print(f"{i}. Failed to generate embedding for bio")
                
    except Exception as e:
        print(f"❌ Error in Test 2: {e}")
        import traceback
        traceback.print_exc()


def test_3_vector_store_stats():
    """Test 3: Check vector store statistics"""
    print("\n" + "="*60)
    print("TEST 3: VECTOR STORE STATISTICS")
    print("="*60)
    
    try:
        from app.vector_store import vector_store
        
        stats = vector_store.get_stats()
        print("📊 VECTOR STORE STATS:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
        # Check if we have any influencers at all
        total_influencers = stats.get('total_influencers', 0)
        if total_influencers == 0:
            print("🚨 NO INFLUENCERS IN VECTOR STORE!")
            return
            
        # Try to find fitness category influencers manually
        print(f"\n🔍 Searching for fitness category influencers...")
        fitness_count = 0
        
        for i, influencer in enumerate(vector_store.influencer_data):
            if hasattr(influencer, 'category') and influencer.category.lower() == 'fitness':
                fitness_count += 1
                print(f"   Found: {influencer.name} (@{influencer.username}) - {influencer.category}")
                
        print(f"\n📈 Total fitness influencers found in store: {fitness_count}")
        
    except Exception as e:
        print(f"❌ Error in Test 3: {e}")
        import traceback
        traceback.print_exc()


def test_4_alternative_queries():
    """Test 4: Try alternative fitness-related queries"""
    print("\n" + "="*60)
    print("TEST 4: ALTERNATIVE FITNESS QUERIES")
    print("="*60)
    
    try:
        from app.search_engine import get_search_engine
        
        search_engine = get_search_engine()
        
        queries = [
            "workout",
            "health",
            "gym", 
            "exercise",
            "training",
            "sohee",
            "curly fit",
            "fitness coach"
        ]
        
        for query in queries:
            results = search_engine.search_text(query, limit=3)
            print(f"'{query}': {len(results)} results")
            
            for r in results:
                if 'fitness' in r.influencer.category.lower() or 'fitness' in r.influencer.bio.lower():
                    print(f"  ✅ FOUND FITNESS: {r.influencer.name} (score: {r.similarity_score:.4f})")
                    
    except Exception as e:
        print(f"❌ Error in Test 4: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔬 RUNNING DIAGNOSTIC TESTS FOR SEARCH ENGINE")
    print("=" * 80)
    
    # Run all tests
    results = test_1_direct_search_engine()
    test_2_embedding_similarity()
    test_3_vector_store_stats()
    test_4_alternative_queries()
    
    print("\n" + "="*80)
    print("🏁 DIAGNOSTIC TESTS COMPLETED")
    print("="*80) 