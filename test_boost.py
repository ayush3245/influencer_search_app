#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app.search_engine import get_search_engine

print("🧪 TESTING CATEGORY BOOST")
print("=" * 50)

search_engine = get_search_engine()
results = search_engine.search_text('fitness', limit=10)

print(f"Total results: {len(results)}\n")

fitness_found = []
non_fitness_found = []

for i, r in enumerate(results, 1):
    is_fitness = r.influencer.category.lower() == 'fitness'
    
    print(f"{i}. {r.influencer.name} (@{r.influencer.username})")
    print(f"   Category: {r.influencer.category} {'✅' if is_fitness else '❌'}")
    print(f"   Score: {r.similarity_score:.6f}")
    
    # Check for boost reasons
    boost_found = False
    for reason in r.match_reasons:
        if 'boost' in reason.lower() or 'match' in reason.lower():
            print(f"   🎯 {reason}")
            boost_found = True
    
    if is_fitness and not boost_found:
        print("   ⚠️  NO BOOST APPLIED!")
    
    if is_fitness:
        fitness_found.append((i, r.influencer.name, r.similarity_score))
    else:
        non_fitness_found.append((i, r.influencer.name, r.similarity_score))
    print()

print("=" * 50)
print(f"✅ FITNESS IN RESULTS: {len(fitness_found)}")
for rank, name, score in fitness_found:
    print(f"   #{rank}: {name} - {score:.6f}")

print(f"❌ FITNESS IN TOP 3: {len([x for x in fitness_found if x[0] <= 3])}")