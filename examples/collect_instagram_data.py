"""
Example script for collecting Instagram influencer data responsibly.

This script demonstrates how to use the InstagramDataCollector class
to gather real influencer data while respecting Instagram's ToS and
rate limits.
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.instagram_scraper import InstagramDataCollector
import json


def collect_sample_influencers():
    """
    Collect data for a curated list of public influencers.
    
    ⚠️  IMPORTANT: Only use this with influencers who have given explicit 
    permission or public figures where data collection is clearly permitted.
    """
    
    # Example list of public influencer usernames
    # These are examples - replace with actual usernames you have permission to collect
    influencer_usernames = {
        "fitness": [
            "soheefit",
            "curlsandfitness",
        ],
        "beauty": [
            "linnygd", 
            "thatcurlytop",
            "freddieharrel",
        ],
        "tech": [
            "bentaylortech",
            "tiffintech",
            "pewdiepie", 
        ],
        "food": [
            "christieathome",
            "nom_life",
        ],
        "lifestyle": [
            "scandinavianstylist",
            "theredhairtraveler",
            "jonathankmonroe",
            "kacierose_",
            "cassyeungmoney",
        ]
    }
    
    # Initialize collector with conservative settings
    collector = InstagramDataCollector(
        output_dir="data",
        delay_between_requests=5.0,  # 5 seconds between requests
        max_retries=2
    )
    
    all_collected_data = []
    
    for category, usernames in influencer_usernames.items():
        if not usernames:  # Skip empty categories
            print(f"⏭️  Skipping {category} - no usernames provided")
            continue
            
        print(f"\n📂 Collecting {category} influencers...")
        
        # Collect data for this category
        category_data = collector.collect_influencer_data(
            usernames, 
            f"{category}_influencers.csv"
        )
        
        all_collected_data.extend(category_data)
        
        print(f"✅ Collected {len(category_data)} {category} influencers")
    
    # Save combined data
    if all_collected_data:
        collector.save_to_csv(all_collected_data, Path("data/all_influencers.csv"))
        print(f"\n🎉 Total collected: {len(all_collected_data)} influencers")
        print(f"📁 Saved to: data/all_influencers.csv")
        
        # Display summary
        categories = {}
        for data in all_collected_data:
            cat = data['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n📊 Collection Summary by Category:")
        for category, count in categories.items():
            print(f"   {category}: {count} influencers")
    else:
        print("❌ No data collected. Please add valid usernames to the script.")


def search_and_collect_by_hashtag():
    """
    Example of finding and collecting influencers via hashtag search.
    
    ⚠️  This method is more intrusive and should be used very carefully.
    """
    
    collector = InstagramDataCollector(
        output_dir="data",
        delay_between_requests=10.0,  # Even longer delays for hashtag search
        max_retries=1
    )
    
    # Search for fitness influencers via hashtag
    hashtags_to_search = [
        # "fitness",
        # "healthylifestyle", 
        # "fitspo"
    ]
    
    all_found_usernames = []
    
    for hashtag in hashtags_to_search:
        print(f"🔍 Searching #{hashtag}...")
        
        # Search hashtag for potential influencers
        found_usernames = collector.search_by_hashtag(
            hashtag, 
            max_posts=20,  # Limit to reduce load
            min_followers=50000
        )
        
        all_found_usernames.extend(found_usernames)
        print(f"   Found {len(found_usernames)} potential influencers")
    
    # Remove duplicates
    unique_usernames = list(set(all_found_usernames))
    
    if unique_usernames:
        print(f"\n📝 Found {len(unique_usernames)} unique influencers")
        print("⚠️  Please review this list and only collect data for public figures")
        print("   or those who have given explicit permission:")
        
        for username in unique_usernames[:10]:  # Show first 10
            print(f"   @{username}")
        
        # Save usernames for manual review
        with open("data/discovered_usernames.json", "w") as f:
            json.dump(unique_usernames, f, indent=2)
        
        print(f"\n💾 Saved usernames to: data/discovered_usernames.json")
        print("👀 Please review these usernames before collecting their data")
    else:
        print("❌ No influencers found matching the criteria")


def load_usernames_from_file():
    """
    Load usernames from a JSON file for batch processing.
    """
    try:
        with open("data/usernames_to_collect.json", "r") as f:
            usernames = json.load(f)
        
        collector = InstagramDataCollector()
        
        if isinstance(usernames, dict):
            # Handle category-based structure
            for category, names in usernames.items():
                if names:
                    collector.collect_influencer_data(names, f"{category}_batch.csv")
        else:
            # Handle simple list
            collector.collect_influencer_data(usernames, "batch_collection.csv")
            
    except FileNotFoundError:
        print("❌ File 'data/usernames_to_collect.json' not found")
        print("💡 Create this file with usernames you want to collect:")
        print("""
        Example format:
        {
          "fitness": ["username1", "username2"],
          "beauty": ["username3", "username4"]
        }
        or simple list:
        ["username1", "username2", "username3"]
        """)


def create_example_config():
    """Create an example configuration file for usernames."""
    
    example_config = {
        "fitness": [
            "# Add fitness influencer usernames here",
            "# Example: 'fitness_influencer_name'"
        ],
        "beauty": [
            "# Add beauty influencer usernames here"
        ],
        "tech": [
            "# Add tech influencer usernames here"
        ],
        "food": [
            "# Add food influencer usernames here"
        ],
        "lifestyle": [
            "# Add lifestyle influencer usernames here"
        ]
    }
    
    config_path = Path("data/usernames_template.json")
    with open(config_path, "w") as f:
        json.dump(example_config, f, indent=2)
    
    print(f"📝 Created template file: {config_path}")
    print("💡 Edit this file with real usernames, then rename to 'usernames_to_collect.json'")


def main():
    """Main function with menu for different collection methods."""
    
    print("🤖 Instagram Influencer Data Collector")
    print("=" * 50)
    
    print("\n⚠️  LEGAL NOTICE:")
    print("- Only collect data from public profiles")
    print("- Respect Instagram's Terms of Service")
    print("- Get explicit permission when possible")
    print("- Use reasonable rate limits")
    print("- Be mindful of privacy laws")
    
    print("\nAvailable options:")
    print("1. Collect data from predefined usernames")
    print("2. Search hashtags for influencers (advanced)")
    print("3. Load usernames from JSON file")
    print("4. Create example configuration file")
    print("5. Exit")
    
    while True:
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == "1":
            collect_sample_influencers()
            break
        elif choice == "2":
            confirmation = input("⚠️  Hashtag search is more intrusive. Continue? (yes/no): ")
            if confirmation.lower() == "yes":
                search_and_collect_by_hashtag()
            break
        elif choice == "3":
            load_usernames_from_file()
            break
        elif choice == "4":
            create_example_config()
            break
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main() 