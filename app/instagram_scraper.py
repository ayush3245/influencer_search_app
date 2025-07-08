"""
Instagram Data Collection Script for Influencer Discovery Tool

⚠️  IMPORTANT LEGAL NOTICE:
- This script is for educational/research purposes only
- Instagram's Terms of Service prohibit automated data collection
- Only collect publicly available data
- Respect rate limits to avoid being blocked
- Consider getting explicit permission from influencers
- Be mindful of privacy and data protection laws (GDPR, CCPA, etc.)
- Use responsibly and at your own risk

This script uses instaloader to collect influencer data and maps it to our schema.
"""

import instaloader
import csv
import os
import time
import requests
import shutil
from typing import List, Dict, Optional, Set
from pathlib import Path
import re
from datetime import datetime
import json
import logging
from urllib.parse import urlparse

from app.schemas import InfluencerCategory


class InstagramDataCollector:
    """Responsible Instagram data collection using instaloader."""
    
    def __init__(self, 
                 output_dir: str = "data",
                 delay_between_requests: float = 2.0,
                 max_retries: int = 3):
        """
        Initialize the Instagram data collector.
        
        Args:
            output_dir: Directory to save collected data
            delay_between_requests: Delay between requests (seconds)
            max_retries: Maximum number of retries for failed requests
        """
        self.output_dir = Path(output_dir)
        self.delay = delay_between_requests
        self.max_retries = max_retries
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images" / "profiles").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "content").mkdir(parents=True, exist_ok=True)
        
        # Initialize instaloader
        self.loader = instaloader.Instaloader(
            download_pictures=False,  # We'll handle images manually
            download_videos=False,
            download_video_thumbnails=False,
            save_metadata=False,
            compress_json=False,
            post_metadata_txt_pattern="",
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Category keywords for automatic categorization
        self.category_keywords = {
            InfluencerCategory.FITNESS: [
                'fitness', 'gym', 'workout', 'training', 'bodybuilding', 'crossfit',
                'yoga', 'pilates', 'running', 'marathon', 'health', 'nutrition',
                'personal trainer', 'coach', 'athlete', 'sports'
            ],
            InfluencerCategory.BEAUTY: [
                'beauty', 'makeup', 'cosmetics', 'skincare', 'hair', 'nails',
                'beauty guru', 'mua', 'makeup artist', 'skincare routine',
                'beauty tips', 'cosmetic', 'glam', 'beauty blogger'
            ],
            InfluencerCategory.TECH: [
                'tech', 'technology', 'gadgets', 'software', 'ai', 'ml',
                'programming', 'coding', 'developer', 'startup', 'innovation',
                'digital', 'cyber', 'data science', 'web dev'
            ],
            InfluencerCategory.LIFESTYLE: [
                'lifestyle', 'life', 'daily', 'routine', 'home', 'decor',
                'minimalist', 'aesthetic', 'vlog', 'blogger', 'content creator',
                'influencer', 'inspiration', 'motivation'
            ],
            InfluencerCategory.FOOD: [
                'food', 'cooking', 'recipe', 'chef', 'kitchen', 'restaurant',
                'foodie', 'cuisine', 'baking', 'culinary', 'meal', 'dinner',
                'lunch', 'breakfast', 'food blogger'
            ],
            InfluencerCategory.FASHION: [
                'fashion', 'style', 'outfit', 'clothes', 'designer', 'model',
                'fashionista', 'stylist', 'brand', 'trends', 'clothing',
                'accessories', 'fashion blogger', 'ootd'
            ],
            InfluencerCategory.TRAVEL: [
                'travel', 'wanderlust', 'adventure', 'explore', 'journey',
                'vacation', 'trip', 'destination', 'tourism', 'backpacker',
                'nomad', 'travel blogger', 'traveler'
            ],
            InfluencerCategory.GAMING: [
                'gaming', 'gamer', 'esports', 'twitch', 'stream', 'streamer',
                'games', 'console', 'pc gaming', 'mobile gaming', 'fps',
                'mmorpg', 'indie games'
            ],
            InfluencerCategory.WELLNESS: [
                'wellness', 'mindfulness', 'meditation', 'mental health',
                'self care', 'therapy', 'healing', 'spiritual', 'mindset',
                'positive', 'motivational', 'life coach'
            ]
        }
    
    def categorize_influencer(self, bio: str, username: str) -> InfluencerCategory:
        """
        Automatically categorize an influencer based on their bio and username.
        
        Args:
            bio: The influencer's bio text
            username: The influencer's username
            
        Returns:
            The predicted category
        """
        text = f"{bio} {username}".lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        # Return category with highest score, default to lifestyle
        if category_scores and max(category_scores.values()) > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return InfluencerCategory.LIFESTYLE
    
    def download_image(self, url: str, filename: str, subfolder: str = "profiles") -> Optional[str]:
        """
        Download an image from URL.
        
        Args:
            url: Image URL
            filename: Local filename
            subfolder: Subfolder (profiles or content)
            
        Returns:
            Local file path or None if failed
        """
        try:
            target_dir = self.output_dir / "images" / subfolder
            target_path = target_dir / filename
            
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            # Return relative URL for our system
            return f"data/images/{subfolder}/{filename}"
            
        except Exception as e:
            self.logger.error(f"Failed to download image {url}: {e}")
            return None
    
    def get_profile_data(self, username: str) -> Optional[Dict]:
        """
        Get profile data for a single Instagram user.
        
        Args:
            username: Instagram username (without @)
            
        Returns:
            Dictionary with profile data or None if failed
        """
        try:
            self.logger.info(f"Fetching profile data for @{username}")
            
            # Get profile
            profile = instaloader.Profile.from_username(self.loader.context, username)
            
            # Get the first post for content thumbnail
            posts = profile.get_posts()
            first_post = next(iter(posts), None)
            
            # Download profile picture
            profile_pic_filename = f"{username}_profile.jpg"
            profile_pic_path = self.download_image(
                profile.profile_pic_url, 
                profile_pic_filename, 
                "profiles"
            )
            
            # Download content thumbnail if available
            content_thumbnail_path = None
            if first_post:
                content_filename = f"{username}_content.jpg"
                content_thumbnail_path = self.download_image(
                    first_post.url, 
                    content_filename, 
                    "content"
                )
            
            # Create influencer data
            influencer_data = {
                'influencer_id': f"IG_{username.upper()}",
                'name': profile.full_name or username,
                'bio': profile.biography or f"Instagram influencer @{username}",
                'category': self.categorize_influencer(
                    profile.biography or "", 
                    username
                ).value,
                'follower_count': profile.followers,
                'profile_photo_url': profile_pic_path or profile.profile_pic_url,
                'content_thumbnail_url': content_thumbnail_path or (first_post.url if first_post else profile.profile_pic_url)
            }
            
            self.logger.info(f"Successfully collected data for @{username}")
            return influencer_data
            
        except Exception as e:
            self.logger.error(f"Failed to get profile data for @{username}: {e}")
            return None
    
    def collect_influencer_data(self, 
                              usernames: List[str], 
                              output_file: str = "influencers.csv") -> List[Dict]:
        """
        Collect data for multiple influencers.
        
        Args:
            usernames: List of Instagram usernames
            output_file: Output CSV filename
            
        Returns:
            List of collected influencer data
        """
        collected_data = []
        failed_usernames = []
        
        self.logger.info(f"Starting data collection for {len(usernames)} influencers")
        
        for i, username in enumerate(usernames, 1):
            self.logger.info(f"Processing {i}/{len(usernames)}: @{username}")
            
            # Attempt to collect data
            data = None
            for attempt in range(self.max_retries):
                try:
                    data = self.get_profile_data(username)
                    if data:
                        break
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for @{username}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.delay * (attempt + 1))  # Exponential backoff
            
            if data:
                collected_data.append(data)
                self.logger.info(f"✅ Successfully collected data for @{username}")
            else:
                failed_usernames.append(username)
                self.logger.error(f"❌ Failed to collect data for @{username}")
            
            # Rate limiting
            if i < len(usernames):  # Don't delay after the last request
                time.sleep(self.delay)
        
        # Save to CSV
        if collected_data:
            output_path = self.output_dir / output_file
            self.save_to_csv(collected_data, output_path)
            self.logger.info(f"Saved {len(collected_data)} influencer records to {output_path}")
        
        # Log summary
        self.logger.info(f"\n📊 Collection Summary:")
        self.logger.info(f"   ✅ Successful: {len(collected_data)}")
        self.logger.info(f"   ❌ Failed: {len(failed_usernames)}")
        if failed_usernames:
            self.logger.info(f"   Failed usernames: {', '.join(failed_usernames)}")
        
        return collected_data
    
    def save_to_csv(self, data: List[Dict], output_path: Path):
        """Save collected data to CSV file."""
        if not data:
            return
        
        fieldnames = [
            'influencer_id', 'name', 'bio', 'category', 
            'follower_count', 'profile_photo_url', 'content_thumbnail_url'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def search_by_hashtag(self, 
                         hashtag: str, 
                         max_posts: int = 50,
                         min_followers: int = 10000) -> List[str]:
        """
        Find influencers by searching a hashtag.
        
        Args:
            hashtag: Hashtag to search (without #)
            max_posts: Maximum number of posts to analyze
            min_followers: Minimum follower count
            
        Returns:
            List of usernames meeting criteria
        """
        try:
            self.logger.info(f"Searching hashtag #{hashtag} for influencers")
            
            hashtag_obj = instaloader.Hashtag.from_name(self.loader.context, hashtag)
            usernames = set()
            
            posts = hashtag_obj.get_posts()
            for i, post in enumerate(posts):
                if i >= max_posts:
                    break
                
                profile = post.owner_profile
                if profile.followers >= min_followers:
                    usernames.add(profile.username)
                
                # Rate limiting
                time.sleep(self.delay)
            
            username_list = list(usernames)
            self.logger.info(f"Found {len(username_list)} potential influencers from #{hashtag}")
            return username_list
            
        except Exception as e:
            self.logger.error(f"Failed to search hashtag #{hashtag}: {e}")
            return []


def main():
    """Example usage of the Instagram data collector."""
    
    print("🚨 LEGAL WARNING:")
    print("This script is for educational/research purposes only.")
    print("Instagram's Terms of Service prohibit automated data collection.")
    print("Use responsibly and at your own risk.\n")
    
    # Get user confirmation
    confirmation = input("Do you understand and accept the risks? (yes/no): ").lower()
    if confirmation != 'yes':
        print("Operation cancelled.")
        return
    
    # Initialize collector
    collector = InstagramDataCollector(
        output_dir="data",
        delay_between_requests=3.0,  # 3 seconds between requests
        max_retries=3
    )
    
    # Example: Collect data for specific usernames
    example_usernames = [
        "fitness",
        "beauty",
        "tech",
        "food",
        "fashion"
    ]
    
    print(f"\n📝 This is an example that would collect data for: {example_usernames}")
    print("🚫 We're NOT running it automatically to respect Instagram's ToS")
    print("\n💡 To use this script:")
    print("1. Modify the usernames list with real influencer accounts")
    print("2. Ensure you have permission to collect their data")
    print("3. Run: collector.collect_influencer_data(usernames)")
    
    # Example of searching by hashtag (commented out)
    # fitness_influencers = collector.search_by_hashtag("fitness", max_posts=20, min_followers=50000)
    # collector.collect_influencer_data(fitness_influencers, "fitness_influencers.csv")


if __name__ == "__main__":
    main() 