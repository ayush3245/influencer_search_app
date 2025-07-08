"""
Excel Data Processor for Influencer Discovery Tool

This module processes the uploaded Excel file containing influencer data
and integrates it with our Instagram data collection and search system.
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from urllib.parse import urlparse

from app.instagram_scraper import InstagramDataCollector
from app.schemas import InfluencerCategory, InfluencerData


class ExcelDataProcessor:
    """Process Excel file and integrate with Instagram data collection."""
    
    def __init__(self, excel_path: str = "data/real_influencers_list.xlsx"):
        """
        Initialize the Excel data processor.
        
        Args:
            excel_path: Path to the Excel file with influencer data
        """
        self.excel_path = Path(excel_path)
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
    def extract_username_from_url(self, instagram_url: str) -> Optional[str]:
        """
        Extract Instagram username from URL.
        
        Args:
            instagram_url: Instagram URL (e.g., "https://www.instagram.com/username/")
            
        Returns:
            Username without @ symbol, or None if extraction fails
        """
        try:
            if not instagram_url or pd.isna(instagram_url):
                return None
                
            # Handle different URL formats
            url = str(instagram_url).strip()
            
            # Remove @ if present at the start
            if url.startswith('@'):
                return url[1:]
            
            # Extract from URL
            if 'instagram.com' in url:
                # Parse URL
                parsed = urlparse(url)
                path = parsed.path.strip('/')
                
                # Handle different path formats
                if path:
                    # Remove any additional path segments
                    username = path.split('/')[0]
                    return username
            
            # If it's just a username
            if url and not '/' in url and not '.' in url:
                return url.replace('@', '')
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract username from URL '{instagram_url}': {e}")
            return None
    
    def map_category(self, category: str) -> InfluencerCategory:
        """
        Map Excel category to our InfluencerCategory enum.
        
        Args:
            category: Category from Excel file
            
        Returns:
            Mapped InfluencerCategory
        """
        if not category or pd.isna(category):
            return InfluencerCategory.LIFESTYLE
            
        category_lower = str(category).lower().strip()
        
        category_mapping = {
            'fitness': InfluencerCategory.FITNESS,
            'beauty': InfluencerCategory.BEAUTY,
            'tech': InfluencerCategory.TECH,
            'technology': InfluencerCategory.TECH,
            'food': InfluencerCategory.FOOD,
            'cooking': InfluencerCategory.FOOD,
            'fashion': InfluencerCategory.FASHION,
            'style': InfluencerCategory.FASHION,
            'travel': InfluencerCategory.TRAVEL,
            'gaming': InfluencerCategory.GAMING,
            'games': InfluencerCategory.GAMING,
            'wellness': InfluencerCategory.WELLNESS,
            'lifestyle': InfluencerCategory.LIFESTYLE,
        }
        
        return category_mapping.get(category_lower, InfluencerCategory.LIFESTYLE)
    
    def load_excel_data(self) -> Optional[pd.DataFrame]:
        """
        Load and validate the Excel file.
        
        Returns:
            DataFrame with Excel data or None if failed
        """
        try:
            if not self.excel_path.exists():
                self.logger.error(f"Excel file not found: {self.excel_path}")
                return None
            
            self.logger.info(f"Loading Excel file: {self.excel_path}")
            df = pd.read_excel(self.excel_path)
            
            self.logger.info(f"Loaded {len(df)} rows from Excel file")
            
            # Validate required columns
            required_columns = ['Name/Handle', 'Category', 'Instagram URL']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load Excel file: {e}")
            return None
    
    def prepare_usernames_for_collection(self, df: pd.DataFrame) -> List[str]:
        """
        Extract and prepare usernames for Instagram data collection.
        
        Args:
            df: DataFrame with Excel data
            
        Returns:
            List of clean usernames ready for collection
        """
        usernames = []
        
        for idx, row in df.iterrows():
            # Try to get username from Instagram URL first
            username = self.extract_username_from_url(row.get('Instagram URL'))
            
            # If that fails, try from Name/Handle
            if not username:
                handle = row.get('Name/Handle', '')
                if handle:
                    username = str(handle).replace('@', '').strip()
            
            if username:
                usernames.append(username)
                self.logger.info(f"Prepared username: {username}")
            else:
                self.logger.warning(f"Could not extract username from row {idx + 1}")
        
        return usernames
    
    def collect_instagram_data(self, usernames: List[str]) -> List[Dict]:
        """
        Collect Instagram data for the extracted usernames.
        
        Args:
            usernames: List of Instagram usernames
            
        Returns:
            List of collected influencer data
        """
        self.logger.info(f"Starting Instagram data collection for {len(usernames)} usernames")
        
        # Initialize Instagram collector with conservative settings
        collector = InstagramDataCollector(
            output_dir="data",
            delay_between_requests=5.0,  # Conservative 5-second delay
            max_retries=2
        )
        
        # Collect data
        collected_data = collector.collect_influencer_data(
            usernames, 
            "real_influencers_instagram_data.csv"
        )
        
        return collected_data
    
    def merge_excel_and_instagram_data(self, 
                                     excel_df: pd.DataFrame, 
                                     instagram_data: List[Dict]) -> List[Dict]:
        """
        Merge Excel data with collected Instagram data.
        
        Args:
            excel_df: Original Excel data
            instagram_data: Collected Instagram data
            
        Returns:
            Merged and enhanced influencer data
        """
        merged_data = []
        
        # Create lookup for Instagram data by username
        instagram_lookup = {}
        for item in instagram_data:
            # Extract username from influencer_id (format: IG_USERNAME)
            username = item['influencer_id'].replace('IG_', '').lower()
            instagram_lookup[username] = item
        
        for idx, row in excel_df.iterrows():
            try:
                # Get username for this row
                username = self.extract_username_from_url(row.get('Instagram URL'))
                if not username:
                    username = str(row.get('Name/Handle', '')).replace('@', '').strip()
                
                if not username:
                    self.logger.warning(f"Skipping row {idx + 1} - no username found")
                    continue
                
                # Look up Instagram data
                instagram_item = instagram_lookup.get(username.lower())
                
                if instagram_item:
                    # Use Instagram data as base and enhance with Excel data
                    merged_item = instagram_item.copy()
                    
                    # Override category with Excel category if available
                    excel_category = self.map_category(row.get('Category'))
                    merged_item['category'] = excel_category.value
                    
                    # Enhance bio with Excel notes if available
                    excel_bio = row.get('Bio/Notes', '')
                    if excel_bio and not pd.isna(excel_bio):
                        original_bio = merged_item.get('bio', '')
                        # Combine bios if both exist
                        if original_bio and excel_bio.lower() not in original_bio.lower():
                            merged_item['bio'] = f"{original_bio} | {excel_bio}"
                        elif not original_bio:
                            merged_item['bio'] = str(excel_bio)
                    
                    merged_data.append(merged_item)
                    self.logger.info(f"✅ Merged data for @{username}")
                    
                else:
                    # Create fallback data from Excel only
                    self.logger.warning(f"⚠️  No Instagram data for @{username}, creating fallback")
                    
                    fallback_item = {
                        'influencer_id': f"IG_{username.upper()}",
                        'name': str(row.get('Name/Handle', username)).replace('@', ''),
                        'bio': str(row.get('Bio/Notes', f"Instagram influencer @{username}")),
                        'category': self.map_category(row.get('Category')).value,
                        'follower_count': 0,  # Unknown
                        'profile_photo_url': f"https://www.instagram.com/{username}/",
                        'content_thumbnail_url': f"https://www.instagram.com/{username}/"
                    }
                    merged_data.append(fallback_item)
                    
            except Exception as e:
                self.logger.error(f"Error processing row {idx + 1}: {e}")
                continue
        
        self.logger.info(f"Successfully merged {len(merged_data)} influencer records")
        return merged_data
    
    def save_merged_data(self, merged_data: List[Dict], output_file: str = "merged_influencers.csv"):
        """
        Save merged data to CSV file.
        
        Args:
            merged_data: List of merged influencer data
            output_file: Output filename
        """
        if not merged_data:
            self.logger.warning("No data to save")
            return
            
        try:
            # Convert to DataFrame and save
            df = pd.DataFrame(merged_data)
            output_path = Path("data") / output_file
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Saved merged data to: {output_path}")
            
            # Show summary
            category_counts = df['category'].value_counts()
            self.logger.info("Category distribution:")
            for category, count in category_counts.items():
                self.logger.info(f"  {category}: {count} influencers")
                
        except Exception as e:
            self.logger.error(f"Failed to save merged data: {e}")
    
    def validate_merged_data(self, merged_data: List[Dict]) -> List[Dict]:
        """
        Validate merged data against our schema.
        
        Args:
            merged_data: List of merged influencer data
            
        Returns:
            List of validated data (invalid entries removed)
        """
        validated_data = []
        
        for idx, item in enumerate(merged_data):
            try:
                # Validate with Pydantic
                influencer = InfluencerData(**item)
                validated_data.append(item)
                
            except Exception as e:
                self.logger.error(f"Validation failed for item {idx + 1}: {e}")
                self.logger.error(f"Item data: {item}")
                continue
        
        self.logger.info(f"Validated {len(validated_data)}/{len(merged_data)} records")
        return validated_data
    
    def process_excel_file(self, 
                          collect_instagram_data: bool = True,
                          build_search_index: bool = True) -> Optional[List[Dict]]:
        """
        Complete processing pipeline for Excel file.
        
        Args:
            collect_instagram_data: Whether to collect data from Instagram
            build_search_index: Whether to build search index after processing
            
        Returns:
            List of processed influencer data or None if failed
        """
        try:
            # Step 1: Load Excel data
            self.logger.info("🚀 Starting Excel data processing pipeline")
            excel_df = self.load_excel_data()
            if excel_df is None:
                return None
            
            # Step 2: Prepare usernames
            usernames = self.prepare_usernames_for_collection(excel_df)
            if not usernames:
                self.logger.error("No valid usernames found")
                return None
            
            self.logger.info(f"Found {len(usernames)} usernames to process")
            
            # Step 3: Collect Instagram data (optional)
            instagram_data = []
            if collect_instagram_data:
                self.logger.info("📱 Collecting Instagram data...")
                
                # Show legal warning
                print("\n" + "="*60)
                print("⚠️  LEGAL NOTICE FOR INSTAGRAM DATA COLLECTION")
                print("="*60)
                print("This will collect data from Instagram using instaloader.")
                print("Please ensure you:")
                print("- Have permission to collect this data")
                print("- Are complying with Instagram's Terms of Service")
                print("- Are following applicable privacy laws")
                print("- Are using this data responsibly")
                
                confirm = input("\nDo you want to proceed with Instagram data collection? (yes/no): ").lower().strip()
                if confirm == 'yes':
                    instagram_data = self.collect_instagram_data(usernames)
                else:
                    self.logger.info("Skipping Instagram data collection")
                    collect_instagram_data = False
            
            # Step 4: Merge data
            if collect_instagram_data and instagram_data:
                self.logger.info("🔗 Merging Excel and Instagram data...")
                merged_data = self.merge_excel_and_instagram_data(excel_df, instagram_data)
            else:
                self.logger.info("📋 Using Excel data only...")
                # Create data from Excel only
                merged_data = []
                for idx, row in excel_df.iterrows():
                    username = self.extract_username_from_url(row.get('Instagram URL'))
                    if not username:
                        username = str(row.get('Name/Handle', '')).replace('@', '').strip()
                    
                    if username:
                        item = {
                            'influencer_id': f"EXL_{username.upper()}",
                            'name': str(row.get('Name/Handle', username)).replace('@', ''),
                            'bio': str(row.get('Bio/Notes', f"Influencer @{username}")),
                            'category': self.map_category(row.get('Category')).value,
                            'follower_count': 100000,  # Default estimate
                            'profile_photo_url': f"https://www.instagram.com/{username}/",
                            'content_thumbnail_url': f"https://www.instagram.com/{username}/"
                        }
                        merged_data.append(item)
            
            # Step 5: Validate data
            self.logger.info("✅ Validating merged data...")
            validated_data = self.validate_merged_data(merged_data)
            
            if not validated_data:
                self.logger.error("No valid data after validation")
                return None
            
            # Step 6: Save data
            output_filename = "real_influencers_processed.csv"
            self.save_merged_data(validated_data, output_filename)
            
            # Step 7: Build search index (optional)
            if build_search_index:
                self.logger.info("🔍 Building search index...")
                try:
                    # Update the main data file
                    import shutil
                    shutil.copy(
                        Path("data") / output_filename,
                        Path("data") / "sample_influencers.csv"
                    )
                    
                    # Build index (import locally to avoid circular dependency)
                    from app.build_index import main as build_index
                    build_index()
                    self.logger.info("✅ Search index built successfully")
                    
                except Exception as e:
                    self.logger.error(f"Failed to build search index: {e}")
            
            self.logger.info("🎉 Excel processing pipeline completed successfully!")
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return None


def main():
    """Main function to run the Excel data processing."""
    
    print("📊 Excel Data Processor for Influencer Discovery Tool")
    print("=" * 60)
    
    processor = ExcelDataProcessor()
    
    # Process the Excel file
    result = processor.process_excel_file(
        collect_instagram_data=True,  # Set to False to skip Instagram collection
        build_search_index=True
    )
    
    if result:
        print(f"\n✅ Successfully processed {len(result)} influencers!")
        print("💡 You can now test the search system with your real data:")
        print("   python main.py")
    else:
        print("\n❌ Processing failed. Check the logs above for details.")


if __name__ == "__main__":
    main() 