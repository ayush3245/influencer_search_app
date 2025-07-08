"""
Standalone script to process Instagram influencer data.

This script loads the Instagram CSV file, generates embeddings for text and images,
and stores them in the vector database. It can be run independently of the main application.

Usage:
    python process_instagram_data.py [--csv-path PATH] [--force] [--batch-size SIZE]
"""

import argparse
import logging
import sys
from pathlib import Path
import time

from app.instagram_data_processor import process_instagram_data_file, InstagramDataProcessor
from app.data_init import get_data_status


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_status():
    """Print current data status."""
    print("🔍 Checking current data status...")
    status = get_data_status()
    
    print(f"📁 CSV file exists: {'✅' if status['csv_file_exists'] else '❌'}")
    print(f"💾 Saved index exists: {'✅' if status['has_saved_index'] else '❌'}")
    print(f"🔄 Data loaded in memory: {'✅' if status['data_loaded'] else '❌'}")
    
    if status['data_loaded']:
        stats = status.get('vector_store_stats', {})
        print(f"📊 Total influencers: {stats.get('total_influencers', 0)}")
        print(f"📝 Text embeddings: {stats.get('text_embeddings', 0)}")
        print(f"🖼️  Image embeddings: {stats.get('image_embeddings', 0)}")
    
    print(f"📂 CSV path: {status['csv_file_path']}")
    print(f"💾 Vector store path: {status['vector_store_path']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process Instagram influencer data and generate embeddings"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/instagram_influencers_final_20250708.csv",
        help="Path to the Instagram CSV file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if data already exists"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for processing (default: 5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show data status, don't process"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("🚀 Instagram Data Processor")
    print("=" * 50)
    
    # Show status
    print_status()
    
    if args.status_only:
        return 0
    
    # Check if CSV file exists
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return 1
    
    # Check if processing is needed
    status = get_data_status()
    if status['data_loaded'] and not args.force:
        print("✅ Data is already loaded. Use --force to reprocess.")
        return 0
    
    print(f"\n🔄 Processing data from: {csv_path}")
    print(f"📦 Batch size: {args.batch_size}")
    
    try:
        start_time = time.time()
        
        # Process the data
        processor = InstagramDataProcessor()
        stats = processor.process_instagram_csv(str(csv_path))
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Processing completed in {elapsed_time:.2f} seconds")
        print(f"📊 Final statistics:")
        print(f"   Status: {stats.get('status', 'unknown')}")
        print(f"   Influencers loaded: {stats.get('influencers_loaded', 0)}")
        
        embedding_stats = stats.get('embedding_stats', {})
        if embedding_stats:
            print(f"   Successfully processed: {embedding_stats.get('processed_successfully', 0)}")
            print(f"   Failed: {embedding_stats.get('failed', 0)}")
            print(f"   Average time per influencer: {embedding_stats.get('average_time_per_influencer', 0):.2f}s")
        
        vector_stats = stats.get('vector_store_stats', {})
        if vector_stats:
            print(f"   Total in vector store: {vector_stats.get('total_influencers', 0)}")
            print(f"   Text embeddings: {vector_stats.get('text_embeddings', 0)}")
            print(f"   Image embeddings: {vector_stats.get('image_embeddings', 0)}")
        
        if stats.get('status') == 'completed':
            print(f"\n🎉 Success! Data is ready for use.")
            return 0
        else:
            print(f"\n❌ Processing failed or incomplete.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 