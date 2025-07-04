import logging
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_index():
    """
    Index the documents in the data directory for influencer search.
    """
    from app.index import STORAGE_DIR
    from app.settings import init_settings
    from llama_index.core.indices import (
        VectorStoreIndex,
    )
    from llama_index.core.readers import SimpleDirectoryReader

    load_dotenv()
    init_settings()

    logger.info("Creating new influencer search index")
    # load the documents and create the index
    reader = SimpleDirectoryReader(
        os.environ.get("DATA_DIR", "data"),
        recursive=True,
    )
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    # store it for later
    index.storage_context.persist(STORAGE_DIR)
    logger.info(f"Finished creating influencer search index. Stored in {STORAGE_DIR}")


def generate_embeddings():
    """
    Generate embeddings for influencer data using CLIP model.
    This function will be expanded to handle multimodal data.
    """
    load_dotenv()
    
    logger.info("Generating embeddings for influencer data...")
    # TODO: Implement CLIP-based embedding generation for text and images
    # This will be implemented in Phase 3
    logger.info("Embedding generation placeholder - to be implemented in Phase 3")
