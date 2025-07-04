import os

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.clip import ClipEmbedding


def init_settings():
    if os.getenv("OPENAI_API_KEY") is None:
        raise RuntimeError("OPENAI_API_KEY is missing in environment variables")
    Settings.llm = OpenAI(model=os.getenv("MODEL") or "gpt-4o")

    Settings.embed_model = ClipEmbedding()

