"""Pinned embedding model configuration."""

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Pinned embedding model configuration."""

    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    dimensions: int = Field(default=384)
    max_length: int = Field(default=512)
    cache_dir: str | None = Field(default=None)


EMBEDDING_CONFIG = EmbeddingConfig()
