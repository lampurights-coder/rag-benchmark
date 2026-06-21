from pydantic import BaseModel, Field

from milvus_rag.config import load_settings

_settings = load_settings()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieve_k: int = Field(default=_settings.default_retrieve_k, ge=1, le=100)
    mode: str = Field(
        default="hybrid_rerank",
        pattern="^(dense|sparse|hybrid|hybrid_rerank)$",
    )


class RagRequest(SearchRequest):
    max_context_chars: int = Field(default=6000, ge=500, le=20000)


class BenchmarkRequest(BaseModel):
    split: str = Field(default="validation", pattern="^(train|validation)$")
    limit: int = Field(default=25, ge=1, le=200)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieve_k: int = Field(default=_settings.default_retrieve_k, ge=1, le=100)
    modes: list[str] = Field(
        default_factory=lambda: ["dense", "sparse", "hybrid", "hybrid_rerank"]
    )
