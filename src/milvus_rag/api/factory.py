from contextlib import asynccontextmanager
import logging

import gradio as gr
from fastapi import FastAPI, HTTPException

from milvus_rag.api.schemas import BenchmarkRequest, RagRequest, SearchRequest
from milvus_rag.container import AppContainer, get_container
from milvus_rag.ui.gradio_app import CUSTOM_CSS, GradioLab

logger = logging.getLogger(__name__)


def create_app(container: AppContainer | None = None) -> FastAPI:
    container = container or get_container()
    container.connect()
    container.warmup_models()
    logger.info(
        "Models ready | config=%s | device=%s | embedder=%s | reranker=%s",
        container.settings.config_path,
        container.settings.device,
        container.embedder.runtime_device(),
        container.reranker.runtime_device(),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="Milvus BGE-M3 RAG",
        description=(
            "Hybrid retrieval with BAAI/bge-m3 (dense + sparse), "
            "reranking with BAAI/bge-reranker-v2-m3, and HotpotQA benchmarks."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "entities": container.store.get_entity_count(),
            "device": container.settings.device_info(),
            "embedder_device": container.embedder.runtime_device(),
            "reranker_device": container.reranker.runtime_device(),
        }

    @app.post("/search")
    def search_endpoint(request: SearchRequest) -> dict:
        matches = container.retrieval.retrieve(
            request.query,
            mode=request.mode,
            top_k=request.top_k,
            retrieve_k=request.retrieve_k,
        )
        return {
            "query": request.query,
            "mode": request.mode,
            "top_k": request.top_k,
            "retrieve_k": max(request.top_k, request.retrieve_k),
            "matches": matches,
        }

    @app.post("/search/dense")
    def search_dense_endpoint(request: SearchRequest) -> dict:
        encoded = container.embedder.embed_query(request.query)
        matches = container.store.search_dense(encoded["dense"], top_k=request.top_k)
        return {"query": request.query, "mode": "dense", "matches": matches}

    @app.post("/search/sparse")
    def search_sparse_endpoint(request: SearchRequest) -> dict:
        encoded = container.embedder.embed_query(request.query)
        matches = container.store.search_sparse(encoded["sparse"], top_k=request.top_k)
        return {"query": request.query, "mode": "sparse", "matches": matches}

    @app.post("/search/hybrid")
    def search_hybrid_endpoint(request: SearchRequest) -> dict:
        encoded = container.embedder.embed_query(request.query)
        matches = container.store.search_hybrid(
            encoded["dense"], encoded["sparse"], top_k=request.top_k
        )
        return {"query": request.query, "mode": "hybrid", "matches": matches}

    @app.post("/benchmark")
    def benchmark_endpoint(request: BenchmarkRequest) -> dict:
        return container.benchmark.run(
            split=request.split,
            limit=request.limit,
            top_k=request.top_k,
            retrieve_k=request.retrieve_k,
            modes=request.modes,
        )

    @app.post("/rag")
    def rag_endpoint(request: RagRequest) -> dict:
        try:
            return container.rag.answer(
                request.query,
                mode=request.mode,
                top_k=request.top_k,
                retrieve_k=request.retrieve_k,
                max_context_chars=request.max_context_chars,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    demo = GradioLab(container).build()
    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
