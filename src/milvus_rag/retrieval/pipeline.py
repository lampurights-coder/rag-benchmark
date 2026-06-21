from milvus_rag.embeddings.service import EmbeddingService
from milvus_rag.reranker.service import RerankerService
from milvus_rag.store.milvus_store import MilvusStore
from milvus_rag.text_utils import format_document_text


class RetrievalPipeline:
    MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")

    def __init__(
        self,
        embedder: EmbeddingService,
        reranker: RerankerService,
        store: MilvusStore,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._store = store

    def retrieve(
        self,
        query: str,
        mode: str,
        top_k: int,
        retrieve_k: int | None = None,
    ) -> list[dict]:
        encoded = self._embedder.embed_query(query)
        candidate_k = max(top_k, retrieve_k or top_k)

        if mode == "dense":
            matches = self._store.search_dense(encoded["dense"], top_k=candidate_k)
        elif mode == "sparse":
            matches = self._store.search_sparse(encoded["sparse"], top_k=candidate_k)
        elif mode == "hybrid":
            matches = self._store.search_hybrid(
                encoded["dense"], encoded["sparse"], top_k=candidate_k
            )
        elif mode == "hybrid_rerank":
            candidates = self._store.search_hybrid(
                encoded["dense"], encoded["sparse"], top_k=candidate_k
            )
            matches = self._apply_rerank(query, candidates)
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        return matches[:top_k]

    def _apply_rerank(self, query: str, matches: list[dict]) -> list[dict]:
        if not matches:
            return []

        passages = [
            format_document_text(match.get("title", ""), match.get("text", ""))
            for match in matches
        ]
        scores = self._reranker.score_pairs(query, passages)
        reranked = []
        for match, rerank_score in zip(matches, scores, strict=True):
            item = dict(match)
            item["retrieval_score"] = item.get("score")
            item["rerank_score"] = rerank_score
            item["score"] = rerank_score
            reranked.append(item)
        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked
