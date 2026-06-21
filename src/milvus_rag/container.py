from milvus_rag.benchmark.runner import BenchmarkRunner
from milvus_rag.config import Settings, load_settings
from milvus_rag.embeddings.service import EmbeddingService
from milvus_rag.ingest.hotpotqa import HotpotQAIngester
from milvus_rag.rag.service import RagService
from milvus_rag.reranker.service import RerankerService
from milvus_rag.retrieval.pipeline import RetrievalPipeline
from milvus_rag.store.milvus_store import MilvusStore


class AppContainer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.store = MilvusStore(self.settings)
        self.embedder = EmbeddingService(self.settings)
        self.reranker = RerankerService(self.settings)
        self.retrieval = RetrievalPipeline(self.embedder, self.reranker, self.store)
        self.benchmark = BenchmarkRunner(self.settings, self.retrieval)
        self.ingester = HotpotQAIngester(self.settings, self.embedder, self.store)
        self.rag = RagService(self.settings, self.retrieval)

    def connect(self) -> None:
        self.store.connect()

    def warmup_models(self) -> None:
        if not self.settings.warmup_on_start:
            return
        self.embedder.warmup()
        self.reranker.warmup()


_container: AppContainer | None = None


def get_container() -> AppContainer:
    global _container
    if _container is None:
        _container = AppContainer()
    return _container
