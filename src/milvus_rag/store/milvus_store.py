from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    WeightedRanker,
)

from milvus_rag.config import Settings

OUTPUT_FIELDS = ["source_id", "split", "title", "question", "answer", "text"]


class MilvusStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: MilvusClient | None = None
        self._collection_ready = False

    @property
    def collection_name(self) -> str:
        return self._settings.milvus_collection

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._settings.milvus_uri)
        return self._client

    def connect(self) -> None:
        self._get_client()

    def _create_collection(self, client: MilvusClient) -> None:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._settings.embed_dim,
        )
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="split", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def ensure_collection(self, reset: bool = False) -> None:
        client = self._get_client()
        name = self.collection_name

        if reset:
            self._collection_ready = False
            if client.has_collection(name):
                client.drop_collection(name)

        if not client.has_collection(name):
            self._create_collection(client)

        if not self._collection_ready:
            client.load_collection(name)
            self._collection_ready = True

    def insert_rows(self, rows: list[dict]) -> None:
        if not rows:
            return

        self.ensure_collection()
        self._get_client().insert(
            self.collection_name,
            [
                {
                    "pk": row["pk"],
                    "dense_vector": row["dense_vector"],
                    "sparse_vector": row["sparse_vector"],
                    "source_id": row["source_id"],
                    "split": row["split"],
                    "title": row["title"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "text": row["text"],
                }
                for row in rows
            ],
        )

    @staticmethod
    def _hit_to_match(hit: dict) -> dict:
        return {
            "score": float(hit.get("distance", hit.get("score", 0.0))),
            "source_id": hit.get("source_id"),
            "split": hit.get("split"),
            "title": hit.get("title"),
            "question": hit.get("question"),
            "answer": hit.get("answer"),
            "text": hit.get("text"),
        }

    def search_dense(self, dense_vector: list[float], top_k: int = 5) -> list[dict]:
        self.ensure_collection()
        results = self._get_client().search(
            collection_name=self.collection_name,
            data=[dense_vector],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
        )
        return [self._hit_to_match(hit) for hit in results[0]]

    def search_sparse(self, sparse_vector: dict[int, float], top_k: int = 5) -> list[dict]:
        self.ensure_collection()
        results = self._get_client().search(
            collection_name=self.collection_name,
            data=[sparse_vector],
            anns_field="sparse_vector",
            search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
        )
        return [self._hit_to_match(hit) for hit in results[0]]

    def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int = 5,
    ) -> list[dict]:
        self.ensure_collection()
        dense_req = AnnSearchRequest(
            [dense_vector],
            "dense_vector",
            {"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
        )
        sparse_req = AnnSearchRequest(
            [sparse_vector],
            "sparse_vector",
            {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
        )
        results = self._get_client().hybrid_search(
            collection_name=self.collection_name,
            reqs=[sparse_req, dense_req],
            ranker=WeightedRanker(
                self._settings.hybrid_sparse_weight,
                self._settings.hybrid_dense_weight,
            ),
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
        )
        return [self._hit_to_match(hit) for hit in results[0]]

    def get_entity_count(self) -> int:
        self.ensure_collection()
        stats = self._get_client().get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def get_indexed_splits(self, limit: int = 16384) -> list[str]:
        self.ensure_collection()
        if self.get_entity_count() == 0:
            return []

        rows = self._get_client().query(
            collection_name=self.collection_name,
            filter='split in ["train", "validation"]',
            output_fields=["split"],
            limit=limit,
        )
        return sorted({row["split"] for row in rows if row.get("split")})

    def reset_collection(self) -> None:
        self.ensure_collection(reset=True)
