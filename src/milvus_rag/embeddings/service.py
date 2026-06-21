from FlagEmbedding import BGEM3FlagModel

from milvus_rag.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: BGEM3FlagModel | None = None

    def _device_list(self) -> list[str] | None:
        if not self._settings.device:
            return None
        return [self._settings.device]

    @property
    def model(self) -> BGEM3FlagModel:
        if self._model is None:
            self._model = BGEM3FlagModel(
                self._settings.embed_model,
                use_fp16=self._settings.embed_use_fp16,
                devices=self._device_list(),
            )
        return self._model

    def runtime_device(self) -> str:
        if self._model is None:
            return self._settings.device
        devices = getattr(self._model, "target_devices", None)
        if devices:
            return ", ".join(str(device) for device in devices)
        return self._settings.device

    def warmup(self) -> None:
        self.embed_query("warmup")

    @staticmethod
    def _lexical_to_sparse(lexical_weights: dict) -> dict[int, float]:
        return {int(token_id): float(weight) for token_id, weight in lexical_weights.items()}

    def encode_texts(self, texts: list[str]) -> dict:
        if not texts:
            return {"dense": [], "sparse": []}

        output = self.model.encode(
            texts,
            batch_size=self._settings.embed_batch_size,
            max_length=self._settings.embed_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"].astype("float32").tolist()
        sparse_vectors = [self._lexical_to_sparse(w) for w in output["lexical_weights"]]
        return {"dense": dense_vectors, "sparse": sparse_vectors}

    def embed_documents(self, texts: list[str]) -> dict:
        return self.encode_texts(texts)

    def embed_query(self, text: str) -> dict:
        encoded = self.encode_texts([text])
        return {"dense": encoded["dense"][0], "sparse": encoded["sparse"][0]}
