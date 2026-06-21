from pathlib import Path

from datasets import load_from_disk

from milvus_rag.config import Settings
from milvus_rag.embeddings.service import EmbeddingService
from milvus_rag.store.milvus_store import MilvusStore
from milvus_rag.text_utils import format_hotpot_chunk_text


class HotpotQAIngester:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingService,
        store: MilvusStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    @staticmethod
    def clean_text(value: str, max_length: int) -> str:
        return " ".join(str(value or "").split())[:max_length]

    def iter_chunks(self, split: str, limit: int | None = None, dataset_path: str | None = None):
        path = dataset_path or self._settings.dataset_path
        dataset = load_from_disk(path)[split]
        count = len(dataset) if limit is None else min(limit, len(dataset))

        for row_index in range(count):
            row = dataset[row_index]
            source_id = self.clean_text(row.get("id", str(row_index)), 256)
            question = self.clean_text(row.get("question", ""), 4096)
            answer = self.clean_text(row.get("answer", ""), 4096)
            context = row.get("context", {})
            titles = context.get("title", [])
            sentences_by_title = context.get("sentences", [])

            for context_index, title in enumerate(titles):
                sentences = (
                    sentences_by_title[context_index]
                    if context_index < len(sentences_by_title)
                    else []
                )
                text = self.clean_text(" ".join(sentences), 65535)
                if not text:
                    continue
                yield {
                    "pk": f"{split}:{source_id}:{context_index}",
                    "source_id": source_id,
                    "split": split,
                    "title": self.clean_text(title, 1024),
                    "question": question,
                    "answer": answer,
                    "text": text,
                }

    def ingest(
        self,
        split: str,
        limit: int | None = None,
        reset: bool = False,
        dataset_path: str | None = None,
    ) -> int:
        path = dataset_path or self._settings.dataset_path
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        self._store.ensure_collection(reset=reset)

        pending: list[dict] = []
        total = 0
        for chunk in self.iter_chunks(split, limit, path):
            pending.append(chunk)
            if len(pending) >= self._settings.embed_batch_size:
                total += self._insert_batch(pending)
                print(f"Inserted {total} chunks")
                pending = []

        if pending:
            total += self._insert_batch(pending)
            print(f"Inserted {total} chunks")

        print(f"Done. Total chunks inserted: {total}")
        return total

    def _insert_batch(self, rows: list[dict]) -> int:
        encoded = self._embedder.embed_documents(
            [
                format_hotpot_chunk_text(row["question"], row["title"], row["text"])
                for row in rows
            ]
        )
        for row, dense_vector, sparse_vector in zip(
            rows, encoded["dense"], encoded["sparse"], strict=True
        ):
            row["dense_vector"] = dense_vector
            row["sparse_vector"] = sparse_vector
        self._store.insert_rows(rows)
        return len(rows)
