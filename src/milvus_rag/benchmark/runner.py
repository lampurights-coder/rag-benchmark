from pathlib import Path

from datasets import load_from_disk

from milvus_rag.benchmark.models import BenchmarkSample
from milvus_rag.config import Settings
from milvus_rag.retrieval.pipeline import RetrievalPipeline


class BenchmarkRunner:
    DEFAULT_MODES = ["dense", "sparse", "hybrid", "hybrid_rerank"]

    def __init__(self, settings: Settings, retrieval: RetrievalPipeline) -> None:
        self._settings = settings
        self._retrieval = retrieval

    def load_samples(self, split: str, limit: int, dataset_path: str | None = None) -> list[BenchmarkSample]:
        path = dataset_path or self._settings.dataset_path
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        dataset = load_from_disk(path)[split]
        count = min(limit, len(dataset))
        return [
            BenchmarkSample(
                row_index=row_index,
                source_id=str(dataset[row_index].get("id", row_index)),
                question=dataset[row_index]["question"],
                expected_answer=dataset[row_index]["answer"],
            )
            for row_index in range(count)
        ]

    @staticmethod
    def evaluate_matches(matches: list[dict], sample: BenchmarkSample, top_k: int) -> dict:
        same_source_rank = None
        answer_rank = None

        for index, match in enumerate(matches[:top_k], start=1):
            if same_source_rank is None and match.get("source_id") == sample.source_id:
                same_source_rank = index
            if answer_rank is None and match.get("answer") == sample.expected_answer:
                answer_rank = index

        return {
            "same_source_hit": same_source_rank is not None,
            "same_source_rank": same_source_rank,
            "answer_hit": answer_rank is not None,
            "answer_rank": answer_rank,
            "reciprocal_rank": 1.0 / same_source_rank if same_source_rank else 0.0,
            "top_score": matches[0]["score"] if matches else 0.0,
            "top_rerank_score": matches[0].get("rerank_score", matches[0]["score"]) if matches else 0.0,
        }

    @staticmethod
    def summarize_mode_results(rows: list[dict]) -> dict:
        if not rows:
            return {
                "samples": 0,
                "same_source_hit_rate": 0.0,
                "answer_hit_rate": 0.0,
                "mrr": 0.0,
                "avg_top_score": 0.0,
            }

        sample_count = len(rows)
        return {
            "samples": sample_count,
            "same_source_hit_rate": sum(row["same_source_hit"] for row in rows) / sample_count,
            "answer_hit_rate": sum(row["answer_hit"] for row in rows) / sample_count,
            "mrr": sum(row["reciprocal_rank"] for row in rows) / sample_count,
            "avg_top_score": sum(row["top_score"] for row in rows) / sample_count,
        }

    def run(
        self,
        split: str = "validation",
        limit: int = 25,
        top_k: int = 5,
        retrieve_k: int | None = None,
        modes: list[str] | None = None,
        dataset_path: str | None = None,
    ) -> dict:
        retrieve_k = retrieve_k or self._settings.default_retrieve_k
        modes = modes or self.DEFAULT_MODES
        path = dataset_path or self._settings.dataset_path
        samples = self.load_samples(split, limit, path)

        per_query = []
        mode_summaries: dict[str, list[dict]] = {mode: [] for mode in modes}

        for sample in samples:
            query_result = {
                "row_index": sample.row_index,
                "source_id": sample.source_id,
                "question": sample.question,
                "expected_answer": sample.expected_answer,
                "modes": {},
            }
            for mode in modes:
                matches = self._retrieval.retrieve(
                    sample.question, mode=mode, top_k=top_k, retrieve_k=retrieve_k
                )
                metrics = self.evaluate_matches(matches, sample, top_k=top_k)
                mode_summaries[mode].append(metrics)
                query_result["modes"][mode] = {"metrics": metrics, "matches": matches}
            per_query.append(query_result)

        summary = {mode: self.summarize_mode_results(rows) for mode, rows in mode_summaries.items()}
        return {
            "dataset_path": path,
            "split": split,
            "limit": limit,
            "top_k": top_k,
            "retrieve_k": retrieve_k,
            "modes": modes,
            "summary": summary,
            "results": per_query,
        }
