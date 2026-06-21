from openai import OpenAI

from milvus_rag.config import Settings
from milvus_rag.retrieval.pipeline import RetrievalPipeline


class RagService:
    SYSTEM_PROMPT = (
        "Answer using only the provided context. "
        "If the answer is not present, say you do not know."
    )

    def __init__(self, settings: Settings, retrieval: RetrievalPipeline) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    def answer(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 5,
        retrieve_k: int | None = None,
        max_context_chars: int = 6000,
    ) -> dict:
        matches = self._retrieval.retrieve(
            query, mode=mode, top_k=top_k, retrieve_k=retrieve_k
        )
        context = "\n\n".join(
            f"[{index}] {match['title']}\n{match['text']}"
            for index, match in enumerate(matches, start=1)
        )[:max_context_chars]

        if not context:
            raise ValueError("No context found in Milvus. Run ingestion first.")

        response = self._client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
            ],
            temperature=0.2,
        )
        return {
            "query": query,
            "mode": mode,
            "answer": response.choices[0].message.content,
            "matches": matches,
        }
