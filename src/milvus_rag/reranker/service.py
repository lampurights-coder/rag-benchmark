import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from milvus_rag.config import Settings


class RerankerService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer = None
        self._model = None

    def _resolve_device(self) -> torch.device:
        device_name = self._settings.device
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name if ":" in device_name else "cuda:0")
        return torch.device("cpu")

    def runtime_device(self) -> str:
        if self._model is None:
            return self._settings.device
        return str(next(self._model.parameters()).device)

    def warmup(self) -> None:
        self.score_pairs("warmup", ["warmup"], normalize=False)

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._settings.rerank_model)
        return self._tokenizer

    def _load_model(self):
        if self._model is None:
            device = self._resolve_device()
            model = AutoModelForSequenceClassification.from_pretrained(self._settings.rerank_model)
            model.eval()
            if self._settings.rerank_use_fp16 and device.type == "cuda":
                model = model.half()
            self._model = model.to(device)
        return self._model

    def score_pairs(self, query: str, passages: list[str], normalize: bool = True) -> list[float]:
        if not passages:
            return []

        tokenizer = self._load_tokenizer()
        model = self._load_model()
        device = next(model.parameters()).device
        pairs = [[query, passage] for passage in passages]
        scores: list[float] = []

        for start in range(0, len(pairs), self._settings.rerank_batch_size):
            batch = pairs[start : start + self._settings.rerank_batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits.view(-1)
                if self._settings.rerank_use_fp16 and device.type == "cuda":
                    logits = logits.float()
                if normalize:
                    logits = torch.sigmoid(logits)
                scores.extend(logits.cpu().tolist())

        return [float(score) for score in scores]
