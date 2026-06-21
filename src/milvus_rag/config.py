import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class Settings:
    milvus_uri: str
    milvus_collection: str
    embed_model: str
    embed_dim: int
    embed_batch_size: int
    embed_max_length: int
    device: str
    embed_use_fp16: bool
    rerank_model: str
    rerank_use_fp16: bool
    rerank_batch_size: int
    warmup_on_start: bool
    hybrid_sparse_weight: float
    hybrid_dense_weight: float
    default_retrieve_k: int
    dataset_path: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    config_path: str

    def device_info(self) -> dict:
        cuda_available = torch.cuda.is_available()
        return {
            "device": self.device,
            "cuda_available": cuda_available,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "embed_fp16": self.embed_use_fp16,
            "rerank_fp16": self.rerank_use_fp16,
            "warmup_on_start": self.warmup_on_start,
        }


def _resolve_device(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized in {"cuda", "gpu"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return normalized


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _section(data: dict, key: str) -> dict:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _pick(env_key: str | None, file_value, default):
    if file_value is not None:
        return file_value
    if env_key and os.getenv(env_key) is not None:
        return os.getenv(env_key)
    return default


def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path or os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    data = _load_toml(path)

    milvus = _section(data, "milvus")
    device_cfg = _section(data, "device")
    embed = _section(data, "embed")
    rerank = _section(data, "rerank")
    retrieval = _section(data, "retrieval")
    dataset = _section(data, "dataset")
    llm = _section(data, "llm")

    device_name = _pick("DEVICE", device_cfg.get("name"), "auto")
    device = _resolve_device(str(device_name))
    env_device = os.getenv("DEVICE")
    if env_device and env_device != str(device_name):
        logger.warning(
            "Ignoring DEVICE=%s from environment; using config.toml value %r",
            env_device,
            device_name,
        )
    embed_fp16_default = device.startswith("cuda")
    warmup_on_start = _pick("WARMUP_ON_START", device_cfg.get("warmup_on_start"), True)

    embed_use_fp16 = _pick("EMBED_USE_FP16", embed.get("use_fp16"), embed_fp16_default)
    rerank_use_fp16 = _pick("RERANK_USE_FP16", rerank.get("use_fp16"), embed_fp16_default)

    return Settings(
        milvus_uri=str(_pick("MILVUS_URI", milvus.get("uri"), "http://localhost:19530")),
        milvus_collection=str(
            _pick("MILVUS_COLLECTION", milvus.get("collection"), "hotpotqa_bge_m3")
        ),
        embed_model=str(_pick("EMBED_MODEL", embed.get("model"), "BAAI/bge-m3")),
        embed_dim=int(_pick("EMBED_DIM", embed.get("dim"), 1024)),
        embed_batch_size=int(_pick("EMBED_BATCH_SIZE", embed.get("batch_size"), 8)),
        embed_max_length=int(_pick("EMBED_MAX_LENGTH", embed.get("max_length"), 512)),
        device=device,
        embed_use_fp16=str(embed_use_fp16).lower() == "true",
        rerank_model=str(_pick("RERANK_MODEL", rerank.get("model"), "BAAI/bge-reranker-v2-m3")),
        rerank_use_fp16=str(rerank_use_fp16).lower() == "true",
        rerank_batch_size=int(_pick("RERANK_BATCH_SIZE", rerank.get("batch_size"), 16)),
        warmup_on_start=str(warmup_on_start).lower() == "true",
        hybrid_sparse_weight=float(
            _pick("HYBRID_SPARSE_WEIGHT", retrieval.get("sparse_weight"), 0.3)
        ),
        hybrid_dense_weight=float(
            _pick("HYBRID_DENSE_WEIGHT", retrieval.get("dense_weight"), 0.7)
        ),
        default_retrieve_k=int(
            _pick("DEFAULT_RETRIEVE_K", retrieval.get("default_retrieve_k"), 20)
        ),
        dataset_path=str(_pick("DATASET_PATH", dataset.get("path"), "../hotpotqa_data")),
        llm_base_url=str(_pick("LLM_BASE_URL", llm.get("base_url"), "http://localhost:8080/v1")),
        llm_api_key=str(_pick("LLM_API_KEY", llm.get("api_key"), "not-needed")),
        llm_model=str(_pick("LLM_MODEL", llm.get("model"), "any-model")),
        config_path=str(path.resolve()),
    )
