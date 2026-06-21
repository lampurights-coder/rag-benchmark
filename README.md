# Milvus RAG with BGE-M3

HotpotQA retrieval pipeline using [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) hybrid embeddings, [Milvus](https://milvus.io/) vector search, and [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) reranking.

## Project layout

```text
milvus_rag/
├── app.py                 # FastAPI + Gradio entry point
├── config.toml            # Runtime settings (edit this instead of env exports)
├── docker-compose.yml     # Milvus stack
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── ingest.py          # Index HotpotQA into Milvus
│   ├── clear_milvus.py    # Drop or inspect collection
│   └── verify_retrieval.py
└── src/milvus_rag/
    ├── config.py          # Settings dataclass
    ├── container.py       # AppContainer (wires all services)
    ├── text_utils.py      # Text formatting helpers
    ├── api/
    │   ├── factory.py     # create_app()
    │   └── schemas.py     # Pydantic request models
    ├── embeddings/
    │   └── service.py     # EmbeddingService (BGE-M3)
    ├── reranker/
    │   └── service.py     # RerankerService
    ├── store/
    │   └── milvus_store.py # MilvusStore
    ├── retrieval/
    │   └── pipeline.py    # RetrievalPipeline
    ├── benchmark/
    │   ├── models.py      # BenchmarkSample
    │   └── runner.py      # BenchmarkRunner
    ├── ingest/
    │   └── hotpotqa.py    # HotpotQAIngester
    ├── rag/
    │   └── service.py     # RagService
    └── ui/
        ├── formatters.py  # HTML/chart helpers
        └── gradio_app.py  # GradioLab UI class
```

## How it works

### HotpotQA row → Milvus chunks

Each dataset row is one multi-hop question with multiple Wikipedia passages in `context`:

| Field | Example |
|-------|---------|
| `id` | `5ae732685542991e8301cbc3` |
| `question` | Zimbabwe's Guwe Secondary School has a sister school in what New York county? |
| `answer` | `Nassau County` |
| `context.title[]` | Guwe Secondary School, Carle Place High School, … (10 articles) |
| `context.sentences[][]` | Sentences per article |

**One QA row → one vector per context article** (~10 chunks per question).

The two supporting passages for the example above are:

- **Guwe Secondary School** — mentions sister school in Carle Place, NY
- **Carle Place High School** — located in **Nassau County**, New York

`HotpotQAIngester` embeds each chunk as:

```text
{question}
{title}
{text}
```

### Retrieval pipeline

```text
Query → EmbeddingService (dense + sparse)
     → MilvusStore hybrid search
     → RerankerService (optional)
     → top-K passages
```

| Class | Role |
|-------|------|
| `EmbeddingService` | BGE-M3 dense + sparse vectors |
| `MilvusStore` | Insert/search/clear vectors |
| `RerankerService` | Cross-encoder relevance scores |
| `RetrievalPipeline` | Orchestrates dense/sparse/hybrid/rerank modes |
| `BenchmarkRunner` | HotpotQA MRR and hit-rate evaluation |
| `RagService` | Retrieval + OpenAI-compatible LLM |
| `GradioLab` | Search, compare, benchmark UI |

### Train vs validation

Benchmark split **must match** ingested split:

| Indexed | Benchmark | Result |
|---------|-----------|--------|
| validation | validation | Good |
| train + validation | either | Good |
| validation only | train | 0% hits |

## Quick start

### 1. Start Milvus

```bash
cd milvus_rag
docker compose up -d
```

### 2. Install

```bash
conda activate ollama
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install 'transformers>=4.46,<5'
pip install -e .
```

### 3. Clear and ingest

```bash
python scripts/clear_milvus.py --reset
python scripts/clear_milvus.py                    # num_entities: 0

python scripts/ingest.py --split validation --limit 200 --reset
python scripts/ingest.py --split train --limit 200  # no --reset

python scripts/clear_milvus.py                    # ~4000 entities, both splits
```

Edit `config.toml` for GPU, batch size, and model settings — no shell exports needed.

### 4. Run app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- UI: http://localhost:8000/
- API: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 5. Verify

```bash
python scripts/verify_retrieval.py --split validation --row 0 --mode hybrid_rerank
```

## Scripts

| Command | Description |
|---------|-------------|
| `python scripts/clear_milvus.py` | Show entity count and indexed splits |
| `python scripts/clear_milvus.py --reset` | Drop all indexed data |
| `python scripts/ingest.py --split validation --limit 200 --reset` | Ingest 200 validation questions |
| `python scripts/ingest.py --split train --limit 200` | Add train data |
| `python scripts/verify_retrieval.py --row 0` | Test one known question |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Milvus count + GPU info |
| POST | `/search` | Hybrid retrieval (configurable mode) |
| POST | `/search/dense` | Dense only |
| POST | `/search/sparse` | Sparse only |
| POST | `/search/hybrid` | Hybrid only |
| POST | `/benchmark` | Run HotpotQA benchmark |
| POST | `/rag` | Retrieve + LLM answer |

## Configuration

Edit **`config.toml`** in the project root. Example:

```toml
[device]
# auto | cuda | cuda:0 | cpu
name = "cuda"
warmup_on_start = true

[embed]
batch_size = 16
use_fp16 = true

[rerank]
use_fp16 = true

[dataset]
path = "../hotpotqa_data"
```

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `[device]` | `name` | `auto` | `auto`, `cuda`, `cuda:0`, or `cpu` |
| `[device]` | `warmup_on_start` | `true` | Load models to GPU at startup |
| `[milvus]` | `uri` | `http://localhost:19530` | Milvus address |
| `[milvus]` | `collection` | `hotpotqa_bge_m3` | Collection name |
| `[embed]` | `model` | `BAAI/bge-m3` | Embedding model |
| `[embed]` | `batch_size` | `16` | Ingest/search batch size |
| `[embed]` | `use_fp16` | `true` on GPU | FP16 embeddings |
| `[rerank]` | `model` | `BAAI/bge-reranker-v2-m3` | Reranker |
| `[rerank]` | `use_fp16` | `true` on GPU | FP16 reranker |
| `[retrieval]` | `default_retrieve_k` | `20` | Candidate pool before rerank |
| `[dataset]` | `path` | `../hotpotqa_data` | HotpotQA on disk |
| `[llm]` | `base_url` | `http://localhost:8080/v1` | RAG LLM endpoint |

Use a different file with `CONFIG_PATH=/path/to/config.toml`. Environment variables are only used when a key is missing from the config file.

If GPU still shows CPU, check for a stale `DEVICE=cpu` in your shell (`echo $DEVICE`) and restart the app.

## GPU setup

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install CUDA PyTorch (not `+cpu`):

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install 'transformers>=4.46,<5'
```

## Troubleshooting

- **0% benchmark hits** — benchmark split not ingested; check `python scripts/clear_milvus.py`
- **`TrainingArguments` error** — downgrade transformers: `pip install 'transformers>=4.46,<5'`
- **Wrong search results** — query not in indexed subset; use Examples tab or matching split
- **CUDA OOM** — lower `EMBED_BATCH_SIZE=4` or stop uvicorn before ingest

## References

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
