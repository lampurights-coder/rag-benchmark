#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from milvus_rag.config import load_settings
from milvus_rag.container import AppContainer


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Embed HotpotQA with BGE-M3 and store in Milvus.")
    parser.add_argument("--dataset-path", default=settings.dataset_path)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=200, help="QA rows. Use 0 for all.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection.")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    container = AppContainer(settings)
    container.ingester.ingest(
        split=args.split,
        limit=limit,
        reset=args.reset,
        dataset_path=args.dataset_path,
    )


if __name__ == "__main__":
    main()
