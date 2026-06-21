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
    parser = argparse.ArgumentParser(description="Drop or inspect the Milvus collection.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate empty collection.")
    args = parser.parse_args()

    container = AppContainer(settings)
    if args.reset:
        container.store.reset_collection()
        print(f"Cleared collection. num_entities={container.store.get_entity_count()}")
    else:
        print(f"collection: {settings.milvus_collection}")
        print(f"num_entities: {container.store.get_entity_count()}")
        splits = container.store.get_indexed_splits()
        print(f"indexed_splits: {', '.join(splits) if splits else 'none'}")


if __name__ == "__main__":
    main()
