#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from milvus_rag.config import load_settings
from milvus_rag.container import AppContainer


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Test retrieval with a known HotpotQA row.")
    parser.add_argument("--dataset-path", default=settings.dataset_path)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=settings.default_retrieve_k)
    parser.add_argument(
        "--mode",
        default="hybrid_rerank",
        choices=["dense", "sparse", "hybrid", "hybrid_rerank"],
    )
    args = parser.parse_args()

    container = AppContainer(settings)
    dataset = load_from_disk(args.dataset_path)[args.split]
    row = dataset[args.row]

    matches = container.retrieval.retrieve(
        row["question"],
        mode=args.mode,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
    )

    print("mode:", args.mode)
    print("question:", row["question"])
    print("expected_answer:", row["answer"])
    print()

    found_answer = False
    found_source = False
    for index, match in enumerate(matches, start=1):
        if match["answer"] == row["answer"]:
            found_answer = True
        if match["source_id"] == row["id"]:
            found_source = True
        print(f"{index}. score={match['score']:.4f}")
        if "rerank_score" in match:
            print(f"   rerank_score={match['rerank_score']:.4f}")
        print("   source_id:", match["source_id"])
        print("   title:", match["title"])
        print("   text:", match["text"][:240].replace("\n", " "), "...")
        print()

    print("same_source_found:", found_source)
    print("expected_answer_found:", found_answer)
    print("status: OK" if (found_source or found_answer) else "status: NOT FOUND")


if __name__ == "__main__":
    main()
