from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_from_disk

from milvus_rag.container import AppContainer
from milvus_rag.ui.formatters import (
    MODES,
    MODE_COLORS,
    format_matches_html,
    mode_label,
    mode_value,
)


CUSTOM_CSS = """
[class*="gradio-container"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  width: 100% !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}
gradio-app {
  display: block;
}
gradio-app .app {
  max-width: 1200px;
  margin: 0 auto;
}
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #0ea5e9 100%);
  color: white;
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 16px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
}
.hero h1 { margin: 0 0 8px 0; font-size: 1.8rem; }
.hero p { margin: 0; opacity: 0.92; }
.pipeline {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 14px;
}
.pipeline span {
  background: rgba(255,255,255,0.15);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 0.82rem;
}
.result-card {
  border: 1px solid #dbe3f0;
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 10px;
  background: #f8fafc;
}
.result-card.high { border-left: 4px solid #10b981; }
.result-card.mid { border-left: 4px solid #f59e0b; }
.result-card.low { border-left: 4px solid #ef4444; }
.result-card header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 0.75rem;
  margin-left: 6px;
}
.badge.good { background: #dcfce7; color: #166534; }
.pill {
  display: inline-block;
  border-radius: 6px;
  padding: 2px 8px;
  font-size: 0.8rem;
  font-weight: 600;
}
.pill.high { background: #d1fae5; color: #065f46; }
.pill.mid { background: #fef3c7; color: #92400e; }
.pill.low { background: #fee2e2; color: #991b1b; }
.score-bar {
  height: 6px;
  background: #e2e8f0;
  border-radius: 999px;
  overflow: hidden;
  margin: 6px 0 10px;
}
.score-fill { height: 100%; border-radius: 999px; }
.snippet { color: #475569; font-size: 0.92rem; }
.empty { color: #64748b; }
.compare-section { margin-bottom: 18px; }
.compare-section h3 { margin-bottom: 8px; }
"""


class GradioLab:
    def __init__(self, container: AppContainer) -> None:
        self._container = container

    def _empty_figure(self, title: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title=title, height=320)
        return fig

    def search_ui(self, query: str, mode: str, top_k: int, retrieve_k: int) -> tuple[str, str]:
        query = (query or "").strip()
        if not query:
            return "Enter a question to search.", ""

        matches = self._container.retrieval.retrieve(
            query, mode=mode, top_k=int(top_k), retrieve_k=int(retrieve_k)
        )
        summary = (
            f"Mode: **{mode_label(mode)}** | returned **{len(matches)}** results "
            f"| retrieve pool **{max(int(top_k), int(retrieve_k))}**"
        )
        return summary, format_matches_html(matches)

    def compare_ui(self, query: str, top_k: int, retrieve_k: int) -> tuple[str, go.Figure]:
        query = (query or "").strip()
        if not query:
            return "Enter a question to compare retrieval modes.", self._empty_figure("No query")

        sections = []
        chart_rows = []
        for label, mode in MODES:
            matches = self._container.retrieval.retrieve(
                query, mode=mode, top_k=int(top_k), retrieve_k=int(retrieve_k)
            )
            top_score = matches[0]["score"] if matches else 0.0
            chart_rows.append({"mode": label, "top_score": top_score})
            sections.append(
                f"<section class='compare-section'>"
                f"<h3 style='color:{MODE_COLORS.get(mode, '#334155')}'>{label}</h3>"
                f"{format_matches_html(matches)}</section>"
            )

        df = pd.DataFrame(chart_rows)
        fig = px.bar(
            df,
            x="mode",
            y="top_score",
            color="mode",
            color_discrete_map={label: MODE_COLORS.get(value, "#64748b") for label, value in MODES},
            title="Top-1 score by retrieval mode",
            labels={"top_score": "Top score", "mode": "Mode"},
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(t=48, b=40, l=40, r=20))
        return "\n".join(sections), fig

    def _benchmark_summary_df(self, report: dict) -> pd.DataFrame:
        rows = []
        for mode, metrics in report["summary"].items():
            rows.append(
                {
                    "mode": mode_label(mode),
                    "mode_key": mode,
                    "samples": metrics["samples"],
                    "same_source_hit_rate": round(metrics["same_source_hit_rate"] * 100, 1),
                    "answer_hit_rate": round(metrics["answer_hit_rate"] * 100, 1),
                    "mrr": round(metrics["mrr"], 3),
                    "avg_top_score": round(metrics["avg_top_score"], 3),
                }
            )
        return pd.DataFrame(rows)

    def _benchmark_charts(self, summary_df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
        if summary_df.empty:
            empty = self._empty_figure("Run a benchmark first")
            return empty, empty

        color_map = {
            row["mode"]: MODE_COLORS.get(row["mode_key"], "#64748b")
            for _, row in summary_df.iterrows()
        }

        mrr_fig = px.bar(
            summary_df,
            x="mode",
            y="mrr",
            color="mode",
            color_discrete_map=color_map,
            title="Mean Reciprocal Rank (higher is better)",
            labels={"mrr": "MRR", "mode": "Mode"},
        )
        mrr_fig.update_layout(showlegend=False, height=340, margin=dict(t=48, b=40, l=40, r=20))

        hit_fig = go.Figure()
        hit_fig.add_trace(
            go.Bar(
                name="Same-source hit %",
                x=summary_df["mode"],
                y=summary_df["same_source_hit_rate"],
                marker_color="#38bdf8",
            )
        )
        hit_fig.add_trace(
            go.Bar(
                name="Answer hit %",
                x=summary_df["mode"],
                y=summary_df["answer_hit_rate"],
                marker_color="#34d399",
            )
        )
        hit_fig.update_layout(
            barmode="group",
            title="Hit rates on HotpotQA (higher is better)",
            yaxis_title="Percent",
            height=340,
            margin=dict(t=48, b=40, l=40, r=20),
        )
        return mrr_fig, hit_fig

    def benchmark_ui(
        self, split: str, limit: int, top_k: int, retrieve_k: int
    ) -> tuple[str, pd.DataFrame, go.Figure, go.Figure, str]:
        report = self._container.benchmark.run(
            split=split,
            limit=int(limit),
            top_k=int(top_k),
            retrieve_k=int(retrieve_k),
        )

        summary_df = self._benchmark_summary_df(report)
        mrr_fig, hit_fig = self._benchmark_charts(summary_df)

        detail_rows = []
        for item in report["results"]:
            for mode, payload in item["modes"].items():
                metrics = payload["metrics"]
                detail_rows.append(
                    {
                        "row": item["row_index"],
                        "mode": mode_label(mode),
                        "question": item["question"][:120],
                        "expected_answer": item["expected_answer"],
                        "same_source_hit": metrics["same_source_hit"],
                        "same_source_rank": metrics["same_source_rank"] or "-",
                        "answer_hit": metrics["answer_hit"],
                        "top_score": round(metrics["top_score"], 3),
                    }
                )

        indexed_splits = self._container.store.get_indexed_splits()
        best_mode = max(
            report["summary"].items(),
            key=lambda pair: (pair[1]["mrr"], pair[1]["avg_top_score"]),
        )[0]

        settings = self._container.settings
        split_warning = ""
        if indexed_splits and split not in indexed_splits:
            split_warning = (
                f"\n\n> **Warning:** Milvus contains `{', '.join(indexed_splits)}` only, "
                f"but you benchmarked **`{split}`**. Ingest that split first:\n"
                f"> `python scripts/ingest.py --split {split} --limit 200 --reset`"
            )
        elif not indexed_splits:
            split_warning = "\n\n> **Warning:** Milvus collection is empty. Run ingestion first."

        markdown = (
            f"Evaluated **{report['limit']}** `{report['split']}` questions from HotpotQA.\n\n"
            f"- Collection: `{settings.milvus_collection}` "
            f"({self._container.store.get_entity_count()} vectors)\n"
            f"- Indexed splits: `{', '.join(indexed_splits) or 'none'}`\n"
            f"- Embedding: `{settings.embed_model}`\n"
            f"- Reranker: `{settings.rerank_model}`\n"
            f"- Best MRR: **{mode_label(best_mode)}** ({report['summary'][best_mode]['mrr']:.3f})"
            f"{split_warning}"
        )
        return markdown, summary_df, mrr_fig, hit_fig, pd.DataFrame(detail_rows).to_markdown(index=False)

    def example_choices(self, split: str) -> list[tuple[str, int]]:
        try:
            samples = self._container.benchmark.load_samples(split, limit=50)
        except FileNotFoundError:
            return [("Dataset not found", -1)]
        return [(f"{s.row_index}: {s.question[:90]}", s.row_index) for s in samples]

    def example_ui(
        self, split: str, row_index: int, top_k: int, retrieve_k: int
    ) -> tuple[str, str, str, go.Figure]:
        if row_index < 0:
            return "Dataset not found.", "", "", self._empty_figure("Dataset missing")

        dataset = load_from_disk(self._container.settings.dataset_path)[split]
        row = dataset[row_index]
        question = row["question"]
        expected_answer = row["answer"]

        chart_rows = []
        for label, mode in MODES:
            matches = self._container.retrieval.retrieve(
                question, mode=mode, top_k=int(top_k), retrieve_k=int(retrieve_k)
            )
            chart_rows.append(
                {
                    "mode": label,
                    "top_score": matches[0]["score"] if matches else 0.0,
                }
            )

        best_matches = self._container.retrieval.retrieve(
            question, mode="hybrid_rerank", top_k=int(top_k), retrieve_k=int(retrieve_k)
        )
        header = (
            f"**Question:** {question}\n\n"
            f"**Expected answer:** {expected_answer}\n\n"
            f"**Source id:** {row.get('id', row_index)}"
        )

        df = pd.DataFrame(chart_rows)
        fig = px.bar(
            df,
            x="mode",
            y="top_score",
            color="mode",
            color_discrete_map={label: MODE_COLORS.get(value, "#64748b") for label, value in MODES},
            title="Relevance scores for this example question",
            labels={"top_score": "Top score", "mode": "Mode"},
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=48, b=40, l=40, r=20))
        return header, format_matches_html(best_matches, expected_answer=expected_answer), question, fig

    def system_status(self) -> str:
        try:
            settings = self._container.settings
            entity_count = self._container.store.get_entity_count()
            indexed_splits = self._container.store.get_indexed_splits()
            dataset_ok = "available" if Path(settings.dataset_path).exists() else "missing"
            device_info = settings.device_info()
            embed_device = self._container.embedder.runtime_device()
            rerank_device = self._container.reranker.runtime_device()
        except Exception as exc:
            return f"Milvus status check failed: {exc}"

        gpu_label = device_info.get("gpu_name") or "not detected"
        splits_label = ", ".join(indexed_splits) if indexed_splits else "none"
        return (
            f"| Component | Status |\n"
            f"|-----------|--------|\n"
            f"| Configured device | **{settings.device}** |\n"
            f"| CUDA available | **{device_info['cuda_available']}** ({gpu_label}) |\n"
            f"| Embedder runtime | **{embed_device}** (fp16={settings.embed_use_fp16}) |\n"
            f"| Reranker runtime | **{rerank_device}** (fp16={settings.rerank_use_fp16}) |\n"
            f"| Milvus collection `{settings.milvus_collection}` | **{entity_count}** vectors |\n"
            f"| Indexed splits | **{splits_label}** |\n"
            f"| Embedding model | `{settings.embed_model}` |\n"
            f"| Reranker | `{settings.rerank_model}` |\n"
            f"| Benchmark dataset | **{dataset_ok}** at `{settings.dataset_path}` |"
        )

    def build(self) -> gr.Blocks:
        settings = self._container.settings
        with gr.Blocks(title="BGE-M3 Milvus RAG Lab", fill_width=False) as demo:
            gr.HTML(
                """
                <div class="hero">
                  <h1>BGE-M3 Retrieval Lab</h1>
                  <p>Hybrid Milvus search with BGE-M3 embeddings and bge-reranker-v2-m3.</p>
                  <div class="pipeline">
                    <span>BAAI/bge-m3 dense</span>
                    <span>BAAI/bge-m3 sparse</span>
                    <span>Milvus hybrid</span>
                    <span>BAAI/bge-reranker-v2-m3</span>
                    <span>HotpotQA benchmark</span>
                  </div>
                </div>
                """
            )
            status = gr.Markdown(self.system_status())

            with gr.Tabs():
                with gr.Tab("Search"):
                    query = gr.Textbox(label="Query", lines=2)
                    with gr.Row():
                        mode = gr.Dropdown(
                            choices=[label for label, _ in MODES],
                            value=MODES[-1][0],
                            label="Retrieval mode",
                        )
                        top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                        retrieve_k = gr.Slider(
                            5, 50, value=settings.default_retrieve_k, step=1, label="Retrieve pool"
                        )
                    search_btn = gr.Button("Search", variant="primary")
                    search_summary = gr.Markdown()
                    search_results = gr.HTML()
                    search_btn.click(
                        lambda q, m, k, r: self.search_ui(q, mode_value(m), k, r),
                        [query, mode, top_k, retrieve_k],
                        [search_summary, search_results],
                    )

                with gr.Tab("Compare modes"):
                    compare_query = gr.Textbox(label="Query", lines=2)
                    with gr.Row():
                        compare_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                        compare_retrieve_k = gr.Slider(
                            5, 50, value=settings.default_retrieve_k, step=1, label="Retrieve pool"
                        )
                    compare_btn = gr.Button("Compare all modes", variant="primary")
                    compare_chart = gr.Plot()
                    compare_results = gr.HTML()
                    compare_btn.click(
                        self.compare_ui,
                        [compare_query, compare_top_k, compare_retrieve_k],
                        [compare_results, compare_chart],
                    )

                with gr.Tab("Benchmark"):
                    with gr.Row():
                        split = gr.Dropdown(["validation", "train"], value="validation", label="Dataset split")
                        sample_limit = gr.Slider(5, 100, value=25, step=1, label="Sample count")
                        bench_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                        bench_retrieve_k = gr.Slider(
                            5, 50, value=settings.default_retrieve_k, step=1, label="Retrieve pool"
                        )
                    benchmark_btn = gr.Button("Run benchmark", variant="primary")
                    benchmark_summary = gr.Markdown()
                    with gr.Row():
                        mrr_chart = gr.Plot()
                        hit_chart = gr.Plot()
                    benchmark_table = gr.Dataframe(interactive=False)
                    benchmark_details = gr.Markdown()
                    benchmark_btn.click(
                        self.benchmark_ui,
                        [split, sample_limit, bench_top_k, bench_retrieve_k],
                        [benchmark_summary, benchmark_table, mrr_chart, hit_chart, benchmark_details],
                    )

                with gr.Tab("Examples"):
                    with gr.Row():
                        example_split = gr.Dropdown(["validation", "train"], value="validation", label="Split")
                        example_pick = gr.Dropdown(
                            label="Pick a question",
                            choices=self.example_choices("validation"),
                        )
                    with gr.Row():
                        example_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                        example_retrieve_k = gr.Slider(
                            5, 50, value=settings.default_retrieve_k, step=1, label="Retrieve pool"
                        )
                    example_btn = gr.Button("Run example", variant="primary")
                    example_header = gr.Markdown()
                    example_chart = gr.Plot()
                    example_results = gr.HTML()
                    example_query = gr.Textbox(label="Loaded question", interactive=False)

                    example_split.change(
                        lambda s: gr.Dropdown(choices=self.example_choices(s), value=self.example_choices(s)[0][1]),
                        example_split,
                        example_pick,
                    )
                    example_btn.click(
                        self.example_ui,
                        [example_split, example_pick, example_top_k, example_retrieve_k],
                        [example_header, example_results, example_query, example_chart],
                    )

            refresh_btn = gr.Button("Refresh status")
            refresh_btn.click(self.system_status, outputs=status)
            demo.load(self.system_status, outputs=status)

        return demo
