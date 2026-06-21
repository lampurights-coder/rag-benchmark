import html

MODES = [
    ("Dense", "dense"),
    ("Sparse (lexical)", "sparse"),
    ("Hybrid (dense + sparse)", "hybrid"),
    ("Hybrid + rerank", "hybrid_rerank"),
]

MODE_COLORS = {
    "dense": "#3b82f6",
    "sparse": "#f59e0b",
    "hybrid": "#8b5cf6",
    "hybrid_rerank": "#10b981",
}


def mode_label(mode: str) -> str:
    for label, value in MODES:
        if value == mode:
            return label
    return mode


def mode_value(label: str) -> str:
    return dict(MODES).get(label, "hybrid_rerank")


def score_bar(score: float, color: str = "#3b82f6") -> str:
    width = max(0, min(100, score * 100))
    return (
        f"<div class='score-bar'><div class='score-fill' "
        f"style='width:{width:.1f}%;background:{color};'></div></div>"
    )


def relevance_class(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.35:
        return "mid"
    return "low"


def format_matches_html(matches: list[dict], expected_answer: str | None = None) -> str:
    if not matches:
        return "<p class='empty'>No matches returned. Ingest HotpotQA data first.</p>"

    blocks = []
    for index, match in enumerate(matches, start=1):
        score = float(match.get("score", 0.0))
        retrieval_score = match.get("retrieval_score")
        rerank_score = match.get("rerank_score")
        answer = match.get("answer", "")
        answer_match = expected_answer is not None and answer == expected_answer
        badge = "<span class='badge good'>answer match</span>" if answer_match else ""
        rel_class = relevance_class(rerank_score if rerank_score is not None else score)

        score_bits = [f"<span class='pill {rel_class}'>{score:.3f}</span>"]
        if retrieval_score is not None:
            score_bits.append(f"retrieval {retrieval_score:.3f}")
        if rerank_score is not None:
            score_bits.append(f"rerank {rerank_score:.3f}")

        blocks.append(
            f"""
            <article class='result-card {rel_class}'>
              <header>
                <strong>#{index} {html.escape(match.get('title') or 'Untitled')}</strong>
                <span>{' | '.join(score_bits)}</span>
              </header>
              {score_bar(rerank_score if rerank_score is not None else score, MODE_COLORS.get('hybrid_rerank', '#10b981'))}
              <p><b>Source:</b> {html.escape(str(match.get('source_id', '')))} {badge}</p>
              <p><b>Answer:</b> {html.escape(answer)}</p>
              <p class='snippet'>{html.escape((match.get('text') or '')[:500])}</p>
            </article>
            """
        )
    return "\n".join(blocks)
