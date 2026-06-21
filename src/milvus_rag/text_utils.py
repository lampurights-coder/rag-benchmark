def format_document_text(title: str, text: str) -> str:
    title = title.strip()
    if title:
        return f"{title}\n{text}"
    return text


def format_hotpot_chunk_text(question: str, title: str, text: str) -> str:
    passage = format_document_text(title, text)
    question = question.strip()
    if question:
        return f"{question}\n{passage}"
    return passage
