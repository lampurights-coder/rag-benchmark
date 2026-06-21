from dataclasses import dataclass


@dataclass
class BenchmarkSample:
    row_index: int
    source_id: str
    question: str
    expected_answer: str
